#!/usr/bin/env python3
"""Generate new Apollo v4 trajectories directly in PyTorch, recording activations."""

import argparse
import copy
import hashlib
import importlib.metadata
import json
import platform
import re
import time
from pathlib import Path

from run import SOURCE, save
from run_interactive import INSTRUCTIONS, SIMULATION, TOOLS, trajectory


def parse_qwen_output(raw):
    """Decode Qwen's documented tool wire format; never execute generated code.

    Preserve every call so the shared runner can reject multiple calls. Invalid
    wire syntax is retained as text with a parse error, not silently repaired.
    """
    message = {"role": "assistant", "content": raw}
    if "<tool_call>" not in raw:
        return message, None
    try:
        start = raw.index("<tool_call>")
        blocks = list(re.finditer(r"<tool_call>(.*?)</tool_call>", raw[start:], re.S))
        if not blocks or re.sub(r"<tool_call>.*?</tool_call>", "", raw[start:], flags=re.S).strip():
            raise ValueError("incomplete tool block or trailing text")
        calls = []
        for block in blocks:
            function = re.fullmatch(r"\s*<function=(\w+)>(.*?)</function>\s*", block[1], re.S)
            if not function:
                raise ValueError("malformed function block")
            body = function[2]
            parameters = list(re.finditer(r"<parameter=(\w+)>(.*?)</parameter>", body, re.S))
            if re.sub(r"<parameter=\w+>.*?</parameter>", "", body, flags=re.S).strip():
                raise ValueError("malformed parameter block")
            arguments = {}
            for parameter in parameters:
                name, value = parameter[1], parameter[2].strip()
                if name in arguments:
                    raise ValueError("duplicate parameter")
                # Qwen's template writes strings unquoted; only shares is an integer.
                if name == "shares" and re.fullmatch(r"-?\d+", value):
                    value = int(value)
                arguments[name] = value
            calls.append({"function": {"name": function[1], "arguments": arguments}})
        message.update(content=raw[:start].strip(), tool_calls=calls)
        return message, None
    except ValueError as error:
        return message, str(error)


class PresencePenalty:
    """Additive penalty for unique tokens in the most recent context window."""

    def __init__(self, penalty=1.5, window=64):
        self.penalty, self.window = penalty, window

    def __call__(self, input_ids, scores):
        if self.penalty and self.window:
            for row in range(input_ids.shape[0]):
                ids = input_ids[row, -self.window:].unique()
                scores[row, ids] -= self.penalty
        return scores


class ActivationRecorder:
    """Capture actual generation forwards without modifying model outputs.

    At each forward, record the last sequence position: the token representation
    used to predict the next generated token. With KV caching, the first is the
    last prompt token, followed by previously generated tokens. No replay required.
    """

    def __init__(self, model, max_steps=32):
        self.model, self.max_steps = model, max_steps
        self.handles, self.values, self.positions, self.modules = [], {}, [], {}
        self.step = -1
        self.position = -1

    def _begin(self, module, args, kwargs):
        ids = kwargs.get("input_ids", args[0] if args else None)
        if ids is None or ids.shape[0] != 1:
            raise ValueError("activation capture requires one token-ID sequence")
        self.step += 1
        self.position += ids.shape[1]
        if not self.max_steps or self.step < self.max_steps:
            self.positions.append(self.position)

    def _store(self, key, tensor):
        if not self.max_steps or self.step < self.max_steps:
            if isinstance(tensor, tuple):
                tensor = tensor[0]
            self.values.setdefault(key, []).append(tensor[0, -1].detach().cpu().clone())

    def __enter__(self):
        self.handles.append(self.model.register_forward_pre_hook(self._begin, with_kwargs=True))
        for name, layer in self.model.named_modules():
            if layer.__class__.__name__ != "Qwen3_5DecoderLayer":
                continue
            self.modules[name] = layer.block_type
            for suffix, module, before in (
                ("resid_pre", layer, True),
                ("resid_mid", layer.post_attention_layernorm, True),
                ("resid_post", layer, False),
                ("mlp_out", layer.mlp, False),
                ("mixer_out", getattr(layer, "linear_attn", getattr(layer, "self_attn", None)), False),
            ):
                key = name + "." + suffix
                if before:
                    def hook(module, args, kwargs, key=key):
                        tensor = kwargs["hidden_states"] if "hidden_states" in kwargs else args[0]
                        self._store(key, tensor)
                    self.handles.append(module.register_forward_pre_hook(hook, with_kwargs=True))
                else:
                    def hook(module, args, output, key=key):
                        self._store(key, output)
                    self.handles.append(module.register_forward_hook(hook))
        if not self.modules:
            self.__exit__(None, None, None)
            raise ValueError("no Qwen3.5 decoder layers found")
        return self

    def __exit__(self, *args):
        for handle in self.handles:
            handle.remove()

    def write(self, stem):
        import torch
        from safetensors.torch import save_file

        tensors = {key: torch.stack(values).contiguous() for key, values in self.values.items()}
        path = Path(str(stem) + "-activations.safetensors")
        if path.exists():
            raise FileExistsError(path)
        save_file(tensors, str(path))
        save(Path(str(stem) + "-activations.json"), {
            "file": path.name, "modules": self.modules,
            "positions_in_prompt_plus_generated": self.positions,
            "alignment": "Each row is a residual/intermediate state at position p used "
                         "to predict token p+1. First row is the last prompt token. "
                         "The final emitted token has no forward state unless fed back. "
                         "MPS deferred stopping may run an extra forward on that token; "
                         "its next-token prediction is discarded, not emitted.",
            "max_generation_forwards_captured": self.max_steps,
            "total_generation_forwards": self.step + 1,
            "shapes": {key: list(value.shape) for key, value in tensors.items()},
            "definitions": {
                "resid_pre": "Decoder input, before input normalization.",
                "mixer_out": "Projected full/linear attention contribution, before residual addition.",
                "resid_mid": "After mixer residual addition, before MLP normalization.",
                "mlp_out": "MLP output before residual addition.",
                "resid_post": "Decoder output after MLP residual addition, before next layer/final norm.",
            },
        })


class TransformersBackend:
    def __init__(self, model, tokenizer, output, presence_penalty=1.5, capture_steps=32):
        self.model, self.tokenizer, self.output = model, tokenizer, output
        self.presence_penalty, self.capture_steps = presence_penalty, capture_steps

    def infer(self, payload, index, turn, attempt):
        import torch
        from transformers import GenerationConfig, LogitsProcessorList, set_seed

        stem = self.output / f"trajectory-{index:02d}-turn-{turn:02d}-attempt-{attempt:02d}"
        save(Path(str(stem) + "-request.json"), payload)
        prompt = self.tokenizer.apply_chat_template(
            payload["messages"], tools=payload["tools"], tokenize=False,
            add_generation_prompt=True, enable_thinking=False)
        with Path(str(stem) + "-prompt.txt").open("x") as f:
            f.write(prompt)
        inputs = self.tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
        options = payload["options"]
        if inputs.input_ids.shape[1] + options["num_predict"] > options["num_ctx"]:
            raise ValueError("context limit exceeded; refusing silent prompt truncation")
        inputs = inputs.to(next(self.model.parameters()).device)
        model_eos = self.model.generation_config.eos_token_id
        stop_ids = model_eos if isinstance(model_eos, list) else [model_eos]
        stop_ids = sorted({x for x in stop_ids + [self.tokenizer.eos_token_id] if x is not None})
        if not stop_ids:
            raise ValueError("model/tokenizer have no end-of-message token")
        config = GenerationConfig(
            do_sample=True, temperature=options["temperature"], top_p=options["top_p"],
            top_k=options["top_k"], repetition_penalty=options["repeat_penalty"],
            max_new_tokens=options["num_predict"], use_cache=True,
            eos_token_id=stop_ids,
            pad_token_id=self.tokenizer.pad_token_id)
        save(Path(str(stem) + "-generation-config.json"), config.to_dict())
        set_seed(options["seed"])
        started = time.monotonic()
        with torch.inference_mode(), ActivationRecorder(self.model, self.capture_steps) as recorder:
            sequence = self.model.generate(
                **inputs, generation_config=config,
                logits_processor=LogitsProcessorList([PresencePenalty(self.presence_penalty)]))
        tokens = sequence[0].tolist()
        prompt_length = inputs.input_ids.shape[1]
        generated = tokens[prompt_length:]
        eos = config.eos_token_id
        eos = eos if isinstance(eos, list) else [eos]
        completed = bool(generated and generated[-1] in eos)
        raw = self.tokenizer.decode(generated[:-1] if completed else generated, skip_special_tokens=False)
        message, parse_error = parse_qwen_output(raw)
        result = {"message": message, "done": True, "done_reason": "stop" if completed else "length",
                  "eval_count": len(generated), "generation_seconds": time.monotonic() - started,
                  "raw_output": raw, "wire_parse_error": parse_error}
        save(Path(str(stem) + "-tokens.json"), {
            "prompt_length": prompt_length, "input_ids": tokens,
            "generated_token_ids": generated, "eos_emitted": completed})
        save(Path(str(stem) + "-response.json"), result)
        recorder.write(stem)
        return result


def main():
    import torch
    import transformers
    import accelerate
    from huggingface_hub import HfApi
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--device", choices=("cuda", "mps", "cpu"), default="cuda")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--max-turns", type=int, default=8)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--num-predict", type=int, default=2048)
    parser.add_argument("--capture-steps", type=int, default=32, help="First N generation forwards; 0 captures all")
    parser.add_argument("--presence-penalty", type=float, default=1.5)
    parser.add_argument("--cache-dir", type=Path, default=Path(".hf-cache"))
    args = parser.parse_args()
    if min(args.count, args.max_turns, args.num_predict) < 1 or min(args.start, args.max_retries, args.capture_steps) < 0:
        parser.error("counts must be positive; start, retries and capture steps nonnegative")
    if args.output.exists():
        parser.error("use a fresh output directory")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable. The 9B BF16 model needs a larger machine than the 16 GB Mac.")
    if args.device == "mps" and not torch.backends.mps.is_available():
        parser.error("MPS unavailable")
    args.output.mkdir(parents=True)
    try:
        revision = HfApi().model_info(args.model, revision=args.revision).sha
        config = json.loads((SOURCE / "default.json").read_text())
        options = {"temperature": 1.0, "top_p": 1.0, "top_k": 0, "repeat_penalty": 1.0,
                   "num_predict": args.num_predict, "num_ctx": 16384}
        save(args.output / "protocol.json", {
            "experiment": "apollo-interactive-v4-transformers-v1", "mode": "structured",
            "model": args.model, "revision": revision, "device": args.device, "dtype": args.dtype,
            "torch": torch.__version__, "transformers": transformers.__version__,
            "accelerate": accelerate.__version__, "platform": platform.platform(),
            "installed_packages": {d.metadata["Name"]: d.version
                                   for d in importlib.metadata.distributions()
                                   if d.metadata["Name"]},
            "options": options, "count": args.count, "start": args.start,
            "max_turns": args.max_turns, "max_retries": args.max_retries,
            "seed_rule": "1000 + trajectory index, reset on every continuation/correction",
            "instructions": INSTRUCTIONS["structured"], "tools": TOOLS, "simulation": SIMULATION,
            "capture_steps": args.capture_steps,
            "presence_penalty": args.presence_penalty, "presence_window": 64,
            "presence_note": "Ollama model metadata inherited 1.5; HF applies additive penalty "
            "to unique IDs in last 64 context tokens including prompt. Effective equivalence "
            "to the Ollama engine is not established.",
            "backend_differences": "Unquantized HF checkpoint and official HF chat template; "
            "no exact numeric/tokenization/sampling equivalence to Ollama Q4_K_M asserted. "
            "Tool XML is decoded into the shared runner's structured calls. No inference API "
            "retries for local failures; failures retained with api_error legacy status.",
            "source_sha256": hashlib.sha256((SOURCE / "default.json").read_bytes()).hexdigest(),
            "code_sha256": {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                            for name in ("run_transformers.py", "run_interactive.py", "run.py")},
        })
        save(args.output / "source-config.json", config)
        tokenizer = AutoTokenizer.from_pretrained(args.model, revision=revision, cache_dir=args.cache_dir)
        model = AutoModelForImageTextToText.from_pretrained(
            args.model, revision=revision, cache_dir=args.cache_dir,
            dtype=getattr(torch, args.dtype), device_map=args.device, attn_implementation="eager")
        model.eval()
        with (args.output / "chat_template.jinja").open("x") as f:
            f.write(tokenizer.chat_template)
        save(args.output / "model-config.json", model.config.to_dict())
        save(args.output / "loaded-model.json", {
            "parameter_dtypes": sorted({str(p.dtype) for p in model.parameters()}),
            "parameter_devices": sorted({str(p.device) for p in model.parameters()}),
            "parameter_count": sum(p.numel() for p in model.parameters()),
        })
        backend = TransformersBackend(model, tokenizer, args.output, args.presence_penalty, args.capture_steps)
        for index in range(args.start, args.start + args.count):
            started = time.strftime("%Y-%m-%dT%H:%M:%S%z")
            record = trajectory(config, "structured", lambda payload, turn, attempt:
                                backend.infer(payload, index, turn, attempt), options,
                                1000 + index, args.max_turns, args.max_retries)
            record.update(index=index, seed=1000 + index, started_at=started,
                          finished_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
            save(args.output / f"trajectory-{index:02d}.json", record)
            print(json.dumps({"index": index, "status": record["status"],
                              "trades": record["state"]["trades"]}), flush=True)
            if record["status"] == "api_error":
                raise RuntimeError("local inference failed; details retained in trajectory")
    except Exception as error:
        save(args.output / "run-error.json", {"error": f"{type(error).__name__}: {error}"})
        raise


if __name__ == "__main__":
    main()
