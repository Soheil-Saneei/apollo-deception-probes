"""Frozen, off-policy teacher forcing of Apollo's released roleplaying pairs."""
import argparse
import hashlib
import json
import random
from pathlib import Path

import yaml

REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
SOURCE_COMMIT = "f8ec4010e74927394709dffa22b97bdf8cd5a62f"


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def build_examples(data, seed=42):
    groups = sorted({sha(" ".join(d["scenario"].split())) for d in data})
    random.Random(seed).shuffle(groups)
    n = len(groups)
    splits = {g: "train" if i < int(.7*n) else "validation" if i < int(.85*n)
              else "test" for i, g in enumerate(groups)}
    examples = []
    for i, d in enumerate(data):
        for key in ("scenario", "question", "honest_completion", "deceptive_completion"):
            if not isinstance(d.get(key), str) or not d[key].strip():
                raise ValueError(f"Invalid pair {i}: {key}")
        if d["honest_completion"] == d["deceptive_completion"]:
            raise ValueError(f"Identical completions in pair {i}")
        group = sha(" ".join(d["scenario"].split()))
        for label in ("honest", "deceptive"):
            examples.append(dict(id=f"scenario-{i:03d}-{label}", source_index=i,
                scenario_id=group, split=splits[group], label=label,
                messages=[dict(role="system", content=d["scenario"]),
                          dict(role="user", content=d["question"])],
                completion=d[label+"_completion"],
                completion_sha256=sha(d[label+"_completion"])))
    return examples


def encode(tokenizer, example):
    prefix = tokenizer.apply_chat_template(example["messages"], tokenize=False,
        add_generation_prompt=True, enable_thinking=False)
    completion = example["completion"]
    encoded = tokenizer(prefix + completion, add_special_tokens=False,
                        return_offsets_mapping=True)
    offsets = encoded.pop("offset_mapping")
    positions = [i for i, (a, b) in enumerate(offsets) if b > len(prefix)]
    if not positions or offsets[positions[0]][0] < len(prefix):
        raise ValueError("Completion boundary straddles a token")
    ids = encoded["input_ids"]
    if tokenizer.decode(ids[positions[0]:], skip_special_tokens=False,
                        clean_up_tokenization_spaces=False) != completion:
        raise ValueError("Completion token roundtrip failed")
    return ids, positions, prefix


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--source", type=Path, default=Path(__file__).parent/"source/dataset.yaml")
    p.add_argument("--cache-dir", default=".hf-cache")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    from collections import Counter
    import torch
    import transformers
    from safetensors.torch import save_file
    from transformers import AutoTokenizer, AutoModelForImageTextToText
    model_id = "Qwen/Qwen3.5-9B"
    examples = build_examples(yaml.safe_load(args.source.read_text()))
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=REVISION,
                                               cache_dir=args.cache_dir)
    encoded = [encode(tokenizer, e) for e in examples]
    config = dict(model=model_id, revision=REVISION, source_commit=SOURCE_COMMIT,
        source_sha256=hashlib.sha256(args.source.read_bytes()).hexdigest(),
        seed=42, dtype="bfloat16", attention="eager", thinking=False,
        torch=torch.__version__, transformers=transformers.__version__,
        splits=dict(Counter(e["split"] for e in examples)),
        examples=len(examples), max_sequence_tokens=max(len(x[0]) for x in encoded),
        activation="decoder block output, before final model normalization",
        padding="right padding to the longer sequence within each pair",
        pooling="mean over completion tokens only; no end-of-turn token",
        original_provider_token_ids_available=False)
    (args.output/"config.json").write_text(json.dumps(config, indent=2))
    with (args.output/"manifest.jsonl").open("w") as f:
        for e, (ids, positions, prefix) in zip(examples, encoded):
            e.update(input_ids=ids, completion_positions=positions, rendered_prefix=prefix,
                     activation_path=f"activations/{e['id']}.safetensors")
            f.write(json.dumps(e)+"\n")
    print(json.dumps(config), flush=True)
    model = AutoModelForImageTextToText.from_pretrained(model_id, revision=REVISION,
        cache_dir=args.cache_dir, dtype=torch.bfloat16, device_map="cuda",
        attn_implementation="eager").eval().requires_grad_(False)
    layers = model.model.language_model.layers
    (args.output/"activations").mkdir()
    captured = {}
    start = 0
    stop = 0
    def hook(i):
        def capture(module, inputs, output):
            h = output[0] if isinstance(output, tuple) else output
            captured[i] = h[0, start:stop].detach().cpu().contiguous()
        return capture
    handles = [layer.register_forward_hook(hook(i)) for i, layer in enumerate(layers)]
    means, boundaries = [], []
    fidelity = []
    with torch.inference_mode():
        for i, (e, (ids, positions, prefix)) in enumerate(zip(examples, encoded)):
            # Capture the shared prompt boundary as a causal/no-label-leakage check.
            start = positions[0]-1
            stop = len(ids)
            pair_length = max(len(encoded[i//2*2][0]), len(encoded[i//2*2+1][0]))
            inputs = torch.tensor([ids + [tokenizer.pad_token_id]*(pair_length-stop)], device="cuda")
            attention_mask = torch.tensor([[1]*stop + [0]*(pair_length-stop)], device="cuda")
            result = model(input_ids=inputs, attention_mask=attention_mask,
                           use_cache=False, logits_to_keep=1)
            if i == 0:
                for h in handles:
                    h.remove()
                reference = model(input_ids=inputs, attention_mask=attention_mask,
                                  use_cache=False, logits_to_keep=1)
                error = (result.logits-reference.logits).abs().max().item()
                if error != 0:
                    raise ValueError(f"Hooks changed logits: {error}")
                fidelity.append(dict(check="hooks_vs_no_hooks_max_logit_error", value=error))
                handles = [layer.register_forward_hook(hook(j)) for j, layer in enumerate(layers)]
            values = torch.stack([captured[j] for j in range(len(layers))])
            boundary, completion_values = values[:, 0], values[:, 1:]
            means.append(completion_values.float().mean(dim=1))
            boundaries.append(boundary.float())
            save_file(dict(resid_post=completion_values.contiguous(),
                           prompt_boundary=boundary.contiguous()),
                str(args.output/e["activation_path"]),
                metadata=dict(example_id=e["id"], axes="layer,completion_token,hidden"))
            if i % 20 == 0 or i == len(examples)-1:
                print(json.dumps(dict(completed=i+1, total=len(examples))), flush=True)
    for h in handles:
        h.remove()
    boundary = torch.stack(boundaries)
    pair_error = (boundary[::2]-boundary[1::2]).abs().max().item()
    fidelity.append(dict(check="paired_prompt_boundary_max_abs_difference", value=pair_error))
    save_file(dict(mean_resid_post=torch.stack(means), prompt_boundary=boundary),
              str(args.output/"features.safetensors"))
    (args.output/"fidelity.json").write_text(json.dumps(fidelity, indent=2))
    if pair_error != 0:
        raise ValueError("Paired prompt activations differ; inspect fidelity.json before probing")
    print("Extraction complete", flush=True)


if __name__ == "__main__":
    main()
