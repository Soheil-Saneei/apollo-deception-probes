"""Replay Apollo's released report cohort through frozen FP16 Llama 3.3 70B.

Default: validate four pilot trajectories, then process all 1,000 without reloading.
This reconstructs analysis inputs from text, not original provider generation tokens.
"""
import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
from pathlib import Path
import random
import shutil
import time

MODEL = "meta-llama/Llama-3.3-70B-Instruct"
TOKENIZER = "meta-llama/Meta-Llama-3.1-70B-Instruct"
DATE = "26 Jul 2024"


def read(path):
    return json.loads(Path(path).read_text())


def write(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2) + "\n")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_cohort(root):
    rows = []
    for label in ("honest", "deceptive"):
        for line in (root / "reports" / f"{label}.jsonl").read_text().splitlines():
            r = json.loads(line)
            assert r["included"] and r["label"] == label and r["stage"] == "reports"
            assert r["report_label"] == {"honest": "explicit", "deceptive": "concealed"}[label]
            assert r["trade_label"] == "yes" and not r["exclusion_reasons"]
            assert digest(root / r["transcript_path"]) == r["transcript_sha256"]
            d = read(root / r["transcript_path"])
            assert all(d[k] == r[k] for k in ("trajectory_id", "raw_index", "apollo_loader_id", "label"))
            r = dict(r, messages=d["apollo_normalized_messages"])
            r.pop("activation_replay_status")
            rows.append(r)
    assert Counter(r["label"] for r in rows) == {"honest": 666, "deceptive": 334}
    assert len({r["raw_index"] for r in rows}) == len(rows)
    return sorted(rows, key=lambda r: r["raw_index"])


def assign_splits(rows):
    """Stratify unique trajectory groups; never split tokens/stages independently."""
    assignments = {}
    rng = random.Random(42)
    for label in ("honest", "deceptive"):
        groups = sorted({r["raw_index"] for r in rows if r["label"] == label})
        rng.shuffle(groups)
        for i, group in enumerate(groups):
            if group in assignments:
                raise ValueError("A trajectory has inconsistent labels")
            assignments[group] = "test" if i < round(.2 * len(groups)) else "train"
    for r in rows:
        r["split"] = assignments[r["raw_index"]]
        r["group_id"] = f"llama33-raw-{r['raw_index']:06d}"


def span_tokens(offsets, start, end):
    positions = [i for i, (a, b) in enumerate(offsets) if b > start and a < end]
    if not positions or positions != list(range(positions[0], positions[-1]+1)):
        raise ValueError("Empty or noncontiguous token span")
    return positions


def encode_report(tokenizer, row):
    messages = row["messages"]
    assert messages[-1]["role"] == "assistant"
    body = messages[-1]["content"]
    text = tokenizer.apply_chat_template(messages, tokenize=False,
        add_generation_prompt=False, date_string=DATE)
    prefix = tokenizer.apply_chat_template(messages[:-1], tokenize=False,
        add_generation_prompt=True, date_string=DATE)
    if text != prefix + body + "<|eot_id|>":
        raise ValueError("Unexpected analysis chat-template reconstruction")
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = enc["input_ids"], enc["offset_mapping"]
    if tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False) != text:
        raise ValueError("Full transcript token roundtrip failed")
    body_start, body_end = len(prefix), len(prefix)+len(body)
    action_start = body_start + body.rindex("Action:")
    # The email's first substantive character, not the report reasoning or tool name.
    report_start = body_start + body.index("Executed trade update", body.rindex("Action:"))
    spans = {name: span_tokens(offsets, start, body_end) for name, start in
             (("assistant", body_start), ("action", action_start), ("report", report_start))}
    first = spans["assistant"][0]
    if offsets[first][0] != body_start or offsets[spans["assistant"][-1]][1] != body_end:
        raise ValueError("Assistant text shares token with template boundary")
    if tokenizer.decode(ids[first:spans["assistant"][-1]+1],
        clean_up_tokenization_spaces=False) != body:
        raise ValueError("Assistant report roundtrip failed")
    return dict(row, input_ids=ids, token_positions=spans,
        boundary_positions={k: v[0]-1 for k, v in spans.items()},
        char_spans={"assistant": [body_start, body_end], "action": [action_start, body_end],
                    "report": [report_start, body_end]},
        boundary_token_offsets={k: offsets[v[0]] for k, v in spans.items()},
        rendered_text_sha256=hashlib.sha256(text.encode()).hexdigest())


def check_released_tokens(tokenizer, root):
    """Use released HTML token strings as independent tokenizer checks."""
    from build_manifest import display_render, normalized_messages
    raw = read(root / "source/data/insider_trading/llama-70b-3.3-generations.json")
    checks = []
    for path in sorted((root / "overlap_examples").glob("*.json")):
        obj = read(path)
        if obj["stage"] != "reports" or not obj["matching_raw_indices"]:
            continue
        i = obj["matching_raw_indices"][0]
        d = raw[i]
        messages = d["transcript"][:-2] if "doubling_down_label" in d["metadata"] else d["transcript"]
        expected = display_render(messages, DATE)
        ids = tokenizer(expected, add_special_tokens=False)["input_ids"]
        decoded = []
        for t in ids:
            value = (tokenizer.decode([891, t])[3:] if "llama" in tokenizer.__class__.__name__.lower()
                     else tokenizer.decode(t))
            decoded.append(value)
        template_text = tokenizer.apply_chat_template(
            # Apollo strips each message and merges adjacent same-role messages.
            normalized_messages(messages), tokenize=False,
            add_generation_prompt=False, date_string=DATE)
        result = dict(example=path.name, raw_index=i, count=len(ids),
                      template_text_match=template_text == expected,
                      token_strings_match=decoded == obj["released_token_strings"])
        checks.append(result)
    if not checks or not all(c["template_text_match"] and c["token_strings_match"] for c in checks):
        raise ValueError(f"Released token-string validation failed: {checks}")
    return checks


class ResidualCapture:
    """Unnormalized output of every decoder block, with unchanged outputs."""
    def __init__(self, model):
        self.values = {}
        self.positions = []
        self.handles = [layer.register_forward_hook(self.hook(i))
                        for i, layer in enumerate(model.model.layers)]

    def hook(self, i):
        def capture(module, args, output):
            h = output[0] if isinstance(output, tuple) else output
            self.values[i] = h[0, self.positions].detach().cpu().contiguous()
        return capture

    def stack(self):
        import torch
        return torch.stack([self.values[i] for i in range(len(self.handles))])

    def close(self):
        for h in self.handles:
            h.remove()


def forward(model, ids):
    import torch
    device = model.get_input_embeddings().weight.device
    x = torch.tensor([ids], device=device)
    return model(input_ids=x, attention_mask=torch.ones_like(x), use_cache=False,
                 logits_to_keep=1).logits.detach().float().cpu()


def extract(model, capture, row):
    import torch
    capture.values.clear()
    first = row["boundary_positions"]["assistant"]
    last = row["token_positions"]["assistant"][-1]
    capture.positions = list(range(first, last+1))
    logits = forward(model, row["input_ids"])
    values = capture.stack()
    if not torch.isfinite(values).all() or not torch.isfinite(logits).all():
        raise ValueError("Nonfinite activations or logits")
    features = {}
    for name in ("assistant", "action", "report"):
        idx = row["boundary_positions"][name]-first
        features[name+"_boundary"] = values[:, idx].float().contiguous()
        idx = row["token_positions"][name][0]-first
        features[name+"_mean"] = values[:, idx:].float().mean(1)
    return values[:, 1:].contiguous(), features, logits


def pilot_check(model, capture, rows, output):
    import torch
    from safetensors.torch import save_file
    output.mkdir()
    checks = []
    for row in rows:
        values, features, logits = extract(model, capture, row)
        _, repeated, repeated_logits = extract(model, capture, row)
        assert torch.equal(logits, repeated_logits), "Repeated forward logits differ"
        assert all(torch.equal(features[k], repeated[k]) for k in features), "Repeated features differ"
        boundary = row["boundary_positions"]["report"]
        capture.positions = [boundary]
        forward(model, row["input_ids"][:boundary+1])
        prefix = capture.stack()[:, 0].float()
        target = features["report_boundary"]
        relative = (prefix-target).norm(dim=1)/target.norm(dim=1).clamp_min(1e-8)
        # FP16 full/prefix GEMM shapes can differ. Gate on per-layer relative L2.
        if relative.max().item() > .005:
            raise ValueError(f"Full/prefix relative L2 exceeds 0.5%: {relative.max().item()}")
        save_file(dict(resid_post=values, **features), str(output/(row["trajectory_id"]+".safetensors")))
        checks.append(dict(trajectory_id=row["trajectory_id"], label=row["label"],
            repeat_logits_max_abs=0., repeat_features_max_abs=0.,
            full_prefix_relative_l2_per_layer=relative.tolist(),
            full_prefix_max_abs=(prefix-target).abs().max().item(),
            activation_shape=list(values.shape)))
    # Confirm hooks have no effect on the model's output.
    row = rows[0]
    capture.positions = [row["boundary_positions"]["report"]]
    baseline = forward(model, row["input_ids"])
    capture.close()
    unhooked = forward(model, row["input_ids"])
    assert torch.equal(baseline, unhooked), "Hooks changed model logits"
    write(output/"fidelity.json", dict(passed=True, hooks_max_logit_error=0., checks=checks))
    return ResidualCapture(model)


def main():
    import torch
    from huggingface_hub import HfApi, snapshot_download
    from safetensors.torch import save_file
    from transformers import AutoModelForCausalLM, AutoTokenizer
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=Path(__file__).parent/"apollo-released-llama33-manifest")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--cache-dir", default=".hf-cache-llama")
    p.add_argument("--tokenizer", default=TOKENIZER,
                   help="Default is Apollo's 3.1 tokenizer; any override is recorded in config")
    p.add_argument("--pilot-only", action="store_true")
    args = p.parse_args()
    if args.output.exists():
        p.error("Use a fresh output directory")
    rows = load_cohort(args.data)
    assign_splits(rows)
    api = HfApi()
    model_revision = api.model_info(MODEL).sha
    tokenizer_revision = api.model_info(args.tokenizer).sha
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, revision=tokenizer_revision,
                                              cache_dir=args.cache_dir)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.bos_token
    overlap = check_released_tokens(tokenizer, args.data)
    encoded = [encode_report(tokenizer, row) for row in rows]
    args.output.mkdir(parents=True)
    estimated = sum(len(r["token_positions"]["assistant"]) for r in encoded)*80*8192*2
    config = dict(model=MODEL, model_revision=model_revision, tokenizer=args.tokenizer,
        tokenizer_revision=tokenizer_revision, source_provenance=read(args.data/"provenance.json"),
        packages={k: importlib.metadata.version(k) for k in
                  ("torch", "transformers", "accelerate", "huggingface-hub", "safetensors", "tokenizers")},
        dtype="float16", quantization=None, attention="eager", frozen=True,
        use_cache=False, template_date=DATE, seed=42,
        split_counts=dict(Counter(r["split"]+"/"+r["label"] for r in encoded)),
        max_input_tokens=max(len(r["input_ids"]) for r in encoded),
        estimated_token_activation_bytes=estimated,
        residual="unnormalized decoder block output; zero-based layer index",
        report_boundary="last token before token overlapping Executed trade update",
        report_mean="tokens overlapping Executed trade update through assistant content end; excludes EOT",
        provider_replay_exact=False, original_generation_token_ids_available=False)
    write(args.output/"config.json", config)
    write(args.output/"tokenizer_fidelity.json", overlap)
    (args.output/"planned_manifest.jsonl").write_text("".join(json.dumps(r)+"\n" for r in encoded))
    print(json.dumps({k: v for k, v in config.items() if k != "source_provenance"}), flush=True)
    if shutil.disk_usage(args.output).free < estimated + 160_000_000_000:
        raise ValueError("Insufficient disk for weights, token activations, and working space")
    snapshot_download(MODEL, revision=model_revision, cache_dir=args.cache_dir,
        allow_patterns=["*.safetensors", "*.json", "*.jinja"], max_workers=8)
    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=model_revision,
        cache_dir=args.cache_dir, torch_dtype=torch.float16, device_map="balanced",
        max_memory={i: "43GiB" for i in range(torch.cuda.device_count())},
        attn_implementation="eager").eval().requires_grad_(False)
    assert len(model.model.layers) == 80 and model.config.hidden_size == 8192
    assert not any(p.requires_grad for p in model.parameters())
    assert all(p.device.type == "cuda" for p in model.parameters()), "Unexpected CPU/offload placement"
    write(args.output/"device_map.json", {k: str(v) for k, v in model.hf_device_map.items()})
    capture = ResidualCapture(model)
    pilots = ([r for r in encoded if r["label"] == "honest"][:2]
              + [r for r in encoded if r["label"] == "deceptive"][:2])
    with torch.inference_mode():
        capture = pilot_check(model, capture, pilots, args.output/"pilot")
        print("PILOT PASSED: scaling to full cohort" if not args.pilot_only else "PILOT PASSED", flush=True)
        if args.pilot_only:
            capture.close()
            return
        (args.output/"activations").mkdir()
        all_features = {}
        t0 = time.monotonic()
        with (args.output/"manifest.jsonl").open("w") as manifest:
            for i, row in enumerate(encoded):
                values, features, _ = extract(model, capture, row)
                path = Path("activations")/row["label"]/(row["trajectory_id"]+".safetensors")
                (args.output/path).parent.mkdir(exist_ok=True)
                tmp = (args.output/path).with_suffix(".tmp")
                save_file(dict(resid_post=values, **features), str(tmp), metadata=dict(
                    trajectory_id=row["trajectory_id"], label=row["label"], split=row["split"],
                    axes="layer,final_assistant_token,hidden"))
                tmp.replace(args.output/path)
                manifest.write(json.dumps(dict(row, activation_path=str(path),
                    activation_replay_status="completed"))+"\n")
                manifest.flush()
                for name, value in features.items():
                    all_features.setdefault(name, []).append(value)
                if i % 10 == 0 or i == len(encoded)-1:
                    print(json.dumps(dict(completed=i+1, total=len(encoded),
                        elapsed_seconds=round(time.monotonic()-t0, 1))), flush=True)
        capture.close()
        tensors = {k: torch.stack(v) for k, v in all_features.items()}
        tensors["labels"] = torch.tensor([int(r["label"] == "deceptive") for r in encoded])
        tensors["train_mask"] = torch.tensor([r["split"] == "train" for r in encoded])
        tensors["raw_indices"] = torch.tensor([r["raw_index"] for r in encoded])
        save_file(tensors, str(args.output/"features.safetensors"), metadata=dict(
            row_order="manifest.jsonl", label_encoding="0=honest,1=deceptive"))
    write(args.output/"summary.json", dict(completed=1000, honest=666, deceptive=334,
        layers=80, hidden_size=8192, pilot_passed=True,
        split_counts=config["split_counts"], feature_shape=[1000, 80, 8192]))
    print("FULL EXTRACTION COMPLETE", flush=True)


if __name__ == "__main__":
    main()
