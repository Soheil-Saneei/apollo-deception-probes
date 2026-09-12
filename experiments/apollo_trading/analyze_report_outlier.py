#!/usr/bin/env python3
"""Rank trajectory 62 against disclosure leave-one-out distances, without a model."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import load_file


def distances(x, center):
    if (x.norm(dim=-1) == 0).any() or (center.norm(dim=-1) == 0).any():
        raise ValueError("cosine distance requires nonzero vectors")
    return {"l2": (x - center).norm(dim=-1),
            "cosine_distance": 1 - torch.nn.functional.cosine_similarity(x, center, dim=-1)}


def compare(target, group):
    target, group = target.double(), group.double()
    if group.ndim != 3 or target.shape != group.shape[1:] or len(group) < 2:
        raise ValueError("expected target [layers,width], group [members,layers,width]")
    if not torch.isfinite(target).all() or not torch.isfinite(group).all():
        raise ValueError("nonfinite vectors")
    loo = (group.sum(dim=0, keepdim=True) - group) / (len(group) - 1)
    control = distances(group, loo)
    candidate = distances(target, group.mean(dim=0))
    result = {}
    for name, values in control.items():
        below = (values < candidate[name]).sum(dim=0)
        ties = (values == candidate[name]).sum(dim=0)
        # Ascending average rank when the candidate is inserted among controls.
        result[name] = dict(control=values, target=candidate[name], below=below, ties=ties,
                            rank=1 + below + ties / 2,
                            percentile=100 * (below + ties / 2) / len(group))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Prior comparison output directory")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("use a new output directory")
    path = args.input / "comparison.safetensors"
    data = load_file(str(path))
    source = json.loads((args.input / "manifest.json").read_text())
    group = data["disclosure_individuals"]
    ids = source["disclosure_indices"]
    if len(ids) != len(group) or len(set(ids)) != len(ids) or source["target"] in ids:
        raise ValueError("invalid or overlapping cohort IDs")
    results = compare(data["trajectory_62"], group)
    args.output.mkdir(parents=True)
    n, layers, _ = group.shape
    rows, individuals = [], []
    for metric, result in results.items():
        for layer in range(layers):
            values = result["control"][:, layer]
            q = torch.quantile(values, torch.tensor([0., .25, .5, .75, 1.], dtype=torch.float64))
            rows.append(dict(layer=layer, metric=metric, target_distance=result["target"][layer].item(),
                             disclosure_min=q[0].item(), disclosure_q25=q[1].item(),
                             disclosure_median=q[2].item(), disclosure_q75=q[3].item(),
                             disclosure_max=q[4].item(), percentile=result["percentile"][layer].item(),
                             ascending_rank=result["rank"][layer].item(), rank_denominator=n + 1,
                             disclosures_below=result["below"][layer].item(),
                             exact_ties=result["ties"][layer].item()))
            for i, index in enumerate(ids):
                individuals.append(dict(layer=layer, metric=metric, trajectory=index,
                                        loo_distance=values[i].item()))
    for filename, table in (("layer-ranks.csv", rows), ("disclosure-loo-distances.csv", individuals)):
        with (args.output / filename).open("w") as f:
            writer = csv.DictWriter(f, fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(table)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    x = list(range(layers))
    for ax, (metric, result) in zip(axes[:2], results.items()):
        values = result["control"]
        q = torch.quantile(values, torch.tensor([0., .25, .5, .75, 1.], dtype=torch.float64), dim=0)
        ax.fill_between(x, q[0], q[4], color="#bdd4e7", alpha=.5, label="Disclosure min-max")
        ax.fill_between(x, q[1], q[3], color="#6d9dc4", alpha=.5, label="Disclosure 25-75%")
        ax.plot(x, values.T, color="#48769a", alpha=.16, linewidth=.6)
        ax.plot(x, q[2], color="#255c86", label="Disclosure median")
        ax.plot(x, result["target"], color="#c54e29", linewidth=2, label="Trajectory 62")
        ax.set_ylabel("L2 distance" if metric == "l2" else "Cosine distance (1 - cosine)")
        ax.grid(alpha=.15)
    axes[0].legend(loc="upper left", fontsize=8)
    for metric, result in results.items():
        axes[2].plot(x, result["percentile"], marker=".", label=metric.replace('_', ' '))
    axes[2].set_ylim(-3, 103)
    axes[2].set_ylabel("Trajectory 62 percentile")
    axes[2].set_xlabel("Decoder layer (zero-based)")
    axes[2].legend()
    axes[2].grid(alpha=.2)
    fig.suptitle("Report-boundary residuals: trajectory 62 vs 24 disclosures\n"
                 "Disclosures use leave-one-out means; trajectory 62 uses all 24")
    fig.tight_layout()
    fig.savefig(args.output / "outlier-vs-layer.png", dpi=180)
    plt.close(fig)
    manifest = dict(source_comparison_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    source_manifest_sha256=hashlib.sha256((args.input / "manifest.json").read_bytes()).hexdigest(),
                    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    target=source["target"], disclosure_indices=ids, layers=source["layers"],
                    arithmetic="float64 from existing saved float32 vectors; no model execution",
                    percentile="100 * (number strictly below + 0.5 * exact ties) / 24",
                    rank="Ascending insertion midrank among 25: 1 + below + 0.5 * ties; 25 is farthest",
                    limitations="Descriptive empirical ranks, not p-values. Controls use means of 23; "
                    "target uses a mean of 24. Layers are correlated, cohort is small, and contexts differ.")
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for name, r in results.items():
        print(name, "percentile range", r['percentile'].min().item(), r['percentile'].max().item(),
              "layers beyond control max", (r['target'] > r['control'].max(dim=0).values).nonzero().flatten().tolist())


if __name__ == "__main__":
    main()
