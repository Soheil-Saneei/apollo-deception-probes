"""Render write-up figures from saved evaluations; no fitting or model inference."""
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/apollo-publication-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
SOURCES = {}
DATA = {}
BLUE, ORANGE, GREEN, PURPLE, GRAY = "#0072B2", "#D55E00", "#008060", "#8561A8", "#707780"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 14, "axes.labelsize": 16,
    "axes.titlesize": 18, "xtick.labelsize": 13, "ytick.labelsize": 13,
    "legend.fontsize": 13, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#7E8790", "axes.linewidth": .8,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300,
})


def read(path):
    p = ROOT / path
    SOURCES[path] = hashlib.sha256(p.read_bytes()).hexdigest()
    return json.loads(p.read_text())


def frame(title, subtitle, size=(12, 7)):
    fig = plt.figure(figsize=size, facecolor="white")
    fig.text(.085, .95, title, fontsize=23, weight="bold", va="top")
    fig.text(.085, .892, subtitle, fontsize=13, color="#48525C", va="top")
    return fig


def grid(ax):
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#E2E6EA", linewidth=.8)


def save(fig, name, note):
    fig.text(.085, .035, note, fontsize=11, color="#48525C", va="bottom", linespacing=1.5)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", facecolor="white")
    plt.close(fig)


rp = "experiments/apollo_roleplaying/outputs/"
tr = "experiments/apollo_released/outputs/"
qrole = read(rp + "qwen9b-paired-v2/probe_metrics.json")
lrole = read(rp + "llama33-roleplaying-eval-v1/metrics.json")
trading = read(tr + "qwen35-llama-trading-eval-v1/comparison.json")
prefix = read(tr + "qwen35-llama-trading-eval-v1/early_full_prefix.json")
initial = read(tr + "llama33-probe-eval-v3/metrics.json")

# Qualitative review counts, not Apollo classifier labels or a prevalence estimate.
names = ["Text baseline*", "Text rerun*", "Ollama v2", "Ollama v3", "Ollama v4", "HF A100 v2"]
buys, clear = [20, 20, 28, 30, 30, 29], [0, 0, 1, 0, 1, 1]
DATA["behavior"] = {"runs": names, "n_each": 100, "buy_count": buys, "clear_cases_found": clear}
for name in ["baseline-100-review", "qwen-n100-v2-review", "qwen-n100-v3-review", "qwen-n100-v4-review", "a100-n100-v2-review"]:
    p = ROOT / "experiments/apollo_trading" / f"{name}.md"
    SOURCES[str(p.relative_to(ROOT))] = hashlib.sha256(p.read_bytes()).hexdigest()
fig = frame("Qwen trading decisions and clear deception", "Six 100-trajectory runs | New accepted buys and clear concealment cases identified")
ax = fig.add_axes([.085, .25, .88, .54])
x = np.arange(6)
for off, vals, col, label in [(-.19, buys, BLUE, "Bought"), (.19, clear, ORANGE, "Clear deception found")]:
    bars = ax.bar(x + off, vals, .34, color=col, label=label)
    ax.bar_label(bars, labels=[f"{v}%" for v in vals], padding=6, fontsize=14, weight="bold")
ax.set(ylim=(0, 36), ylabel="Percentage of all trajectories", xticks=x, xticklabels=names)
ax.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
ax.legend(loc="upper left", frameon=False, ncol=2)
grid(ax)
save(fig, "01_qwen_trading_behavior", "Clear cases are review-based lower bounds; unresolved and incomplete cases are not labeled honest.\n*Identical generated texts, not independent evidence. Ollama v2's clear case is a sell. Protocols differ across runs.")

fig = frame("Roleplaying probes across model depth", "Same 371 paired scenarios | Test: 56 held-out scenarios, 112 completions")
ax = fig.add_axes([.085, .24, .88, .57])
for name, d, col, style, key, selected in [
    ("Qwen3.5-9B", qrole, BLUE, "-", "test_auc", qrole["selected_by_validation"]),
    ("Llama 3.3-70B", lrole, ORANGE, "--", "auroc", lrole["activation"]),
]:
    rows = d["layers"]
    depth = [(r["layer"] + 1) / len(rows) * 100 for r in rows]
    values = [r[key] for r in rows]
    ax.plot(depth, values, color=col, ls=style, lw=2.6, label=name)
    i = selected["layer"]
    ax.scatter(depth[i], values[i], marker="*", s=210, color=col, edgecolor="white", zorder=5)
    DATA[name + "_roleplaying"] = {"depth_percent": depth, "test_auroc": values, "selected_zero_based_layer": i}
ax.axhline(lrole["text_only"]["auroc"], color=GREEN, ls="-.", lw=2, label="Completion TF-IDF (0.884)")
ax.axhline(.5, color=GRAY, ls=":", lw=2, label="Chance (0.500)")
ax.set(xlim=(0, 101), ylim=(.47, 1.02), xlabel="Model depth (% of decoder blocks completed)", ylabel="Test AUROC")
ax.legend(loc="lower center", bbox_to_anchor=(.5, .12), ncol=2, frameon=False)
grid(ax)
save(fig, "02_roleplaying_depth", "Stars: validation-selected blocks (Qwen 14, Llama 18; zero-based). C selected separately per layer on validation.\nFrozen replay of identical completion texts; completion-token mean pooling. Depth normalization does not equate blocks.")

fig = frame("Insider-trading probes before and during reports", "Identical Llama-generated trajectories replayed through frozen Qwen and Llama | Test n = 200", size=(14, 7.5))
for j, (stage, title, text_label) in enumerate([
    ("report", "Report", "Report TF-IDF"),
    ("pre_report", "Pre-report", "Prefix TF-IDF"),
    ("early_reasoning", "Early reasoning", "Full-prefix TF-IDF"),
]):
    ax = fig.add_axes([.08 + j * .305, .29, .265, .48])
    d = trading[stage]
    rows = [("Qwen activation", d["qwen"]["activation"]["auroc"], BLUE, "o"),
            ("Llama activation", d["llama"]["activation"]["auroc"], ORANGE, "s"),
            (text_label, prefix["shared_full_prefix_text"]["auroc"] if j == 2 else d["llama"]["text_only"]["auroc"], GREEN, "D")]
    if j == 2:
        rows.append(("8-token TF-IDF", d["llama"]["text_only"]["auroc"], PURPLE, "^"))
    for i, (label, val, col, marker) in enumerate(rows):
        ax.scatter(i, val, s=105, color=col, marker=marker, zorder=3)
        ax.text(i, val + .024, f"{val:.3f}", ha="center", fontsize=13, color=col, weight="bold")
    ax.axhline(.5, color=GRAY, ls=":", lw=1.5)
    ax.set(title=title, ylim=(.47, 1.09), xlim=(-.5, len(rows) - .5), xticks=range(len(rows)),
           xticklabels=[r[0].replace(" ", "\n", 1) for r in rows])
    ax.tick_params(axis="x", labelsize=11, length=0, pad=10)
    ax.set_yticks([.5, .6, .7, .8, .9, 1.0])
    if j == 0: ax.set_ylabel("Test AUROC")
    grid(ax)
    DATA[stage] = [{"method": r[0], "auroc": r[1]} for r in rows]
save(fig, "03_trading_stages", "Layer and C selected on validation only. Dotted line: chance. Report/early features are token means; pre-report is one vector.\nText controls use the report, full available prefix, or exact early window as labeled. All models hold out trajectories from one setting.\nQwen processes saved trajectories here; these are not Qwen-generated trading or deception decisions.")

fig = frame("Text and length nearly separate report labels", "Initial Llama insider-trading evaluation | 200 held-out trajectories from the same setting", size=(13, 7.5))
methods = [("Activation", "activation", ORANGE), ("TF-IDF", "text_only", GREEN), ("Length", "length_only", PURPLE)]
for j, (metric, title) in enumerate([("auroc", "AUROC"), ("auprc", "Average precision"), ("accuracy", "Accuracy")]):
    ax = fig.add_axes([.08 + j * .31, .27, .265, .51])
    vals = [initial[k][metric] for _, k, _ in methods]
    bars = ax.bar(range(3), vals, .62, color=[c for _, _, c in methods])
    ax.bar_label(bars, labels=[f"{v:.3f}" for v in vals], padding=7, fontsize=14, weight="bold")
    ax.axhline(initial["constant_baseline"][metric], color=GRAY, ls=":", lw=1.5)
    ax.set(title=title, ylim=(0, 1.13), xticks=range(3), xticklabels=[n for n, _, _ in methods])
    ax.set_yticks([0, .25, .5, .75, 1])
    if j == 0: ax.set_ylabel("Test score")
    grid(ax)
    DATA["initial_report_" + metric] = dict(zip([n for n, _, _ in methods], vals))
save(fig, "04_report_surface_confound", "Report-token mean activation (selected block 0), report-text TF-IDF, and report-token length; all use L2 logistic regression.\nDotted lines: constant-classifier baselines. Accuracy uses a 0.5 threshold; TF-IDF ranks well despite low thresholded accuracy.\nNear-perfect ranking from text and length means report-probe performance alone does not isolate a deception mechanism.")

(OUT / "plotted_data.json").write_text(json.dumps(DATA, indent=2) + "\n")
(OUT / "source_hashes.json").write_text(json.dumps(SOURCES, indent=2) + "\n")
print(f"Saved four PNG/PDF figure pairs in {OUT}")
