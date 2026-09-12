# Report-boundary residual comparison

`compare_report_residuals.py` compares trajectory 62 with the 24 manually reviewed
clear-disclosure buy trajectories in `apollo-hf-9b-a100-n100-v2`.
The cohort IDs are explicit in the script; it does not automatically label reports.

It selects the last token before the first token containing non-whitespace text
inside the successful Amy `send_message` call's `<parameter=text>` value. This is
a semantic alignment, not a common absolute token index. The residual is
`resid_post`, after each decoder layer and before the next layer/final normalization.

The original run saved only 32 generation forwards. Trajectory 62's requested
boundary is absolute token position 2055 (generated offset 106), beyond that
capture. By default the script refuses missing positions. `--replay-missing`
allows a separately recorded replay for **all 25 cases**, using the exact saved
prompt and continuation tokens, the pinned checkpoint, and cached sequential
forwards. Weights are frozen, gradients disabled, and no tokens are sampled.
Each replay is checked against its last overlapping original capture; it aborts
if any layer's cosine drops below 0.999 or relative L2 exceeds 0.02. Exact errors
are retained in the manifest. Original files are never changed.

On the A100, from `~/epistemic-deference`:

```sh
.venv-hf/bin/python experiments/apollo_trading/compare_report_residuals.py --run experiments/apollo_trading/outputs/apollo-hf-9b-a100-n100-v2 --output experiments/apollo_trading/outputs/report-residual-62-vs-disclosure --replay-missing
```

Use a fresh output directory. Requires the HF environment plus Matplotlib
(validated with 3.11.2); the checkpoint/tokenizer must already be cached locally.

Outputs:

- `comparison.safetensors`: `trajectory_62` and `disclosure_mean`, each
  `[32, 4096]` in float32; `disclosure_individuals` is `[24, 32, 4096]`, ordered
  by the manifest's disclosure IDs.
- `trajectory-XX-boundary.safetensors`: each selected boundary vector, preserving
  its model dtype, separately from original generation captures.
- `distances.csv`: cosine similarity and raw Euclidean L2 distance between
  trajectory 62 and the equal-weight disclosure mean, for each zero-based layer.
- `distance-vs-layer.png`: L2 distance and cosine similarity versus layer.
- `manifest.json`: boundary tokens/positions, overlap checks, cohort, layer names,
  model revision, package versions and source/script hashes.

```python
from safetensors.torch import load_file
data = load_file("comparison.safetensors")
target_layer_12 = data["trajectory_62"][12]
disclosure_mean_layer_12 = data["disclosure_mean"][12]
```

This is a descriptive comparison of one case with 24 controls. Different report
prefixes, reasoning, trade sizes and contexts can explain distances. Raw L2 also
depends on residual magnitude across layers. A large distance is not by itself a
deception feature or a causal result.

## Leave-one-out reference distribution

Run locally using the saved vectors, without loading Qwen:

```sh
.venv-hf/bin/python experiments/apollo_trading/analyze_report_outlier.py --input experiments/apollo_trading/outputs/report-residual-62-vs-disclosure --output experiments/apollo_trading/outputs/report-residual-62-loo
```

Each disclosure is compared to the mean of the other 23; trajectory 62 is compared
to all 24. Both L2 and cosine distance (1 minus cosine similarity) are computed
in float64. Outputs include all individual distances, per-layer ranks/percentiles,
and a plot showing every control, the min-max envelope, interquartile band,
median, target, and target percentiles. Rank 25/25 is farthest. Ties receive
average ranks and half-credit in empirical percentiles. These are not p-values.

Observed L2 percentiles range from 79.2% to 100%; layers 8-22 are at 95.8%.
Only layer 0 exceeds every control's L2 distance. No layer exceeds every control's
cosine distance. At layer 31, target L2 is 78.72 (87.5th percentile, rank 22/25),
versus control median 53.19 and maximum 101.03. Its cosine-distance percentile
there is 91.7% (rank 23/25). This puts 62 in the upper tail without cleanly
separating it from disclosure variation. Layers are correlated and the sample is
small; context differences remain plausible explanations.

## Four non-clear disclosures versus 24 disclosures

Additional vectors for 06, 21 and 54 are in `outputs/report-boundaries-06-21-54/`.
They were recovered with the same frozen cached-token replay helper, and every
overlap check matched the original residuals exactly. The comparator reuses the
previous vectors for 62 and all 24 disclosures:

```sh
.venv-hf/bin/python experiments/apollo_trading/compare_disclosure_groups.py --reference experiments/apollo_trading/outputs/report-residual-62-vs-disclosure --additional experiments/apollo_trading/outputs/report-boundaries-06-21-54 --output experiments/apollo_trading/outputs/report-group-4-vs-24
```

`group-vectors.safetensors` contains both group means, their signed difference
(non-clear minus disclosure, each `[32,4096]`), and individual vectors. The CSV
and Markdown report list each layer's cosine similarity, L2 separation, exact
permutation p-value, and Holm-adjusted p-value. The plot contrasts observed L2
with the label-permutation null distribution. The full null is also saved.

All 20,475 unique four-member subsets of the 28 cases are enumerated; group sizes
remain 4 and 24. The primary statistic is L2 between means; p is the fraction of
assignments at least as extreme as observed, including the observed split. Holm
correction covers the 32 layerwise L2 tests. Cosine similarity is descriptive.

No layer has raw p below 0.05. The minimum is 0.1543 at layer 18; all adjusted
p-values are 1.0. Final-layer cosine similarity is 0.99344, L2 is 31.759, and
raw p is 0.2963. This test does not establish a shared mean shift beyond the
label-permutation null. It does not prove no signal exists. The group is small,
heterogeneous and selected after behavior inspection; labels are not randomized,
and exchangeability can be undermined by context or prefix differences.
