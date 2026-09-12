# Matched Qwen / Llama probe evaluation

Follow-up length-matched report and pre-report controls are documented in
[REPORT_CONTROLS.md](REPORT_CONTROLS.md), using the same fixed evaluation splits.
The subsequent fixed eight-token reasoning-window comparison is in
[EARLY_REASONING.md](EARLY_REASONING.md).
The frozen Qwen replication on identical Llama-generated trajectories is in
[QWEN_LLAMA_TRADING.md](QWEN_LLAMA_TRADING.md).

The completed results are in
[`outputs/llama33-probe-eval-v3/COMPARISON.md`](outputs/llama33-probe-eval-v3/COMPARISON.md),
with detailed per-layer/per-seed scores in `metrics.json`.

Llama's validation-selected probe uses layer **0**, C=0.001, and achieves test
AUROC **1.0000**, AUPRC **1.0000**, and accuracy **0.9950**. This is the first
decoder block's unnormalized output, not the embedding or final normalization.

The central limitation is strong surface predictability: **length alone achieves
0.9860 AUROC and 0.9500 accuracy**, and TF-IDF achieves **0.9966 AUROC**. Test report
lengths have medians of 59 tokens for honest reports and 43 for deceptive reports.
Honest lengths range 45-106; deceptive lengths range 38-59.

Activation-minus-TF-IDF AUROC is only 0.00337, with a paired-trajectory bootstrap
95% interval [0, 0.01094]. The perfect activation ranking supplies little evidence
of a deception-specific representation beyond these surface cues. The length
signal is especially strong here, unlike the Qwen roleplaying length baseline.

## Exact procedure

`evaluate_reports.py` imports the same classifier-selection, TF-IDF, metric, and
bootstrap helpers used by `apollo_roleplaying/sanity_checks.py`. Numerical
packages match the Qwen checks and are pinned in `requirements-probe.txt`.

- Features: saved FP32 `report_mean`, shape `[1000,80,8192]`. This averages FP16
  report-token residuals in FP32, from the token overlapping `Executed trade
  update` through the final assistant content, excluding EOT. Earlier reasoning
  remains in context but is excluded from the pooling region.
- A StandardScaler is fitted only on training rows, independently per layer.
  Logistic regression uses the default L2/lbfgs procedure, max_iter=2000,
  random_state=42, and C in `[.001,.01,.1,1,10]`.
- C per layer and then the layer are selected using validation AUROC only.
  Equal scores retain the first candidate: smaller C, then earlier layer.
- TF-IDF uses exact report-body strings, word unigrams/bigrams, min_df=2,
  sublinear_tf=True. Vocabulary and IDF are fitted only on training reports.
  It shares the same C grid and validation selection.
- Five shuffled controls use seeds 42-46 to permute training labels only,
  preserving class counts. Each repeats the full layer/C search against true
  validation labels; the test labels remain true. Report the seed-42 result
  and five-seed mean/SD without choosing a seed by test performance.
- Length uses the number of pooled report tokens, a training-only scaler, and
  logistic regression with C=1, max_iter=2000, as in Qwen.
- Deceptive is positive, AUPRC is average precision, and accuracy uses threshold
  0.5. No test-based threshold adjustment or calibration is performed.
- The constant baseline predicts the training positive-class prevalence.

All fits use saved activations; neither model is trained or run during evaluation.
The original Qwen results and Llama extraction files are unchanged.

## Fixed test set and validation

All original 200 Llama test trajectory IDs remain held out. Validation is selected
only from the original 800 training trajectories, stratified by label with seed
42: 150 validation and 650 remaining training trajectories.

| Split | Honest | Deceptive | Total |
| --- | ---: | ---: | ---: |
| Train | 433 | 217 | 650 |
| Validation | 100 | 50 | 150 |
| Test | 133 | 67 | 200 |

`evaluation_manifest.jsonl` preserves trajectory IDs, Apollo loader IDs, original
split, evaluation split, original labels, exact report text, and token count.
Splits are checked for trajectory overlap. Qwen's original 259/56/56 scenario
splits are unchanged; its metrics are read directly from the completed checks.

**Qwen holds out roleplaying scenarios; Llama holds out alternative trajectories
from the same insider-trading setting.** This is not a model-capability comparison.
The tasks, labels, context, and held-out units differ. Qwen's test prevalence is
50% deceptive; Llama's is 33.5%, so chance AUPRC is 0.335 and majority accuracy
is 0.665 for Llama. The shuffled controls average 0.5286 AUROC, 0.3748 AUPRC,
and 0.6080 accuracy; their accuracy should be judged against the majority baseline.

## Why TF-IDF accuracy is 66.5% despite excellent ranking

All TF-IDF C candidates are selected solely by validation AUROC, with the first
candidate winning ties. C=0.001 achieves validation AUROC 1.0 and is retained.
Its test probabilities range from 0.3293 to 0.3416, all below 0.5, so every test
report is classified as honest. Accuracy is consequently 66.5%, while ranking
metrics remain excellent. The procedure is preserved rather than changing the
threshold or regularization after seeing test performance. The large accuracy
gap does not demonstrate a comparably large information advantage for activations.

## Checks and artifacts

- All 1,000 activation file headers, trajectory IDs and labels were checked.
  Four sampled saved report means matched fresh token-residual means exactly.
- The selected saved probe reconstructs its test probabilities to maximum error
  7.9e-8. Test IDs and source manifest hashes match; shuffled training labels
  preserve class counts. These checks are saved in `artifact_checks.json`.
- Three evaluation tests pass: train-only validation/test preservation, duplicate
  trajectory rejection, and exact serial/process-parallel fit equivalence on a
  fixture. Four shared Qwen sanity-check tests also pass. No convergence warnings
  occurred in the completed evaluation.
- `probes.npz` stores all layers' selected coefficients and scaler parameters.
  `predictions.npz` preserves every layer/seed's test probabilities, test IDs,
  true labels, training IDs, and the shuffled training labels. The fitted text
  classifier is in `text_classifier.joblib`.

The completed v3 run uses eight independent CPU processes, restoring layer order
before tie-breaking. The serial v1 and threaded v2 attempts were stopped for
runtime reasons and retained; they are not used in the comparison.

Run on the VM from `~/epistemic-deference`, using a fresh output directory:

```sh
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 .venv-probe/bin/python experiments/apollo_released/evaluate_reports.py experiments/apollo_released/outputs/llama33-reports-v1 --output experiments/apollo_released/outputs/llama33-probe-eval-v3 --qwen-metrics experiments/apollo_roleplaying/outputs/qwen9b-sanity-v1/metrics.json
```
