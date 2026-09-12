# Qwen processing Llama's released insider-trading trajectories

This is an off-policy representation comparison. **Qwen does not generate any
of these trajectories or make a trading/disclosure decision.** Frozen Qwen3.5-9B
teacher-forces the exact saved Llama-generated message and completion text.
Apollo's labels and original trajectory IDs are retained.

## Results on identical examples

| Activation feature | Llama AUROC | AUPRC | Accuracy | Qwen AUROC | AUPRC | Accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Full report | 1.0000 | 1.0000 | 0.9950 | 1.0000 | 1.0000 | 0.9950 |
| Llama-length-matched report | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Pre-report vector | 0.9912 | 0.9800 | 0.9650 | 0.9872 | 0.9585 | 0.9500 |
| Early reasoning mean | 0.8597 | 0.7460 | 0.7600 | 0.8666 | 0.7384 | 0.7800 |

Validation selects layer 0 for both report analyses, layer 1 for both pre-report
probes, and layer 17 for Llama versus layer 16 for Qwen early reasoning. Indices
are zero-based. Qwen minus Llama AUROC is -0.0040 [-0.0110, 0.0018] pre-report and
+0.0068 [-0.0107, 0.0234] early; neither establishes a model advantage. Report
ranking metrics are identical. The length-matched test set has only 14 reports.

[Complete side-by-side TF-IDF, shuffled-label and length tables](outputs/qwen35-llama-trading-eval-v1/COMPARISON.md)
and [all layerwise curves](outputs/qwen35-llama-trading-eval-v1/layerwise_comparison.png)
are saved with the evaluation. Shared TF-IDF test predictions reproduce the
Llama references within 1e-6 for every analysis. Reports remain strongly
predictable from wording and length; high report-probe scores do not isolate
deception-specific internal processing.

For the early window, shared full-prefix TF-IDF scores AUROC 0.7867, AUPRC 0.6880,
accuracy 0.7300. Qwen's selected early probe exceeds its AUROC by 0.0799
[0.0301, 0.1310]; Llama's earlier measured advantage is 0.0731 [0.0271, 0.1195].
These comparisons are against a bag-of-words baseline with the same available
text, not against every possible contextual text classifier.

| Fixed early layer | Llama AUROC | Qwen AUROC |
| --- | ---: | ---: |
| 0 | 0.8365 | 0.8492 |
| 17 | 0.8597 | 0.8709 |

Qwen layer 17 is shown as the requested fixed-layer diagnostic, not substituted
for its validation-selected layer 16. Layer numbers do not imply equivalent
relative depth across these architectures.

## Execution and replay

The completed Qwen work runs on the original A100 VM, `129.146.167.34`, using
its existing `.venv-hf`, cached checkpoint and optimized attention/convolution
kernels. At the user's request, the earlier attempt on `192.222.53.243` was stopped
and its Qwen checkpoint directory, partial output directory, runner and logs were
removed. The three packages added there (flash-linear-attention, fla-core, einops)
were uninstalled. Existing Llama outputs and its checkpoint were preserved.

Qwen model/tokenizer revision:
`Qwen/Qwen3.5-9B@c202236235762e1c871ad0ccb60c8ee5ba337b9a`.
It is FP16, eval mode, requires_grad=False, with inference_mode, eager full
attention, and no KV cache. This matches the existing Llama replay's FP16 dtype.
All 32 Qwen decoder-block outputs are saved before final normalization.
The native Qwen template uses `enable_thinking=False`; saved reasoning is replayed
verbatim, not regenerated. Prior message contents are checked for preservation
in the native template. No generated end-of-turn token is appended to Qwen's
final assistant text.

Token IDs cannot be copied between different tokenizer vocabularies. The original
Llama replay IDs are retained as `llama_input_ids`, while Qwen's IDs are saved
separately. Every full Qwen sequence must decode back to its rendered text.
Report spans and early windows are aligned by **exact character offsets**, with
no boundary-straddling token accepted. Thus both models process the same report
and early-window text, although their token IDs and native chat markers differ.
The original released provider token IDs were not available; the existing Llama
IDs themselves are reconstructed replay IDs, as documented in the source audit.

The four-case Qwen pilot checks exact repeat-forward activations/logits and
hook/no-hook logits. It also compares full-forward versus truncated-prefix
vectors at the pre-report boundary and early-window endpoint. Maximum per-layer
relative L2 differences are below 0.3%; these are not bitwise identical because
FP16 kernel shapes differ. The full frozen replay supplies the saved activations.

## Identical cohorts, spans and splits

| Analysis | Train | Validation | Test | Feature |
| --- | ---: | ---: | ---: | --- |
| Full reports | 650 | 150 | 200 | FP32 mean of report-token residuals |
| Llama-defined length-matched reports | 62 | 10 | 14 | Same mean, same 86 selected IDs |
| Pre-report | 650 | 150 | 200 | Final token immediately before report starts |
| Early reasoning | 649 | 150 | 200 | Mean over the fixed early text window |

All 1,000 strict labels (666 honest / 334 deceptive) and original assignments are
preserved. The early cohort retains exactly the earlier 999 IDs and one training
exclusion. Its training-defined anchor is not estimated again for Qwen. The exact
same eight-token Llama windows also tokenize to eight Qwen tokens for all 999
retained trajectories. Both span endpoints align exactly, and no extra example
is dropped.

The length-matched cohort remains selected by **Llama token count**, ensuring
identical model-comparison examples. Qwen's report token counts are generally
three higher, with some larger differences, so equality of its two label length
distributions is not assumed. The own-tokenizer length baseline and distributions
are reported separately for each model. Pre-report length also uses each model's
own prefix-token count; early-window length is eight for both.
The observed Qwen matched-cohort histograms are identical between labels on
validation and test, but not on training. Its test length baseline still scores
AUROC/AUPRC/accuracy 0.5/0.5/0.5. This is not described as exact Qwen-length
matching across every split.

## Probe and text controls

The same helper fits a training-only StandardScaler and L2/lbfgs logistic
regression per layer, choosing C from [0.001,0.01,0.1,1,10] and then layer using
validation AUROC only. Smallest C and earliest layer win ties. All Qwen activation
fits allow up to 10,000 iterations and reject convergence warnings. Llama's early
probe used that same ceiling; its other completed probes converged within 2,000.
This changes the available optimization time, not the objective or stopping
tolerance. No Llama fit or test-based selection is redone.

The five shuffled controls use the **same actual training-label permutations**
(seeds 42-46), with true validation/test labels and the same searches. All train
and test IDs are asserted equal to each Llama reference.

TF-IDF uses the **identical canonical text inputs** from each Llama evaluation:
report body for the report analyses, full available pre-report prefix for the
pre-report analysis, and the exact early window for the early analysis. Prefix
baselines retain the canonical Llama-rendered chat markers; they are shared
controls rather than separate tokenizer/template-dependent classifiers.
Vocabulary/IDF are fitted only on training text, with word unigrams/bigrams,
min_df=2, sublinear TF and validation-selected L2 logistic regression. Recomputed
TF-IDF predictions must match the saved Llama baseline within 1e-6.

The additional full-prefix control at the early-window endpoint reuses the
completed full-prefix TF-IDF predictions on those same 999 IDs. Its results are
reported alongside Qwen's early selected layer and fixed layers 0 and 17.
Activations incorporate earlier context; strong performance against TF-IDF does
not isolate a deception mechanism or establish intent in Qwen.

Deceptive is positive, AUPRC is average precision, and accuracy uses threshold
0.5. Paired trajectory-bootstrap intervals use 2,000 resamples with seed 2026,
conditional on the fitted models. Both models hold out trajectories from the
same insider-trading setting, not new scenarios. Architectures differ (Qwen
32 x 4096 hybrid blocks; Llama 80 x 8192), as do native templates, tokenizer,
GPU placement and runtime/kernel versions. Revisions and packages are saved.

## Files and reproduction

On the A100, relative to `/home/ubuntu/epistemic-deference/`:

```text
experiments/apollo_released/outputs/qwen35-llama-trading-a100-v1/
  manifest.jsonl                 # IDs, labels, splits, both tokenizations, spans
  activations/<trajectory>.safetensors
  features.safetensors           # report mean, pre-report vector, early mean
  config.json, pilot_fidelity.json, summary.json
experiments/apollo_released/outputs/qwen35-llama-trading-eval-v1/
  comparison.json, COMPARISON.md, layerwise_comparison.png
  early_full_prefix.json, qwen_length_distributions.json
  report/, llama_length_matched/, pre_report/, early_reasoning/
```

Each evaluation directory includes its manifest, pooled activation inputs,
all-layer metrics, five shuffle searches, weights/scalers, predictions, actual
shuffled training labels, and fitted text/length classifiers. Large activations
stay on the A100; small results and metadata are mirrored locally. Source Llama
activations remain on the four-A6000 VM.

Validation checked all 1,000 activation headers, shapes, IDs, labels and splits.
Four sampled report/early means and pre-report indices exactly reproduce saved
features; all pooled features are finite. All 128 saved Qwen layer probes
reconstruct their test predictions with zero maximum absolute error. The four
cohorts preserve the original train/test IDs and actual shuffled-label arrays.
These checks are saved in each run's `artifact_checks.json`.

Run from the A100 project root with fresh output directories:

```sh
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 .venv-hf/bin/python experiments/apollo_released/run_qwen_replay.py experiments/apollo_released/outputs/llama33-reports-v1 --evaluation experiments/apollo_released/outputs/llama33-probe-eval-v3 --early-run experiments/apollo_released/outputs/llama33-early-reasoning-v2 --output experiments/apollo_released/outputs/qwen35-llama-trading-a100-v1
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 .venv-hf/bin/python experiments/apollo_released/evaluate_qwen_replay.py experiments/apollo_released/outputs/qwen35-llama-trading-a100-v1 --references experiments/apollo_released/outputs --output experiments/apollo_released/outputs/qwen35-llama-trading-eval-v1
```
