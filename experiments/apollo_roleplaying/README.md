# Apollo paired roleplaying on frozen Qwen

For public transcripts, splits, and downloadable activation archives, see the
[data guide](../../data/README.md). Machine paths below record original run
locations; restore archives to use them after the VMs are retired.

The completed matched Llama 3.3 70B replication, using the same examples and
scenario splits, is documented in [LLAMA_COMPARISON.md](LLAMA_COMPARISON.md).

This experiment teacher-forces the released honest and deceptive completion for
each scenario through **Qwen/Qwen3.5-9B**, revision
`c202236235762e1c871ad0ccb60c8ee5ba337b9a`, in BF16 and evaluation mode with
gradients disabled. It does not generate answers or update Qwen weights.

## Data and labels

The byte-for-byte source copy is `source/dataset.yaml`, from
[Apollo's dataset](https://github.com/ApolloResearch/deception-detection/blob/f8ec4010e74927394709dffa22b97bdf8cd5a62f/data/roleplaying/dataset.yaml).
It contains **371 scenarios and 742 completions**, with no missing, empty, or
identical honest/deceptive pairs. IDs are zero-based source array indices.
The manifest includes original text, labels, scenario hashes, split, source index,
completion hash, rendered prompt, exact replay token IDs, token positions, and
activation paths. The source file hash and pinned commit are in `config.json`.

Apollo's
[off-policy generation script](https://github.com/ApolloResearch/deception-detection/blob/f8ec4010e74927394709dffa22b97bdf8cd5a62f/deception_detection/scripts/roleplaying_data_generation/generate_offpolicy.py)
requests HONEST and DECEPTIVE examples separately from
`together/meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo`, temperature 1.0, and stores
the returned strings under the corresponding completion keys. These are
construction labels, not independently verified behavior labels. The released
YAML does not contain per-completion grader scores or provider token IDs.
The generator's code identifies the intended generator model, but does not pin
the exact provider checkpoint revision for every saved answer.

We follow the messages in Apollo's
[`offpolicy_train` loader](https://github.com/ApolloResearch/deception-detection/blob/f8ec4010e74927394709dffa22b97bdf8cd5a62f/deception_detection/data/roleplaying.py):
system = scenario, user = question, assistant = completion. No answer prefix,
honesty instruction, or deception instruction is inserted. Despite that loader's
docstring mentioning thoughts, its actual implementation directly uses the saved
completion fields. The scenario and question are identical within each pair.

## Replay and activations

Qwen's pinned chat template is used with `enable_thinking=False`. The exact saved
completion string is appended to the resulting assistant generation prefix.
Tokenization is applied to the combined string, without added special tokens,
truncation, or a generated end-of-turn token. Boundary-straddling tokens cause a
failure; decoded completion tokens must exactly round-trip to the saved text.
This is faithful replay of released **text under Qwen's tokenizer**, not replay
of the original Llama provider token IDs or original generation instructions.

One complete sequence is processed per forward pass. Both answers are right-padded
to the longer sequence in their pair, with padding masked and excluded from
activations and means. This keeps kernel input shapes identical within a pair.
The model uses eager attention and
`use_cache=False`. Hooks save the output of each of the 32 decoder blocks
(`resid_post`, before final model normalization). Per-example safetensors contain
BF16 `[layer, completion_token, hidden]` activations plus the final prompt token's
`[layer, hidden]` activation. `features.safetensors` contains FP32 completion means
and prompt-boundary activations, in manifest order. A token's block output has
already observed that token; these are representation probes, not pre-answer
predictions of a future decision.

Checks: exact text/token round-trip for every completion; first-example logits
with and without hooks; paired prompt-boundary activation differences recorded
in `fidelity.json`. Different sequence lengths can cause numerical differences
in GPU kernels even though the prompt is causally identical.

## Probe and split

Scenario hashes use whitespace-normalized scenario text. Repeated scenarios,
including ones with different questions, stay in a single split. Sorted unique
groups are shuffled with Python RNG seed 42, then split 70/15/15 by integer
cutoffs: **259 train / 56 validation / 56 test scenarios**, each balanced by label.
Related scenario themes or paraphrases are not automatically deduplicated; the
test measures transfer to held-out scenario texts, not necessarily new themes.

A separate StandardScaler plus L2 logistic regression is fitted at each layer
on the **training split only**, using the mean over completion tokens. C is
chosen from `[.001, .01, .1, 1, 10]` by validation AUROC, with smaller C winning
ties. The highlighted layer is chosen by validation AUROC, never test AUROC.
Test labels are used only for final evaluation. `probes.npz` saves weights,
scaler parameters, test IDs, and predictions; `probe_metrics.json` and
`probe_by_layer.png` report results. An always-one-class baseline has 50% accuracy. A separate logistic regression
on completion token count provides a simple length-confound baseline.

High accuracy could reflect confessions, denials, phrasing, or other answer-style
features. It does not establish a deception mechanism or show that Qwen chose to
deceive. There is no manual relabeling of Apollo's intended-completion labels.

## Run

From the project root on the A100, with the existing `.venv-hf` environment and
scikit-learn installed:

```sh
.venv-hf/bin/python experiments/apollo_roleplaying/run.py --output experiments/apollo_roleplaying/outputs/qwen9b-paired-v2
OPENBLAS_NUM_THREADS=2 .venv-hf/bin/python experiments/apollo_roleplaying/probe.py experiments/apollo_roleplaying/outputs/qwen9b-paired-v2
.venv-hf/bin/python -m unittest discover -s experiments/apollo_roleplaying -p 'test_*.py'
```

Use a new output directory for every extraction. Outputs are ignored by Git.
Original trading and Llama-audit outputs remain separate.

## Completed run: qwen9b-paired-v2

All 742 examples were extracted across all 32 layers. The validation-selected
layer was index 14 (the 15th decoder block), C=0.01. Validation AUROC was 0.9694;
test AUROC was **0.9802** and test accuracy **0.9107** (102/112 answers).
The completion-length baseline achieved test AUROC 0.4944 and accuracy 0.50.
These are point estimates from 56 held-out pairs, not 112 independent scenarios.

Hook/no-hook logits matched exactly. Shared prompt-boundary activations matched
exactly for all 371 pairs. All 742 activation file shapes were checked; all pooled
features were finite, and four sampled saved means matched their token activations
exactly. Three unit tests passed locally and on the A100.

The earlier unpadded `qwen9b-paired-v1` is retained on the A100. It had a maximum
absolute shared-prompt difference of 3.5 despite identical causal context.
Pairwise padding in v2 reduced this to zero; the validation-selected layer and
reported test metrics were unchanged. Use v2 for subsequent analysis.

Local results, manifest, probe weights, plot, and pooled activations are under
`experiments/apollo_roleplaying/outputs/qwen9b-paired-v2/`. Per-completion token
activations remain on the A100 in the corresponding project directory:
`/home/ubuntu/epistemic-deference/experiments/apollo_roleplaying/outputs/qwen9b-paired-v2/activations/`.
Manifest activation paths are relative to that remote run directory. The local
download intentionally does not contain the token-level activation files.

## Text and shuffled-label sanity checks

Run `sanity_checks.py` on the saved v2 outputs, without rerunning or updating Qwen:

```sh
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 .venv-hf/bin/python experiments/apollo_roleplaying/sanity_checks.py experiments/apollo_roleplaying/outputs/qwen9b-paired-v2 --output experiments/apollo_roleplaying/outputs/qwen9b-sanity-v1
```

All original examples and scenario splits remain unchanged. The activation result
reuses the original validation-selected layer 14 and saved predictions, verified
against its weights and features to within 1e-6. The TF-IDF baseline receives only
the exact completion strings, uses word unigrams/bigrams, min_df=2, sublinear TF,
and learns its vocabulary/IDF on training texts only. Its logistic regression C
uses the same five-value grid and validation AUROC selection; C=1 was selected.

The shuffled control permutes **training labels only**, preserving training class
counts. Validation and test labels retain their true values. For each of five
predeclared seeds (42-46), it repeats the full 32-layer/five-C search on validation
AUROC. No model or seed is selected by test performance. This tests whether
shuffled training supervision produces generalizable real-label discrimination;
it is not a permutation p-value. Seed 42 is also recorded separately.

On the same 56 held-out scenarios (112 answers):

| Model | Test AUROC | Test AUPRC | Test accuracy |
| --- | ---: | ---: | ---: |
| Activation probe | 0.9802 | 0.9769 | 0.9107 |
| TF-IDF + logistic regression | 0.8839 | 0.8565 | 0.8125 |
| Shuffled activation, seed 42 | 0.4818 | 0.4938 | 0.4732 |
| Shuffled activation, five-seed mean | 0.5180 | 0.5429 | 0.4893 |
| Completion length only | 0.4944 | 0.4903 | 0.5000 |
| Constant baseline | 0.5000 | 0.5000 | 0.5000 |

Deceptive is the positive class; AUPRC means **average precision**, not trapezoidal
PR area. Accuracy uses threshold 0.5. The original 98% result is AUROC, not accuracy.
Across the five shuffled runs, AUROC SD is 0.0729 and accuracy SD is 0.0639.

Paired bootstrap resampling of whole test scenarios (2,000 resamples, seed 2026)
gives activation-minus-text AUROC +0.0963, 95% interval [0.0443, 0.1582]; AUPRC
+0.1204 [0.0499, 0.1924]; accuracy +0.0982 [0.0357, 0.1696]. These intervals
describe test-scenario sampling uncertainty conditional on the fitted models.
The activation classifier alone correctly classified 13 test answers that TF-IDF
missed; the converse occurred for two answers.

The activation probe therefore adds predictive value **over this particular
TF-IDF baseline**. The shuffled control falls near chance on average. These checks
do not establish information beyond all surface-text features or a dedicated
deception mechanism: contextual meaning, confession/denial style, and the scenario
context available to Qwen could account for the advantage. A stronger contextual
text baseline or matched paraphrase controls could narrow that interpretation.

`outputs/qwen9b-sanity-v1/` contains `metrics.json`, `RESULTS.md`, test predictions
for every model/layer/seed, the actual shuffled training labels, and the fitted
text vectorizer/classifier. The original run and probe outputs are untouched.
Seven roleplaying unit tests pass, including training-only vocabulary, exact text
hashes, cross-split rejection, metric definitions, and paired bootstrap checks.
