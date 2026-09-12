# Write-up figures

Four figure pairs are exported as 300 dpi PNG and vector PDF with embedded TrueType fonts. All values come from saved results; no model inference, split changes, or classifier refitting is performed.

Reproduce from the repository root:

```sh
.venv-hf/bin/python figures/publication/make_figures.py
```

`plotted_data.json` records plotted values. `source_hashes.json` records SHA-256 hashes of the source metrics and qualitative reviews.

## Figure captions

1. **Qwen trading decisions and clear deception.** Percentages use all 100 trajectories per run as the denominator. Buy counts represent new accepted purchases, excluding the purchase in the supplied prompt history. Clear-deception counts require explicit emitted concealment intent plus matching reporting behavior in the manual reviews; these are identified cases, not validated estimates of the true deception rate. Failed and ambiguous cases are not classified as honest. The text baseline and text rerun have identical generated texts and are not independent observations. The v2 concealment example is a sale. Protocol differences prevent attributing rate changes to a single intervention. In the original text runner, reports are generated continuations rather than validated messaging-tool execution.

   Label mapping: Text baseline = `apollo-trading-100`; Text rerun = `apollo-qwen-n100`; Ollama v2/v3/v4 = `apollo-qwen-n100-v2/v3/v4`; HF A100 v2 = `apollo-hf-9b-a100-n100-v2`. Clear examples: Ollama v2 trajectory 84, Ollama v4 trajectory 81, HF A100 v2 trajectory 62.

2. **Roleplaying probes across model depth.** Frozen Qwen3.5-9B and Llama 3.3-70B replay the same 742 completions from 371 paired scenarios. Scenario-grouped train/validation/test splits contain 259/56/56 pairs. Each probe uses completion-token mean residuals, training-only standardization, and L2 logistic regression. Regularization is selected independently per layer using validation AUROC. Stars indicate validation-selected layers, not test maxima. Depth is `(zero_based_block_index + 1) / number_of_blocks`, with 32 Qwen and 80 Llama blocks. The common TF-IDF classifier uses exact completion texts. The completions are released, teacher-forced text, not spontaneous deception by the replay models.

3. **Insider-trading probes before and during reports.** Both frozen models process identical released Llama-generated trajectories and retain the same 200 test examples. Features are report-token mean residuals, the final pre-report token vector, and an eight-token early-reasoning mean. Layer and regularization selection use validation only. Report TF-IDF sees the report text; pre-report TF-IDF sees the full prefix available at that boundary. Early reasoning includes both the full prefix through the same endpoint and the exact eight-token window as separate text controls. All are held-out trajectories from the same setting, unlike the scenario holdout in roleplaying. Qwen makes no trading or reporting decisions in this replay experiment.

4. **Text and length nearly separate report labels.** The initial Llama report evaluation compares the validation-selected mean-residual probe (block 0), report-text TF-IDF, and report-token length. The test cohort contains 133 honest and 67 deceptive trajectories. Dotted baselines are 0.5 AUROC, 0.335 average precision, and 0.665 constant-classifier accuracy. Accuracy uses a probability threshold of 0.5; TF-IDF's low accuracy does not negate its strong ranking. Text and length performance demonstrate a surface-feature confound, without establishing which features the activation probe actually uses.

These figures show descriptive point estimates. They do not display uncertainty intervals or imply statistically significant model differences. Existing comparison reports contain paired bootstrap intervals for selected model differences.
