# Interpretation and limitations

- **Generation versus replay:** Qwen's simulated trading runs measure generated behavior. Cross-model probe comparisons teacher-force released texts; they do not measure the replay models' willingness to deceive.
- **Labels:** Insider-trading probes retain strict Apollo report labels (666 honest, 334 deceptive), excluding ambiguous/invalid cases. Local Qwen reviews use a different conservative criterion: emitted concealment intent plus matching delivered reporting. Those counts are identified examples, not comprehensive prevalence.
- **Dependence:** The two text-format Qwen batches have identical outputs. Protocol versions differ. Six batches are not 600 independent, identically distributed experiments.
- **Splits:** Roleplaying uses 259/56/56 scenario pairs in train/validation/test. Insider trading uses 650/150/200 trajectories from one setting; early reasoning excludes one training example. The matched-length test contains only 14 reports.
- **Selection:** Layer and regularization use validation AUROC, never test scores. Accuracy uses a fixed 0.5 threshold; high AUROC can coexist with majority-class accuracy.
- **Surface features:** Report wording and length are highly predictive. Activation performance does not establish a deception-specific mechanism. TF-IDF is not every possible contextual text classifier.
- **Timing:** A block output has observed its current token. Report and completion means represent supplied text. Pre-report vectors exclude report text but include earlier reasoning. The early window follows a training-defined modal-token divergence; it is not necessarily the first eight reasoning tokens.
- **Replay fidelity:** Provider token IDs and serving configurations are unavailable. Analysis tokens are reconstructed. The initial Llama trading replay uses the documented Llama 3.1 tokenizer; the later Llama roleplaying replay uses its native 3.3 tokenizer. Architecture, template, and hardware also differ between models.
- **Emitted reasoning:** Concealment statements support a behavioral interpretation but are not guaranteed faithful descriptions of internal computation.
- **Uncertainty:** Small score differences do not establish model superiority; consult the paired bootstrap intervals in the completed comparisons.
