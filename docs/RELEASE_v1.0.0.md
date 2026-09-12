# Research dataset and reproducibility release

Public code, figures, compact text datasets, and verified research archives for frozen Qwen3.5-9B and Llama 3.3-70B probes on Apollo insider trading and paired roleplaying.

## Start here

- [Dataset guide and exact artifact catalog](https://github.com/Soheil-Saneei/apollo-deception-probes/blob/main/data/README.md)
- [Results and figures](https://github.com/Soheil-Saneei/apollo-deception-probes/blob/main/results/README.md)
- [Reproduction instructions](https://github.com/Soheil-Saneei/apollo-deception-probes/blob/main/docs/REPRODUCING.md)

The repository includes 1,000 strict insider-trading reports (666 honest, 334 deceptive), 999 early windows and corresponding full prefixes, and 742 roleplaying completions from 371 paired scenarios. IDs, labels, and existing train/validation/test assignments are preserved. Both model-comparison experiments replay supplied texts; they do not measure spontaneous deception by Qwen or Llama.

## Full archives

- [A100 artifacts](https://github.com/Soheil-Saneei/apollo-deception-probes/releases/tag/artifacts-a100-v1): Qwen generations, roleplaying, and replay of Llama trading.
- [Llama artifacts](https://github.com/Soheil-Saneei/apollo-deception-probes/releases/tag/artifacts-a6000-v1): report, reasoning-window, and roleplaying activations and evaluations.
- [Local artifacts](https://github.com/Soheil-Saneei/apollo-deception-probes/releases/tag/artifacts-local-v1): Ollama runs, Mac pilots, analyses, and historical source files.

All 70 archive groups are complete. Across the three overlapping snapshots, 18,256 files were preserved, totaling about 317.8 GiB before compression and 267.1 GiB uploaded. Every archive member was reconstructed and checked against the source inventory. All published asset sizes and SHA-256 digests passed an independent public-access check. A public download-and-restore sample passed. No base-model weights, environments, caches, or credentials are distributed.

The existing 58 tests pass. All four publication figures regenerate from a clean checkout. Original experiment paths and historical artifacts are preserved, and the source-specific package inventories are included. Model revisions and replay limitations are documented with each experiment.

Apollo source material is attributed in [THIRD_PARTY_NOTICES.md](https://github.com/Soheil-Saneei/apollo-deception-probes/blob/main/THIRD_PARTY_NOTICES.md). This is an independent research release, not an official Apollo Research dataset release or a blanket license grant over third-party materials.
