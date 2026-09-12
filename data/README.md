# Dataset and artifact downloads

The public repository contains code, figures, and small metrics. Full research artifacts are stored in [GitHub Releases](https://github.com/Soheil-Saneei/apollo-deception-probes/releases), divided by source machine. **Archival is in progress; a release is complete only when its upload ledger marks every group complete.**

| Release | Contents |
| --- | --- |
| `artifacts-a100-v1` | Qwen generated trading runs, both roleplaying extraction versions, Qwen replay of Llama trading, pooled features, evaluations, and source code/inputs |
| `artifacts-a6000-v1` | Llama report and roleplaying activations, early reasoning, report controls, evaluation versions, tokenizer pilot, and source code/inputs |
| `artifacts-local-v1` | Local Ollama runs, Mac pilots, manual-review artifacts, local analyses, and local-only experiment files |

These are overlapping source snapshots, not disjoint datasets. Keep them separate on initial extraction. Historical/pilot outputs are preserved for provenance, not silently pooled with final results.

## Which run should I use?

| Analysis | Extraction run | Evaluation run |
| --- | --- | --- |
| Qwen paired roleplaying | `qwen9b-paired-v2` | `qwen9b-sanity-v1` |
| Llama paired roleplaying | `llama33-paired-v1` | `llama33-roleplaying-eval-v1` |
| Llama insider-trading reports | `llama33-reports-v1` | `llama33-probe-eval-v3` |
| Llama pre-report and matched length | `llama33-report-controls-v2` | Nested `pre_report` / `length_matched` |
| Llama early reasoning | `llama33-early-reasoning-v2` | Nested `probe`; full-prefix control in `llama33-full-prefix-v1` |
| Qwen processing Llama trading | `qwen35-llama-trading-a100-v1` | `qwen35-llama-trading-eval-v1` |
| Generated Qwen trading | `apollo-qwen-n100-v4`, `apollo-hf-9b-a100-n100-v2` | Manual reviews and residual comparisons |

## File formats

- JSON/JSONL: transcripts, trajectory/scenario IDs, labels, model configurations, original and evaluation splits, token positions, fidelity checks, and metrics.
- Safetensors: token-level residuals and pooled features. See each run's schema and manifest; dimensions differ across models and stages.
- NPZ/joblib: predictions, scalers, and logistic-regression probe weights. Load serialized Python objects only from a trusted source.
- PNG/PDF/Markdown: figures and interpretation.

No base-model weights, downloaded model caches, virtual environments, credentials, shell histories, or machine configuration are included. Revisions and runtime metadata are retained where available.

## Restore one archive group

Each group has a `SOURCE--GROUP.manifest.json` and one or more numbered `.tar.zst.part0000` files. Parts are at most 1 GiB. Download **every part for that group**, in addition to the release's `upload-ledger.json`. The ledger lists URLs, byte sizes, and SHA-256 hashes verified against GitHub's uploaded asset digests. The manifest lists original file paths and SHA-256 hashes.

After verifying part hashes, concatenate in numeric order and extract into an empty destination:

```sh
mkdir restored-a100
cat a100--GROUP.tar.zst.part* | zstd -dc | tar -xf - -C restored-a100
```

Replace `GROUP` with the exact downloaded group name. Restore matching source code/inputs alongside a run if reproducing its original VM state. Manifests preserve paths relative to the project root. The archive process checks each reconstructed file against its source checksum before marking a group complete.

See [third-party notices](../THIRD_PARTY_NOTICES.md), [limitations](../docs/LIMITATIONS.md), and [reproduction](../docs/REPRODUCING.md). Apollo labels do not establish intent in a model merely replaying the text.
