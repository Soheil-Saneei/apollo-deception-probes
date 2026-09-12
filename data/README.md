# Dataset and artifact downloads

The public repository contains code, figures, compact text datasets, and small metrics. Full research artifacts are stored in [GitHub Releases](https://github.com/Soheil-Saneei/apollo-deception-probes/releases), divided by source machine. **All three source snapshots are complete and independently verified.**

| Release | Status | Contents |
| --- | --- | --- |
| [A100](https://github.com/Soheil-Saneei/apollo-deception-probes/releases/tag/artifacts-a100-v1) | Verified complete | Qwen generated trading, both roleplaying versions, Qwen replay of Llama trading, features, probes, evaluations, and source inputs |
| [Llama VM](https://github.com/Soheil-Saneei/apollo-deception-probes/releases/tag/artifacts-a6000-v1) | Verified complete | Llama report and roleplaying activations, early reasoning, report controls, evaluation versions, tokenizer pilot, and source inputs |
| [Local](https://github.com/Soheil-Saneei/apollo-deception-probes/releases/tag/artifacts-local-v1) | Verified complete | Ollama runs, Mac pilots, manual-review artifacts, local analyses, and local-only files |

These are overlapping source snapshots, not disjoint datasets. Keep them separate on initial extraction. Historical/pilot outputs are preserved for provenance, not silently pooled with final results.

## Verification and archive sizes

| Snapshot | Files | Archive groups | Original bytes | Uploaded bytes |
| --- | ---: | ---: | ---: | ---: |
| A100 | 5,835 | 16 | 77,269,999,895 | 62,019,155,451 |
| Llama VM | 5,243 | 14 | 262,311,102,738 | 223,932,743,880 |
| Local | 7,178 | 40 | 1,615,379,079 | 836,157,318 |

This totals 18,256 file records across overlapping snapshots, about 317.8 GiB before compression and 267.1 GiB uploaded. Each archive was reconstructed and every member checked against its original filename, byte size, and SHA-256 hash. An independent unauthenticated GitHub API check verified the size and SHA-256 digest of every published asset and confirmed complete inventory coverage, with no unexpected assets. A public download-and-restore sample also passed.

Completed ledgers and independent checks are committed under `catalog/`: [A100 ledger](catalog/a100-upload-ledger.json), [Llama ledger](catalog/a6000-upload-ledger.json), [local ledger](catalog/local-upload-ledger.json); [A100 check](catalog/a100-release-check.json), [Llama check](catalog/a6000-release-check.json), [local check](catalog/local-release-check.json). Package inventories are included alongside them.

## Browse the text datasets without activation downloads

| Table | Rows | Contents |
| --- | ---: | --- |
| [Roleplaying completions](roleplaying/completions.jsonl) | 742 | Exact completion text, scenario IDs, messages, honest/deceptive labels, and scenario-grouped splits |
| [Insider-trading reports](insider_trading/reports.jsonl) | 1,000 | Llama-generated message transcripts, report text, original Apollo IDs/labels, and unchanged evaluation splits |
| [Early windows and full prefixes](insider_trading/early_windows.jsonl) | 999 | Exact eight-token windows, full text through the same endpoint, labels, IDs, and splits |

Each line is one JSON object. The report table's `report_token_count` is the original Llama count. Early window token IDs use the Llama tokenizer, and `window_char_range` indexes its saved final assistant text. These tables are compact views of saved manifests, not new generations or annotations. [Table provenance](table_provenance.json) records source hashes and counts; `scripts/export_public_tables.py` reproduces them after restoring its source manifests.

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

After downloading `upload-ledger.json` from the chosen release, use the helper to download and verify one group:

```sh
python scripts/download_artifact.py upload-ledger.json \
  apollo_roleplaying-qwen9b-paired-v2 --output downloads/qwen-roleplaying
```

See [A100 groups](catalog/a100-groups.json), [Llama VM groups](catalog/a6000-groups.json), and [local groups](catalog/local-groups.json) for archive names and uncompressed sizes. A completed group ledger is required by the helper; partial releases do not establish a complete VM backup.

After verifying part hashes, concatenate in numeric order and extract into an empty destination:

```sh
mkdir restored-a100
cat a100--GROUP.tar.zst.part* | zstd -dc | tar -xf - -C restored-a100
```

Replace `GROUP` with the exact downloaded group name. Restore matching source code/inputs alongside a run if reproducing its original VM state. Manifests preserve paths relative to the project root. The archive process checks each reconstructed file against its source checksum before marking a group complete.

See [third-party notices](../THIRD_PARTY_NOTICES.md), [limitations](../docs/LIMITATIONS.md), and [reproduction](../docs/REPRODUCING.md). Apollo labels do not establish intent in a model merely replaying the text.
