# Frozen Llama 3.3 report-stage replay

Public transcripts and activation downloads are indexed in the
[data guide](../../data/README.md). VM paths below are historical source locations;
the release archives preserve those paths relative to the project root.

The extraction is complete. The matched Qwen/Llama probe evaluation and its
surface-feature controls are documented in [EVALUATION.md](EVALUATION.md).

## Current status

The runner and audited data are installed on `ubuntu@192.222.53.243`, with four
48 GiB RTX A6000 GPUs. This fits the unquantized FP16 weights across the four
devices. All eight tests pass there, including a frozen tiny-Llama end-to-end
activation test and structural/span checks on all 1,000 released trajectories.

**Access to both repositories now works. The four-trajectory pilot passed with
Apollo's 3.1 tokenizer, and the full 1,000-case extraction is complete** in
`outputs/llama33-reports-v1/`. It preserves the 666 honest / 334 deceptive labels
and 800/200 grouped train/test split. Pilot hooks left logits unchanged, repeated
forwards were identical, and maximum full/prefix relative L2 was 0.003506 (0.3506%).

A separate real-70B pilot using the accessible **3.3 tokenizer** passed:
`outputs/llama33-native-tokenizer-pilot-v1/`. It contains two honest trajectories
(raw IDs 4 and 20) and two deceptive trajectories (10 and 15), with all 80 layers
and width 8192. Hooks did not alter logits; repeated forwards were identical.
The largest full/prefix relative L2 was 0.003506 (0.3506%), below the predeclared
0.5% threshold. Full token activations remain on the VM; configuration and fidelity
JSON files were copied locally (the local `fidelity.json` is from remote `pilot/`).

The 3.3 tokenizer matched all four released report token-string examples, and all
1,000 transcripts round-tripped. Its tokenizer files nevertheless differ from
3.1's published file identities. Now that both are accessible, direct comparison
confirms identical input token IDs and report positions for all 1,000 transcripts
(`tokenizer_comparison.json`). The full run nevertheless uses Apollo's specified
3.1 tokenizer and has its own pilot gate.

Downloaded weight revision: `6f6073b423013f6a7d4d9f39144961bfbfbc386b`.
The diagnostic uses that same revision's tokenizer. The currently resolved 3.1
tokenizer revision is `1605565b47bb9346c5515c34102e054115b4f98b`.

Authenticate in the remote terminal using an account with access to both model
repositories, without putting the token in a command or sharing it in chat:

```sh
~/epistemic-deference/.venv-llama/bin/hf auth login
```

Then, from `~/epistemic-deference`, run:

```sh
USE_TF=0 .venv-llama/bin/python -u experiments/apollo_released/run_llama_reports.py --output experiments/apollo_released/outputs/llama33-reports-v1
```

The default command loads the model once, runs the small pilot, and scales to the
full cohort only if the pilot passes. `--pilot-only` stops after the pilot. Use a
fresh output directory for each new run; completed original audit data are never
modified. If interrupted, the incremental completed manifest and per-case files
remain available, but automatic resume is not implemented.

The separate diagnostic used `--pilot-only --tokenizer meta-llama/Llama-3.3-70B-Instruct`.
An explicit `--tokenizer` override is always recorded in the run configuration;
the default remains Apollo's 3.1 tokenizer.

## Cohort, IDs, and split

Input is the existing byte-verified audit at
`apollo-released-llama33-manifest/` beside the script, copied from the local
`experiments/apollo_trading/outputs/apollo-released-llama33-manifest/` directory.
The [audit README](README.md) explains Apollo's original grader and exclusions.

The selected cohort is exactly **666 honest / 334 deceptive reports**. These map
Apollo's report labels `explicit -> honest`, `concealed -> deceptive`, with the
prior strict filter requiring an unambiguous insider trade and structurally valid
report. The runner checks source transcript hashes and the original labels.
It does not regenerate, relabel, or include the later confirmation reply.

Stable trajectory ID, raw source index, Apollo loader ID, original report label,
trade label, transcript path and hash are preserved in the new manifest.
The original raw transcript and grader rationales remain in the audit files.

An 80/20 stratified split uses RNG seed 42 and groups by raw trajectory index:

| Split | Honest | Deceptive | Total |
| --- | ---: | ---: | ---: |
| Train | 533 | 267 | 800 |
| Test | 133 | 67 | 200 |

Every token and report feature from a trajectory inherits its one split. No
trajectory can occur in both. These are alternative rollouts from a common
trading scenario, so this is not a held-out-scenario generalization test. Future
probe hyperparameter selection should use grouped validation within the training
set; do not select layers or settings using the held-out test set.

## Model and reconstruction

- Weights: `meta-llama/Llama-3.3-70B-Instruct`.
- Tokenizer: `meta-llama/Meta-Llama-3.1-70B-Instruct`, as used by Apollo's analysis.
- Both current Hub revisions are resolved to immutable SHAs before loading and
  recorded in the run config. Apollo did not publish the original revisions.
- FP16, no quantization, eager attention, batch size one, balanced placement
  across four GPUs, 43 GiB maximum weight-allocation budget per GPU.
- `.eval()`, `.requires_grad_(False)`, `torch.inference_mode()`, `use_cache=False`.
  CPU/disk model offload is rejected; no optimizer or sampling is used.
- Normalize message whitespace and merge same-role messages according to the
  released Apollo loader. Preserve raw text separately in the source audit.
- Apply the pinned Llama 3.1 chat template, with `add_generation_prompt=False`
  and `date_string="26 Jul 2024"`, matching the released visualizations. Tokenize
  without extra special tokens or truncation. The full saved report remains in
  the teacher-forced input, including its emitted reasoning and final EOT.

The release contains saved text, not original generation token IDs, logits, or
activations. Together's original serving revision and precision are unavailable.
This is **text-reconstructed frozen analysis replay**, not verified equivalence
to the provider that generated the answers.

## Pilot gate

Before the full extraction, the runner:

1. Checks text round-trips and report token spans for all 1,000 trajectories.
2. Checks both rendered text and individual token strings against the four
   released report visualization overlaps (two distinct source trajectories).
   These are independent formatting checks; those example trajectories are not
   added to the strict labeled cohort.
3. Extracts all layers for the first two honest and first two deceptive cases.
4. Repeats the full forward and requires exactly matching logits and pooled
   features; checks finite activations.
5. Compares full-context report-boundary residuals with a teacher-forced prefix
   ending at the same token. Per-layer relative L2 must be <=0.005. Full/prefix
   matrix shapes can change FP16 numerics; all differences are recorded.
6. Requires exactly matching logits with and without activation hooks.

Failure prevents the full loop. Results are written to
`tokenizer_fidelity.json` and `pilot/fidelity.json`. These checks establish internal
consistency and agreement with released formatting, not original provider logits.

## Activation and probe-ready files

Hooks capture **resid_post**, the unnormalized output of all 80 decoder blocks,
before the next block/final normalization. Layer indices are 0-79. This differs
from indexing the model's normalized final `hidden_states` entry.

Each `activations/<label>/<trajectory_id>.safetensors` contains:

- `resid_post`: FP16 `[80, final_assistant_tokens, 8192]`, including the report's
  reasoning and action text, but excluding assistant header and EOT.
- `assistant_boundary`, `action_boundary`, `report_boundary`: FP32 `[80,8192]`.
- `assistant_mean`, `action_mean`, `report_mean`: FP32 `[80,8192]`.

`assistant_boundary` is before the final assistant content, `action_boundary`
before `Action:`, and `report_boundary` before the token overlapping the first
character of `Executed trade update`. A boundary token has not observed that
region's first token. The manifest includes exact reconstructed input token IDs,
token positions, character offsets, split, labels and activation path.

`report_mean` averages from `Executed trade update` through the final assistant
content, including closing tool-call punctuation but excluding EOT. `action_mean`
starts at `Action:`; both retain preceding reasoning in the context. All means
are computed in FP32. Report/body outputs have already observed their own tokens;
they must not be interpreted as predictions made before report generation.

At completion, `features.safetensors` stacks all six features into
`[1000,80,8192]`, in `manifest.jsonl` order, and includes `labels` (0 honest,
1 deceptive), `train_mask`, and `raw_indices`. This is ready to feed to a linear
probe without parsing individual token activation files. Fit any scaler on
training data only. The summary verifies cohort and split counts.

`planned_manifest.jsonl` lists every intended example before extraction;
`manifest.jsonl` lists only successfully completed cases. Per-case writes use a
temporary file followed by rename. `config.json` records source provenance,
package versions, checkpoint SHAs, tokenization and extraction settings;
`device_map.json` records actual weight placement.

Full token activations may take hundreds of GB; the script estimates space after
tokenizing and checks for space for the model and outputs before download.
The six pooled features together require about 15.7 GB. Keep the large token
files on the GPU instance and download selected pooled features as needed.

## Environment and tests

The instance's existing torch 2.7.0/CUDA 12.8 is reused in a system-site-packages
virtual environment. Added packages are pinned in `requirements-llama.txt`.
This is a new, recorded runtime, not Apollo's original unpinned environment.

```sh
USE_TF=0 .venv-llama/bin/python -m unittest discover -s experiments/apollo_released -p 'test_*.py'
```
