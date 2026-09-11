# Direct Transformers backend

`run_transformers.py` runs the original `Qwen/Qwen3.5-9B` checkpoint directly in
PyTorch and calls the existing v4 trajectory loop. This preserves the supplied
conversation, v4 instructions, four simulated tools, successful-trade reporting
restriction, bounded format corrections, and reserved report turn. Existing
Ollama datasets and runner files are unchanged. Results form a new condition:
`apollo-interactive-v4-transformers-v1`.

## Current status and hardware

The real 9B checkpoint now runs successfully on a Lambda A100-SXM4-40GB.
Three pilot trajectories and all activation files were downloaded to
`outputs/apollo-hf-9b-a100-pilot/`. All eight captures passed finite-value,
residual-addition and token-alignment checks across 32 layers. See
[the pilot review](a100-pilot-review.md) for behavioral outcomes and the command.

The backend and 23 automated tests pass locally, including actual cached generation
with tiny randomly initialized Qwen3.5 models containing both block types.
`outputs/hf-backend-validation/` contains an additional integration check using
the real tokenizer/template and tiny random weights. These are implementation
tests, **not behavioral reproduction or a trained-model dataset**.

The current Mac has 16 GB RAM, insufficient for the full model's BF16 weights plus
working memory. The downloaded checkpoint index reports 19,306,216,416 bytes of
weights, about 19.3 GB on disk. A 32 GB NVIDIA GPU is preferable; 24 GB may require
careful context/capture limits. Disk space alone does not resolve the RAM limit.
For 9B, only tokenizer/configuration files were downloaded locally, not the weights.

### Smaller Mac pilot

`Qwen/Qwen3.5-2B` has been downloaded and loaded successfully on the CPU in BF16:
2,213,241,664 parameters, approximately 4.43 GB of parameter storage plus runtime
memory. It is a separate model-size condition, not a reproduction of 9B behavior.
Default asynchronous MPS loading exited with code 139 before generation; that
attempt is retained in `outputs/apollo-hf-2b-mac-pilot/`. The CPU pilot was stopped
because generation was slow, and remains in `outputs/apollo-hf-2b-mac-cpu-pilot/`.
Sequential loading succeeded on MPS with the following command (512-token cap):

```sh
HF_DEACTIVATE_ASYNC_LOAD=1 .venv-hf/bin/python experiments/apollo_trading/run_transformers.py \
  --model Qwen/Qwen3.5-2B --device mps --dtype bfloat16 \
  --count 1 --num-predict 512 \
  --output experiments/apollo_trading/outputs/apollo-hf-2b-mac-mps-sync-pilot
```

Use a new output directory for subsequent runs. Seed 1000 finished at the eight-turn
limit: repeated market-analysis calls, one malformed message attempt, no executed
trade or delivered Amy message. Nine generations took about 115 seconds combined,
excluding loading. This establishes local execution, not useful deceptive behavior.
All nine captures passed checks for finite values, 24 layers, 120 tensors per
inference, residual-addition identities and token alignment.

The pilot used `HF_DEACTIVATE_ASYNC_LOAD=1`; the original protocol file does not
record that environment variable, so this note supplies the loading provenance.
Original manifests also predate the MPS alignment clarification below.

## Start on the GPU machine

From the repository root, with Python 3.12 and a compatible NVIDIA driver:

```sh
uv venv --python 3.12 .venv-hf
uv pip install --python .venv-hf/bin/python -r experiments/apollo_trading/requirements-hf.txt
.venv-hf/bin/python -m unittest discover -s experiments/apollo_trading -p 'test_*.py'
.venv-hf/bin/python experiments/apollo_trading/run_transformers.py \
  --revision c202236235762e1c871ad0ccb60c8ee5ba337b9a \
  --device cuda --dtype bfloat16 --count 3 \
  --output experiments/apollo_trading/outputs/apollo-hf-v4-pilot
```

The first real run downloads weights into `.hf-cache`. The revision above pins
the tokenizer/configuration inspected during development; the runner records the
resolved model commit, package versions, actual parameter dtypes/devices, model
configuration, template, and code hashes. Never reuse an existing output directory.
Review pilot formatting and behavior before running `--count 100` in another
directory. GPU loading and generation were validated on the A100 pilot above.
PyTorch reference implementations of the linear-attention kernels are
correct but slower when optional optimized kernels are absent.

## Activations and exact token alignment

### Optional optimized NVIDIA kernels

Install the base requirements first. The A100 environment uses PyTorch CUDA 13.0;
Lambda's system compiler was CUDA 12.8, so building `causal-conv1d` requires the
matching compiler. From the remote project root:

```sh
~/.local/bin/uv pip install --python .venv-hf/bin/python \
  -c experiments/apollo_trading/requirements-hf.txt \
  flash-linear-attention==0.5.2 fla-core==0.5.2 ninja packaging wheel setuptools \
  'cuda-toolkit[nvcc]==13.0.3'
CUDA_HOME=/home/ubuntu/epistemic-deference/.venv-hf/lib/python3.12/site-packages/nvidia/cu13 \
  MAX_JOBS=4 ~/.local/bin/uv pip install --python .venv-hf/bin/python \
  --no-build-isolation -c experiments/apollo_trading/requirements-hf.txt causal-conv1d==1.7.0
```

New run protocols record installed package versions. Keep optimized runs in new
output directories because kernel changes can alter sampled behavior. The tiny CPU
tests explicitly select reference kernels; GPU correctness must also be checked
with real generation and saved activations. Optional pins are listed in
`requirements-hf-kernels.txt`.

Validated on the A100 in `outputs/apollo-hf-9b-a100-kernels-pilot/` (on the remote
instance): all four dispatcher implementations resolved to the installed
`causal_conv1d`/`fla` packages. Three trajectories completed without runtime errors;
one exhausted report validation, two delivered Amy messages. All ten activation
captures passed checks across 32 layers. All 23 CPU tests passed with explicit
reference-kernel selection.

Excluding each pilot's first response, observed aggregate throughput was 17.51
tokens/sec with optimized kernels versus 17.22 for the reference pilot. These
pilots generated different trajectories, so this is not a controlled benchmark
and does not establish a substantial speedup. First-response time increased from
8.80 to 47.13 seconds, including initial optimized-kernel compilation/autotuning.
Further speed work should profile the runner, including synchronous per-layer
activation transfers, rather than assume kernel installation solves the bottleneck.

### Captured tensors

Each inference saves the input messages/tools, exact rendered prompt, generation
configuration, full token IDs, raw decoded response, parsed tool calls, and
`*-activations.safetensors` plus a JSON manifest.

Hooks run **during actual generation**, not a separate forced replay. For every
decoder layer, including linear-attention and full-attention layers, they capture:

- `resid_pre`: residual before input normalization.
- `mixer_out`: projected attention/linear-attention contribution.
- `resid_mid`: residual after the mixer addition, before MLP normalization.
- `mlp_out`: MLP contribution before residual addition.
- `resid_post`: residual after the MLP addition, before next-layer/final normalization.

Each tensor is `[captured_generation_steps, hidden_width]` for batch size one.
The first row represents the **last prompt token**, which predicts the first
generated token. Subsequent rows represent previously generated tokens used to
predict the next one. The final emitted token is not fed back, so it has no captured
state in ordinary synchronous stopping. Transformers 5.17's MPS deferred stop
performs an extra forward on the final emitted token and discards its prediction.
That final state can appear in captures shorter than the capture limit; do not pair
it with an emitted next token. Exact sequence positions are listed in the activation manifest. These are
not all prompt-token activations or post-final-normalization hidden states.

By default, capture the first 32 generation forwards at all layers; generation
continues normally afterward. `--capture-steps 0` captures every generation step,
which can produce very large files. At 32 layers, width 4096 and five BF16 vectors,
32 steps require roughly 40 MiB per inference (mixed dtypes can increase this).
The Python `ActivationRecorder` exposes ordinary PyTorch hooks; module names/types
in the manifest identify locations for future steering or more targeted recording.
The current hooks only observe; no intervention is applied.

```python
import json
from safetensors.torch import load_file

stem = "experiments/apollo_trading/outputs/apollo-hf-v4-pilot/trajectory-00-turn-00-attempt-00"
tensors = load_file(stem + "-activations.safetensors")
metadata = json.load(open(stem + "-activations.json"))
print(metadata["positions_in_prompt_plus_generated"])
print([(name, tuple(value.shape)) for name, value in tensors.items()][:5])
```

## Backend differences to retain in interpretation

This is fresh generation, not a claim that Ollama responses will reproduce exactly.
BF16 replaces the Ollama Q4_K_M weights. The official HF template renders the same
message/tool objects; its token sequence can differ from Ollama's renderer.
The parser accepts Qwen's documented XML-style tool format and preserves all calls
for the shared exactly-one-call validator. Malformed wire syntax is retained and
rejected, never executed. Generation stops on model EOS or tokenizer end-of-message.

Temperature 1, top-p 1, top-k 0, repetition penalty 1, 2048 output-token limit,
16384 total context limit, and per-trajectory seeds are carried over. Native
thinking stays disabled. An explicit generation configuration avoids inheriting
different HF sampling defaults. Context overflow errors instead of silently
truncating the prefix. Seed identity does not imply identical RNG behavior across
backends.

Ollama's saved model metadata also inherited presence penalty 1.5. This runner
applies an additive 1.5 penalty once per unique token in the last 64 context tokens,
including prompt tokens. This is explicit and recorded; exact equivalence to
Ollama 0.33.3's effective sampler has not been established. `--presence-penalty`
allows a separately recorded alternative.

Local inference has no HTTP/API retry layer. Local failures are retained under
the shared loop's legacy `api_error` status and stop the batch. A raw response may
already be saved if a later activation write fails; consult the trajectory and
run-error files before treating a response as an executed action.

References: [Qwen model configuration](https://huggingface.co/Qwen/Qwen3.5-9B/blob/main/config.json),
[official tool template](https://huggingface.co/Qwen/Qwen3.5-9B/blob/main/chat_template.jinja),
[Transformers Qwen3.5 documentation](https://huggingface.co/docs/transformers/model_doc/qwen3_5).
