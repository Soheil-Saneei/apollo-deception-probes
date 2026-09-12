# Reproduce the analyses

## Inspect without a GPU

Begin with [results](../results/README.md). To regenerate the publication figures, use Python 3.11 or newer with Matplotlib 3.11.2 and NumPy:

```sh
python figures/publication/make_figures.py
```

The figure script reads committed metrics and review counts. It performs no fitting or inference.

## Restore artifacts

Follow the [data guide](../data/README.md) to select a run, download its parts, verify SHA-256 hashes, and restore original paths. Archives preserve source-specific files, including scripts that differ between VMs. Restore different snapshots into separate directories first.

For probing without replay, obtain pooled features, manifests, saved splits, and evaluation artifacts. Token-level work needs activation archives. Probe artifacts may include joblib/pickle files; load these only from a source you trust.

## Replay and probe commands

Exact commands, revisions, pooling spans, checks, and package details:

- [Matched insider trading](../experiments/apollo_released/QWEN_LLAMA_TRADING.md)
- [Llama replay](../experiments/apollo_released/LLAMA_REPLAY.md)
- [Report controls](../experiments/apollo_released/REPORT_CONTROLS.md)
- [Early reasoning](../experiments/apollo_released/EARLY_REASONING.md)
- [Qwen roleplaying](../experiments/apollo_roleplaying/README.md)
- [Llama roleplaying](../experiments/apollo_roleplaying/LLAMA_COMPARISON.md)
- [Qwen generation](../experiments/apollo_trading/TRANSFORMERS.md)

Install experiment-specific pinned requirements rather than mixing environments. Model weights and credentials are not distributed. Acquire checkpoints from their providers. Read saved configurations before substituting tokenizers or templates.

## CPU tests

With the numerical and replay dependencies installed, run directories separately because the research scripts use local imports:

```sh
python -m unittest discover -s experiments/apollo_trading -p 'test_*.py'
python -m unittest discover -s experiments/apollo_roleplaying -p 'test_*.py'
python -m unittest discover -s experiments/apollo_released -p 'test_*.py'
```

Tests exercise protocol validation, replay alignment, labels/splits, pooling, and classifier controls; they do not rerun full GPU experiments.
