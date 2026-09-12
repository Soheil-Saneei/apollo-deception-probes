# Results guide

These links point to completed evaluations. JSON files contain unrounded metrics and layerwise results. Paired bootstrap comparisons and caveats are in the linked write-ups.

| Question | Write-up | Machine-readable results |
| --- | --- | --- |
| Do models separate roleplaying labels across held-out scenarios? | [Matched roleplaying](../experiments/apollo_roleplaying/LLAMA_COMPARISON.md) | [Qwen](../experiments/apollo_roleplaying/outputs/qwen9b-sanity-v1/metrics.json), [Llama](../experiments/apollo_roleplaying/outputs/llama33-roleplaying-eval-v1/metrics.json) |
| How do models compare on identical insider-trading texts? | [Matched insider trading](../experiments/apollo_released/QWEN_LLAMA_TRADING.md) | [All stages](../experiments/apollo_released/outputs/qwen35-llama-trading-eval-v1/comparison.json) |
| Is the report probe confounded by text and length? | [Initial evaluation](../experiments/apollo_released/EVALUATION.md) | [Metrics](../experiments/apollo_released/outputs/llama33-probe-eval-v3/metrics.json) |
| What happens before reporting or after length matching? | [Report controls](../experiments/apollo_released/REPORT_CONTROLS.md) | [Pre-report](../experiments/apollo_released/outputs/llama33-report-controls-v2/pre_report/metrics.json), [Length matched](../experiments/apollo_released/outputs/llama33-report-controls-v2/length_matched/metrics.json) |
| Do early activations add information over the full text prefix? | [Early reasoning](../experiments/apollo_released/EARLY_REASONING.md) | [Full-prefix comparison](../experiments/apollo_released/outputs/qwen35-llama-trading-eval-v1/early_full_prefix.json) |
| Did Qwen generate clearly deceptive reports? | [Ollama v4](../experiments/apollo_trading/qwen-n100-v4-review.md), [HF A100](../experiments/apollo_trading/a100-n100-v2-review.md) | [Figure values](../figures/publication/plotted_data.json) |

## Figures

1. [Qwen behavior](../figures/publication/01_qwen_trading_behavior.png) ([PDF](../figures/publication/01_qwen_trading_behavior.pdf))
2. [Roleplaying depth](../figures/publication/02_roleplaying_depth.png) ([PDF](../figures/publication/02_roleplaying_depth.pdf))
3. [Insider-trading stages](../figures/publication/03_trading_stages.png) ([PDF](../figures/publication/03_trading_stages.pdf))
4. [Report confound](../figures/publication/04_report_surface_confound.png) ([PDF](../figures/publication/04_report_surface_confound.pdf))

Do not pool the generated-Qwen behavior study with the frozen replay studies. The former uses manual review criteria; the latter preserves Apollo's construction/grader labels.
