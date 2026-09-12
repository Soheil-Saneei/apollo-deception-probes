# Frozen Qwen processing released Llama insider-trading trajectories

Qwen teacher-forces the saved text; it does not generate these trajectories.
All cohorts, labels, train/validation/test assignments and text-baseline inputs match Llama.
Native tokenizers/templates and model architectures differ. Both replays use FP16.
The length-matched cohort was selected using Llama token counts. Qwen lengths need not match.
Early windows match exact characters, so Qwen window token counts need not equal eight.

## report

| Method | Llama AUROC | AUPRC | Accuracy | Qwen AUROC | AUPRC | Accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Activation | 1.0000 | 1.0000 | 0.9950 | 1.0000 | 1.0000 | 0.9950 |
| Shared TF-IDF | 0.9966 | 0.9917 | 0.6650 | 0.9966 | 0.9917 | 0.6650 |
| Shuffled, five-seed mean | 0.5286 | 0.3748 | 0.6080 | 0.5774 | 0.4030 | 0.6330 |
| Own-tokenizer length | 0.9860 | 0.9783 | 0.9500 | 0.9861 | 0.9784 | 0.9550 |
| Constant | 0.5000 | 0.3350 | 0.6650 | 0.5000 | 0.3350 | 0.6650 |

Validation-selected layers (zero-based): Llama 0, Qwen 0.

## llama_length_matched

| Method | Llama AUROC | AUPRC | Accuracy | Qwen AUROC | AUPRC | Accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Activation | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Shared TF-IDF | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Shuffled, five-seed mean | 0.6449 | 0.6796 | 0.5571 | 0.6694 | 0.6775 | 0.5714 |
| Own-tokenizer length | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 |
| Constant | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 |

Validation-selected layers (zero-based): Llama 0, Qwen 0.

## pre_report

| Method | Llama AUROC | AUPRC | Accuracy | Qwen AUROC | AUPRC | Accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Activation | 0.9912 | 0.9800 | 0.9650 | 0.9872 | 0.9585 | 0.9500 |
| Shared TF-IDF | 0.9942 | 0.9788 | 0.9750 | 0.9942 | 0.9788 | 0.9750 |
| Shuffled, five-seed mean | 0.5544 | 0.4189 | 0.6150 | 0.5596 | 0.4197 | 0.6520 |
| Own-tokenizer length | 0.8366 | 0.7066 | 0.7700 | 0.8368 | 0.7049 | 0.7700 |
| Constant | 0.5000 | 0.3350 | 0.6650 | 0.5000 | 0.3350 | 0.6650 |

Validation-selected layers (zero-based): Llama 1, Qwen 1.

## early_reasoning

| Method | Llama AUROC | AUPRC | Accuracy | Qwen AUROC | AUPRC | Accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Activation | 0.8597 | 0.7460 | 0.7600 | 0.8666 | 0.7384 | 0.7800 |
| Shared TF-IDF | 0.7862 | 0.6382 | 0.8100 | 0.7862 | 0.6382 | 0.8100 |
| Shuffled, five-seed mean | 0.5529 | 0.3806 | 0.5820 | 0.5616 | 0.3883 | 0.5830 |
| Own-tokenizer length | 0.5000 | 0.3350 | 0.6650 | 0.5000 | 0.3350 | 0.6650 |
| Constant | 0.5000 | 0.3350 | 0.6650 | 0.5000 | 0.3350 | 0.6650 |

Validation-selected layers (zero-based): Llama 17, Qwen 16.
