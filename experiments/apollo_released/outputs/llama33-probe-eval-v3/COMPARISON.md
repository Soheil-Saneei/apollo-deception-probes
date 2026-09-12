# Qwen / Llama probe evaluation

| Model | Qwen AUROC | Qwen AUPRC | Qwen accuracy | Llama AUROC | Llama AUPRC | Llama accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Activation probe | 0.9802 | 0.9769 | 0.9107 | 1.0000 | 1.0000 | 0.9950 |
| TF-IDF + logistic regression | 0.8839 | 0.8565 | 0.8125 | 0.9966 | 0.9917 | 0.6650 |
| Shuffled activation (five-seed mean) | 0.5180 | 0.5429 | 0.4893 | 0.5286 | 0.3748 | 0.6080 |
| Length only | 0.4944 | 0.4903 | 0.5000 | 0.9860 | 0.9783 | 0.9500 |
| Constant/majority | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.3350 | 0.6650 |

AUPRC is average precision; deceptive is positive; accuracy threshold is 0.5.
Qwen: 259/56/56 scenarios (train/validation/test), 112 test completions, 50% deceptive.
Llama: 650/150/200 trajectories, 33.5% deceptive on test; original test IDs unchanged.
Qwen holds out roleplaying scenarios. Llama holds out trajectories from the same insider-trading setting.
This is not a controlled comparison of model capability: dataset, labels, context, and class prevalence differ.
Validation chooses C and layer; test metrics are not used for model selection.
Shuffled controls permute training labels only, with true validation/test labels, across seeds 42-46.
