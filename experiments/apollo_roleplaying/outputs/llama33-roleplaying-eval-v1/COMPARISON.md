# Matched Apollo roleplaying comparison

| Method | Qwen AUROC | Qwen AUPRC | Qwen accuracy | Llama AUROC | Llama AUPRC | Llama accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Activation | 0.9802 | 0.9769 | 0.9107 | 0.9901 | 0.9898 | 0.9018 |
| TF-IDF | 0.8839 | 0.8565 | 0.8125 | 0.8839 | 0.8565 | 0.8125 |
| Shuffled activation, five-seed mean | 0.5180 | 0.5429 | 0.4893 | 0.5439 | 0.5568 | 0.5196 |
| Length | 0.4944 | 0.4903 | 0.5000 | 0.4917 | 0.4883 | 0.4732 |
| Constant | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 |

Both models: identical 371 paired scenarios; 259/56/56 scenario train/validation/test split.
Deceptive is positive. AUPRC is average precision; accuracy threshold is 0.5.
Layer/C selection uses validation only. All layers are shown descriptively, not selected by test scores.
Both are frozen, BF16, eager attention, exact teacher-forced completions, pairwise right padding, FP32 completion means.
Native tokenizers/templates, model architectures, width/depth, and GPU placement differ.
This is off-policy replay of released Llama-3.1-generated answers, not a test of spontaneous deception.
