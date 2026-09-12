# Qwen roleplaying sanity checks

Fixed split: 259 train / 56 validation / 56 test scenarios. Deceptive is positive.
AUPRC is average precision. Layer and C selection use validation AUROC only.

| Model | Test AUROC | Test AUPRC | Test accuracy |
| --- | ---: | ---: | ---: |
| Activation | 0.9802 | 0.9769 | 0.9107 |
| TF-IDF text | 0.8839 | 0.8565 | 0.8125 |
| Shuffled activation (seed 42) | 0.4818 | 0.4938 | 0.4732 |
| Length only | 0.4944 | 0.4903 | 0.5000 |
| Constant | 0.5000 | 0.5000 | 0.5000 |

Activation minus text, paired-scenario bootstrap 95% intervals:
- auroc: 0.0963 [0.0443, 0.1582]
- auprc: 0.1204 [0.0499, 0.1924]
- accuracy: 0.0982 [0.0357, 0.1696]

Shuffled-label results across five seeds (mean +/- sample SD):
- auroc: 0.5180 +/- 0.0729
- auprc: 0.5429 +/- 0.0781
- accuracy: 0.4893 +/- 0.0639
