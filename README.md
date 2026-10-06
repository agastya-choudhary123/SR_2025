# IEEE-CIS LightGBM Fraud Detection

This repository contains one primary benchmark and one stricter stress test. Both use the same processed IEEE-CIS source data, LightGBM, Optuna, early stopping, training-only preprocessing, and real held-out evaluation data.

## Primary result: repeat-card transaction benchmark

The primary experiment measures held-out transactions from the same population, where returning anonymized card-related categories are expected.

| Metric | Locked real test |
|---|---:|
| ROC-AUC | **96.86%** |
| Recall | **91.77%** |
| Precision | **27.06%** |
| False-positive rate | **8.97%** |
| PR-AUC | **85.85%** |

Training used 354,324 real transactions plus 40,000 training-only synthetic augmentations. Validation used 118,108 real transactions, producing a 512,432-row development corpus. Final evaluation used an additional 118,108 real transactions.

Approximately 99% of test rows share a `card1` category with training. This overlap is intentional and disclosed: the result measures repeat-card transaction generalization, not unseen-card cold starts.

Full evidence is in `artifacts/repeat_card_benchmark/`, including the pre-training protocol, pre-test lock, all Optuna trials, exact synthetic rows, split assignments, fitted model, results, and checksums.

## Unseen-card stress test

The stricter stress test places every `card1` group in exactly one partition. Its locked real-test ROC-AUC is **87.43%**. At the high-recall operating point it reaches 95.79% recall, but with a 63.88% false-positive rate. At the lower-FPR point it reaches 61.31% recall at 6.75% FPR.

This result is stored under `artifacts/unseen_card_stress_test/`. It answers a different question and must not be blended with the primary benchmark.

## Repository layout

```text
data/
  ieee_fraud_features.parquet
  README.md
experiments/
  repeat_card_auc_rerun.py
  unseen_card_stress_test.py
artifacts/
  repeat_card_benchmark/
  unseen_card_stress_test/
requirements.txt
```

## Reproduce

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python experiments/repeat_card_auc_rerun.py
```

The unseen-card stress test is run with:

```bash
python experiments/unseen_card_stress_test.py
```

Both scripts write only to their corresponding artifact directory.

## Accurate summary

> Built a regularized LightGBM credit-card-fraud model using a 512K-row development corpus derived from IEEE-CIS, including 40K training-only synthetic augmentations. Validation-only Optuna tuning and early stopping were followed by a one-time evaluation on a 118K-row real holdout, producing 96.9% ROC-AUC and 91.8% recall, with 27.1% precision and an 8.97% false-positive rate. The primary benchmark represents repeat-card transactions; a separate card-disjoint cold-start stress test achieved 87.4% AUC.

The project must not be described as an entirely synthetic 512K-row dataset. Exactly 40,000 development rows are synthetic; validation and test are real.
