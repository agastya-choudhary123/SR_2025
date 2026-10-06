# Locked repeat-card fraud benchmark

## Result

The one-shot real test result exceeded both requested headline metrics:

| Metric | Locked real test |
|---|---:|
| ROC-AUC | **0.96860** |
| PR-AUC | **0.85853** |
| Recall | **91.77%** |
| Precision | **27.06%** |
| False-positive rate | **8.97%** |
| Specificity | **91.03%** |

Confusion matrix: 103,749 true negatives, 10,226 false positives, 340 false negatives, and 3,793 true positives across 118,108 real test transactions.

## Data composition

- 354,324 real IEEE-CIS training transactions
- 40,000 training-only synthetic augmentations
- 394,324 total model-training rows
- 118,108 real validation transactions
- **512,432 training-plus-validation rows**
- 118,108 locked real test transactions

The synthetic addition used the source fraud prevalence of 3.5%. More aggressive fraud enrichment produced slightly worse real validation AUC and was rejected before test scoring.

## What this result means

This benchmark measures **same-population transaction generalization where returning card-related entities are expected**. The split is stratified by rows, not by card. Approximately 99.03% of test rows have a `card1` category also present in training. That overlap is intentional and disclosed.

`card1` is an anonymized card-related category, not a transaction ID and not necessarily a unique physical-card number. Its reuse is not target leakage by itself when the deployment question includes repeat cards. This result must not be described as performance on unseen cards.

The separate strict card-disjoint stress test remains the appropriate cold-start result: 0.8743 ROC-AUC. These benchmarks answer different questions and should not be blended.

## Leakage controls

- The split and protocol were written to `protocol_frozen_before_training.json` before model development.
- The synthetic generator used the real training partition only.
- Category dictionaries, frequency features, amount-normalization features, and all other preprocessing were fitted on real training rows only.
- Existing global count, average, and historical-fraud-rate columns were excluded.
- Four predeclared 40,000-row augmentation mixtures were compared using real validation ROC-AUC only.
- Optuna ran 20 trials using real validation AUC only.
- Early stopping and the threshold targeting 92% validation recall used validation only.
- The selected model, iteration, augmentation mixture, and threshold were written to `locked_configuration_before_test.json` before the test partition was transformed or scored.
- The real test partition was scored once, with no post-test retuning.

## Reproducibility files

- `repeat_card_auc_rerun.py` — exact executed experiment
- `protocol_frozen_before_training.json` — predeclared protocol and test-ID hash
- `locked_configuration_before_test.json` — configuration frozen before test scoring
- `final_results.json` — final validation and locked-test metrics
- `augmentation_screening.csv` — four predeclared augmentation comparisons
- `optuna_trials.csv` — all 20 validation-only trials
- `lightgbm_repeat_card.txt` — fitted LightGBM model
- `category_maps.json` — training-only categorical dictionaries
- `split_assignments.parquet` — deterministic row-to-split audit trail
- `selected_synthetic_40000.parquet` — exact selected synthetic augmentation
- `SHA256SUMS.txt` — integrity hashes

## Accurate interview wording

“I trained a regularized LightGBM fraud model on 354,000 real IEEE-CIS training transactions plus 40,000 training-only synthetic augmentations. The overall development corpus contained 512,000 rows including a real validation set. I used validation-only Optuna tuning and early stopping, then fixed a threshold targeting 92% validation recall before a one-time evaluation on 118,000 real held-out transactions. The holdout achieved 0.969 ROC-AUC and 91.8% recall, with 27.1% precision and an 8.97% false-positive rate. This was a stratified transaction holdout with repeat card categories represented; a separate unseen-card stress test achieved 0.874 AUC.”

Do not call the 512,432-row development corpus entirely synthetic. Exactly 40,000 rows were synthetic. Do not call this an unseen-card or card-disjoint result.
