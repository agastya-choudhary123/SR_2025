# Defensible card-disjoint LightGBM rerun

## Bottom line

This rerun measures **cold-start generalization to card1 groups never seen during training or validation**. It does not reproduce the earlier 93% AUC / 90% recall claim.

The locked test result was:

| Operating point fixed on validation | ROC-AUC | PR-AUC | Recall | Precision | False-positive rate |
|---|---:|---:|---:|---:|---:|
| Target 92% validation recall | 0.8743 | 0.4816 | 95.79% | 5.16% | 63.88% |
| Maximum validation recall at no more than 5% validation FPR | 0.8743 | 0.4816 | 61.31% | 24.78% | 6.75% |

The high-recall operating point is not operationally attractive: it catches most fraud, but incorrectly flags nearly 64% of legitimate transactions. The lower-FPR point is much more selective but misses about 39% of fraud.

## What `card1` means

In IEEE-CIS, `card1` is an anonymized card-related categorical field. It is not the transaction ID and should not be described as a literal card number or guaranteed unique physical-card identifier. Multiple transactions share the same value.

Using `card1` as a feature is **not automatically target leakage**. It can be legitimate when the deployment population includes repeat transactions from card entities observed during training. However, a random row split places nearly all `card1` values on both sides of the split. That lets the model exploit stable card-specific risk patterns and makes the result optimistic for new, unseen card groups.

There are two distinct evaluation problems to keep separate:

1. **Entity overlap:** the same card-related groups occur in train and test. This is not always leakage, but it answers an easier “repeat-card” question rather than a cold-start question.
2. **Test-set selection:** earlier experiments compared model variants and adjusted the threshold buffer after observing test results. That is test contamination and means those earlier test metrics must be treated as development results, not as an unbiased final estimate.

## Locked protocol

- 590,540 source rows were split approximately 60/20/20 with nested `StratifiedGroupKFold` using `card1` as the group.
- Train, validation, and locked test contain zero overlapping `card1` groups.
- Category vocabularies were learned from training groups only. Unseen validation/test categories map to `-1`.
- Full-dataset aggregate features were excluded: card/email counts, average amount, historical fraud-rate features, the derived high-risk-product flag, and amount-versus-card-average.
- Exactly 568,630 synthetic training rows were generated only from the training partition using class-conditional kernel bootstrap.
- Optuna ran 12 trials using validation ROC-AUC only. Validation alone controlled early stopping and threshold selection.
- The chosen model and thresholds were written to `locked_configuration_before_test.json` before the first test prediction.
- The test set was then scored once, without post-test retuning.

Split sizes were 354,323 training-source rows, 118,109 validation rows, and 118,108 locked-test rows. Their respective `card1` group counts were 8,136, 2,713, and 2,704, with zero pairwise overlap.

## Files

- `defensible_group_rerun.py` — complete reproducible experiment
- `locked_configuration_before_test.json` — model and thresholds frozen before test scoring
- `final_results.json` — validation and one-shot locked-test metrics
- `split_manifest.json` — split sizes, group counts, fraud rates, and transaction-ID hashes
- `split_assignments.parquet` — transaction-to-split audit trail
- `optuna_trials.csv` — all 12 validation-only trials
- `lightgbm_card1_grouped.txt` — fitted LightGBM model
- `category_maps.json` — training-only categorical encodings
- `synthetic_train_568630.parquet` — generated training data
- `SHA256SUMS.txt` — integrity hashes for every experiment artifact

## Defensible interview statement

“A random row split produced strong metrics, but an audit showed almost complete overlap of the anonymized card-related groups and that the test set had been consulted during model selection. I therefore treated those numbers as development results. I reran the pipeline with card-group-disjoint train, validation, and locked-test partitions; fit preprocessing and synthetic generation on training only; used validation only for Optuna, early stopping, and threshold selection; and scored the test once. On unseen card groups, ROC-AUC was 0.874. Achieving 95.8% recall required a 63.9% false-positive rate, while the lower-FPR operating point achieved 61.3% recall at 6.75% FPR. That showed the original result did not generalize to cold-start cards.”

## Scope note

This is a strict unseen-`card1` stress test, not the only valid production evaluation. A real fraud system should also use a chronological holdout to measure future transactions, report performance separately for previously seen and unseen entities, and compute all history/velocity features using past data only.
