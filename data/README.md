# Dataset

`ieee_fraud_features.parquet` contains 590,540 processed IEEE-CIS Fraud Detection training transactions and 72 columns.

The original IEEE-CIS data represents real-world e-commerce transactions supplied by Vesta Corporation; it is not a synthetic Kaggle dataset. This project generates synthetic augmentation from the training partition only.

The experiment excludes target-derived/global aggregate columns before modeling. See the experiment scripts and artifact READMEs for the exact feature and evaluation protocols.

## Getting the data

The parquet files are not included in this repository, because the IEEE-CIS competition data can't be redistributed. Download the source data from the [IEEE-CIS Fraud Detection competition](https://www.kaggle.com/c/ieee-fraud-detection/data) on Kaggle, and place the processed table at `data/ieee_fraud_features.parquet`. The synthetic-row and split-assignment parquet files under `artifacts/` are also excluded; the experiment scripts regenerate them.
