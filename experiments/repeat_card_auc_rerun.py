#!/usr/bin/env python3
"""Locked repeat-card IEEE-CIS benchmark with modest synthetic augmentation.

This experiment intentionally answers the repeat-card / same-population question.
It does not claim cold-start performance on unseen card1 groups.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]
SEED = 20251006
SOURCE = ROOT / "data" / "ieee_fraud_features.parquet"
OUT = ROOT / "artifacts" / "repeat_card_benchmark"
N_SYNTHETIC = 40_000
SYNTHETIC_FRAUD_FRACTIONS = (0.035, 0.10, 0.25, 0.50)
N_OPTUNA_TRIALS = 20

EXCLUDED = [
    "transaction_id", "transaction_ts", "card1_historical_fraud_rate",
    "email_historical_fraud_rate", "card1_txn_count", "card1_avg_amt",
    "email_txn_count", "is_high_risk_product", "amt_vs_card_avg_ratio",
]

KNOWN_CATEGORICAL = [
    "product_cd", "card1", "card2", "card3", "card4", "card5", "card6",
    "addr1", "addr2", "purchaser_email_domain", "recipient_email_domain", "device_type",
]


def id_hash(values: pd.Series) -> str:
    joined = "\n".join(map(str, sorted(values.tolist()))).encode()
    return hashlib.sha256(joined).hexdigest()


def freeze_split(frame: pd.DataFrame):
    all_idx = np.arange(len(frame))
    y = frame["is_fraud"].astype("int8").to_numpy()
    dev_idx, test_idx = train_test_split(
        all_idx, test_size=0.20, random_state=SEED, stratify=y
    )
    train_idx, valid_idx = train_test_split(
        dev_idx, test_size=0.25, random_state=SEED + 1, stratify=y[dev_idx]
    )
    assert len(train_idx) + len(valid_idx) + len(test_idx) == len(frame)
    assert not (set(train_idx) & set(valid_idx))
    assert not (set(train_idx) & set(test_idx))
    assert not (set(valid_idx) & set(test_idx))
    return np.sort(train_idx), np.sort(valid_idx), np.sort(test_idx)


class FeatureBuilder:
    def __init__(self) -> None:
        self.base_columns: list[str] = []
        self.categorical: list[str] = []
        self.category_maps: dict[str, dict[str, int]] = {}
        self.frequency_maps: dict[str, dict[str, float]] = {}
        self.amount_medians: dict[str, dict[str, float]] = {}
        self.global_amount_median = 0.0

    @staticmethod
    def _string(series: pd.Series) -> pd.Series:
        return series.astype("string").fillna("__NA__")

    def fit(self, train: pd.DataFrame) -> "FeatureBuilder":
        self.base_columns = [
            c for c in train.columns
            if c != "is_fraud" and c not in EXCLUDED
        ]
        self.categorical = list(
            dict.fromkeys(
                list(train[self.base_columns].select_dtypes(include=["object", "string"]).columns)
                + [c for c in KNOWN_CATEGORICAL if c in self.base_columns]
            )
        )
        for col in self.categorical:
            values = self._string(train[col])
            self.category_maps[col] = {
                str(value): int(code) for code, value in enumerate(pd.Index(values.unique()))
            }

        frequency_keys = {
            "card1_freq": self._string(train["card1"]),
            "card12_freq": self._string(train["card1"]) + "|" + self._string(train["card2"]),
            "card_tuple_freq": (
                self._string(train["card1"]) + "|" + self._string(train["card2"]) + "|"
                + self._string(train["card3"]) + "|" + self._string(train["card4"]) + "|"
                + self._string(train["card5"]) + "|" + self._string(train["card6"])
            ),
            "pemail_freq": self._string(train["purchaser_email_domain"]),
            "remail_freq": self._string(train["recipient_email_domain"]),
            "addr1_freq": self._string(train["addr1"]),
            "device_freq": self._string(train["device_type"]),
            "product_card4_freq": self._string(train["product_cd"]) + "|" + self._string(train["card4"]),
        }
        n = float(len(train))
        for name, values in frequency_keys.items():
            counts = values.value_counts(dropna=False) / n
            self.frequency_maps[name] = {str(k): float(v) for k, v in counts.items()}

        self.global_amount_median = float(train["transaction_amt"].median())
        for col in ("product_cd", "card4", "purchaser_email_domain"):
            keys = self._string(train[col])
            medians = train.assign(_key=keys).groupby("_key", observed=True)["transaction_amt"].median()
            self.amount_medians[col] = {str(k): float(v) for k, v in medians.items()}
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        X = frame[self.base_columns].copy()
        for col in self.categorical:
            X[col] = (
                self._string(frame[col]).map(self.category_maps[col]).fillna(-1).astype("int32")
            )

        card1 = self._string(frame["card1"])
        card2 = self._string(frame["card2"])
        card3 = self._string(frame["card3"])
        card4 = self._string(frame["card4"])
        card5 = self._string(frame["card5"])
        card6 = self._string(frame["card6"])
        raw_keys = {
            "card1_freq": card1,
            "card12_freq": card1 + "|" + card2,
            "card_tuple_freq": card1 + "|" + card2 + "|" + card3 + "|" + card4 + "|" + card5 + "|" + card6,
            "pemail_freq": self._string(frame["purchaser_email_domain"]),
            "remail_freq": self._string(frame["recipient_email_domain"]),
            "addr1_freq": self._string(frame["addr1"]),
            "device_freq": self._string(frame["device_type"]),
            "product_card4_freq": self._string(frame["product_cd"]) + "|" + card4,
        }
        for name, keys in raw_keys.items():
            X[name] = keys.map(self.frequency_maps[name]).fillna(0.0).astype("float32")

        amount = frame["transaction_amt"].astype("float64")
        for col, mapping in self.amount_medians.items():
            denom = self._string(frame[col]).map(mapping).fillna(self.global_amount_median).clip(lower=0.01)
            X[f"amt_over_{col}_median"] = (amount / denom).clip(0, 100).astype("float32")

        X["missing_count"] = frame[self.base_columns].isna().sum(axis=1).astype("int16")
        hour = frame["hour_of_day"].fillna(0).astype("float64")
        dow = frame["day_of_week"].fillna(0).astype("float64")
        X["hour_sin"] = np.sin(2 * np.pi * hour / 24).astype("float32")
        X["hour_cos"] = np.cos(2 * np.pi * hour / 24).astype("float32")
        X["dow_sin"] = np.sin(2 * np.pi * dow / 7).astype("float32")
        X["dow_cos"] = np.cos(2 * np.pi * dow / 7).astype("float32")

        for col in X.columns:
            if X[col].dtype == "float64":
                X[col] = X[col].astype("float32")
            elif X[col].dtype == "int64":
                X[col] = X[col].astype("int32")
        return X


def synthesize(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    categorical: list[str],
    positive_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(SEED + int(round(positive_fraction * 10_000)))
    columns = list(X_train.columns)
    matrix = X_train.to_numpy(dtype=np.float32)
    discrete = set(c for c in categorical if c in columns)
    for col in columns:
        if X_train[col].nunique(dropna=True) <= 64:
            discrete.add(col)
    discrete_idx = np.array([i for i, c in enumerate(columns) if c in discrete], dtype=int)
    continuous_idx = np.array([i for i, c in enumerate(columns) if c not in discrete], dtype=int)
    continuous = X_train.iloc[:, continuous_idx]
    scale = np.nan_to_num(
        (continuous.quantile(0.75) - continuous.quantile(0.25)).to_numpy(dtype=np.float32) * 0.001,
        nan=0.0,
    )
    lower = continuous.min().to_numpy(dtype=np.float32)
    upper = continuous.max().to_numpy(dtype=np.float32)
    positives = int(round(N_SYNTHETIC * positive_fraction))
    counts = {0: N_SYNTHETIC - positives, 1: positives}
    result = np.empty((N_SYNTHETIC, matrix.shape[1]), dtype=np.float32)
    labels = np.empty(N_SYNTHETIC, dtype=np.int8)
    cursor = 0
    for cls in (0, 1):
        pool = np.flatnonzero(y_train == cls)
        count = counts[cls]
        parents = rng.choice(pool, count, replace=True)
        donors = rng.choice(pool, count, replace=True)
        a = matrix[parents]
        b = matrix[donors]
        block = a.copy()
        av = a[:, continuous_idx]
        noise = rng.normal(0.0, scale, size=av.shape).astype(np.float32)
        perturbed = np.minimum(np.maximum(av + noise, lower), upper)
        block[:, continuous_idx] = np.where(np.isfinite(av), perturbed, av)
        choose = rng.random((count, len(discrete_idx))) < 0.015
        choose &= np.isfinite(b[:, discrete_idx])
        block[:, discrete_idx] = np.where(choose, b[:, discrete_idx], a[:, discrete_idx])
        result[cursor:cursor + count] = block
        labels[cursor:cursor + count] = cls
        cursor += count
    order = rng.permutation(N_SYNTHETIC)
    return result[order], labels[order]


def threshold_for_recall(y: np.ndarray, scores: np.ndarray, target: float) -> float:
    positives = np.sort(scores[y == 1])
    rank = max(0, int(np.floor((1.0 - target) * len(positives))))
    return float(positives[rank])


def metric_bundle(y: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    pred = scores >= threshold
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    return {
        "roc_auc": float(roc_auc_score(y, scores)),
        "pr_auc": float(average_precision_score(y, scores)),
        "threshold": float(threshold),
        "recall": float(tp / (tp + fn)),
        "precision": float(tp / (tp + fp)),
        "false_positive_rate": float(fp / (fp + tn)),
        "specificity": float(tn / (tn + fp)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def card_overlap(train: pd.DataFrame, other: pd.DataFrame) -> dict:
    train_cards = set(train["card1"].astype("string").fillna("__NA__"))
    other_cards = other["card1"].astype("string").fillna("__NA__")
    return {
        "distinct_card1_overlap_count": int(len(train_cards & set(other_cards))),
        "row_share_with_card1_seen_in_train": float(other_cards.isin(train_cards).mean()),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.time()
    raw = pd.read_parquet(SOURCE)
    train_idx, valid_idx, test_idx = freeze_split(raw)
    train_raw = raw.iloc[train_idx].reset_index(drop=True)
    valid_raw = raw.iloc[valid_idx].reset_index(drop=True)
    # Test features and labels deliberately remain unmaterialized until configuration lock.

    protocol = {
        "evaluation_question": "repeat-card, same-population transaction generalization",
        "split": "stratified random rows, 60/20/20",
        "seed": SEED,
        "rows": {
            "real_train": int(len(train_idx)),
            "real_validation": int(len(valid_idx)),
            "locked_real_test": int(len(test_idx)),
            "synthetic_training_addition": N_SYNTHETIC,
        },
        "test_transaction_id_sha256": id_hash(raw.iloc[test_idx]["transaction_id"]),
        "synthetic_fraud_fraction_candidates": list(SYNTHETIC_FRAUD_FRACTIONS),
        "selection_metric": "real validation ROC-AUC",
        "optuna_trials": N_OPTUNA_TRIALS,
        "threshold_policy": "target 92% recall on real validation",
        "test_contract": "no test transformation, prediction, or metric until final configuration lock",
        "excluded_columns": EXCLUDED,
    }
    (OUT / "protocol_frozen_before_training.json").write_text(
        json.dumps(protocol, indent=2), encoding="utf-8"
    )

    builder = FeatureBuilder().fit(train_raw)
    X_train = builder.transform(train_raw)
    X_valid = builder.transform(valid_raw)
    y_train = train_raw["is_fraud"].to_numpy(dtype=np.int8)
    y_valid = valid_raw["is_fraud"].to_numpy(dtype=np.int8)
    features = list(X_train.columns)
    cat_idx = [features.index(c) for c in builder.categorical if c in features]
    valid_array = X_valid.to_numpy(dtype=np.float32)

    fixed_screen = {
        "objective": "binary", "metric": "auc", "verbosity": -1, "num_threads": -1,
        "learning_rate": 0.045, "num_leaves": 96, "max_depth": 11,
        "min_data_in_leaf": 180, "feature_fraction": 0.82, "bagging_fraction": 0.86,
        "bagging_freq": 1, "lambda_l1": 0.02, "lambda_l2": 1.0,
        "min_gain_to_split": 0.04, "max_bin": 255, "feature_pre_filter": False,
        "seed": SEED, "feature_fraction_seed": SEED, "bagging_seed": SEED,
        "data_random_seed": SEED,
    }

    candidates: dict[float, tuple[np.ndarray, np.ndarray]] = {}
    screening: list[dict] = []
    for fraction in SYNTHETIC_FRAUD_FRACTIONS:
        synth_X, synth_y = synthesize(X_train, y_train, builder.categorical, fraction)
        candidates[fraction] = (synth_X, synth_y)
        augmented_X = np.vstack([X_train.to_numpy(dtype=np.float32), synth_X])
        augmented_y = np.concatenate([y_train, synth_y])
        train_set = lgb.Dataset(
            augmented_X, label=augmented_y, feature_name=features,
            categorical_feature=cat_idx, free_raw_data=False,
            params={"max_bin": 255, "feature_pre_filter": False},
        )
        valid_set = lgb.Dataset(
            valid_array, label=y_valid, reference=train_set, feature_name=features,
            categorical_feature=cat_idx, free_raw_data=False,
            params={"max_bin": 255, "feature_pre_filter": False},
        )
        model = lgb.train(
            fixed_screen, train_set, num_boost_round=1_500, valid_sets=[valid_set],
            callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
        )
        scores = model.predict(valid_array, num_iteration=model.best_iteration)
        screening.append({
            "synthetic_fraud_fraction": fraction,
            "validation_auc": float(roc_auc_score(y_valid, scores)),
            "best_iteration": int(model.best_iteration),
        })
        del model, train_set, valid_set, augmented_X, augmented_y

    selected_fraction = max(screening, key=lambda row: row["validation_auc"])["synthetic_fraud_fraction"]
    synth_X, synth_y = candidates[selected_fraction]
    augmented_X = np.vstack([X_train.to_numpy(dtype=np.float32), synth_X])
    augmented_y = np.concatenate([y_train, synth_y])
    train_set = lgb.Dataset(
        augmented_X, label=augmented_y, feature_name=features,
        categorical_feature=cat_idx, free_raw_data=False,
        params={"max_bin": 255, "feature_pre_filter": False},
    )
    valid_set = lgb.Dataset(
        valid_array, label=y_valid, reference=train_set, feature_name=features,
        categorical_feature=cat_idx, free_raw_data=False,
        params={"max_bin": 255, "feature_pre_filter": False},
    )

    fixed = {
        "objective": "binary", "metric": "auc", "verbosity": -1, "num_threads": -1,
        "bagging_freq": 1, "max_bin": 255, "feature_pre_filter": False,
        "seed": SEED, "feature_fraction_seed": SEED, "bagging_seed": SEED,
        "data_random_seed": SEED,
    }

    def objective(trial: optuna.Trial) -> float:
        params = {
            **fixed,
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.08, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 48, 192, log=True),
            "max_depth": trial.suggest_int("max_depth", 7, 15),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 80, 500, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.65, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.72, 1.0),
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-4, 8.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-4, 20.0, log=True),
            "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0.0, 0.4),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 4.0, log=True),
        }
        model = lgb.train(
            params, train_set, num_boost_round=2_000, valid_sets=[valid_set],
            callbacks=[lgb.early_stopping(120, verbose=False), lgb.log_evaluation(0)],
        )
        trial.set_user_attr("best_iteration", int(model.best_iteration))
        return float(model.best_score["valid_0"]["auc"])

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED, multivariate=True),
        study_name="repeat_card_real_plus_40k_synthetic",
    )
    study.optimize(objective, n_trials=N_OPTUNA_TRIALS, show_progress_bar=False)

    final_params = {**fixed, **study.best_params}
    model = lgb.train(
        final_params, train_set, num_boost_round=2_500, valid_sets=[valid_set],
        callbacks=[lgb.early_stopping(160, verbose=False), lgb.log_evaluation(0)],
    )
    validation_scores = model.predict(valid_array, num_iteration=model.best_iteration)
    operating_threshold = threshold_for_recall(y_valid, validation_scores, 0.92)
    validation_metrics = metric_bundle(y_valid, validation_scores, operating_threshold)

    lock = {
        "selected_synthetic_fraud_fraction": selected_fraction,
        "augmentation_screening": screening,
        "best_trial": int(study.best_trial.number),
        "best_params": study.best_params,
        "best_optuna_validation_auc": float(study.best_value),
        "final_best_iteration": int(model.best_iteration),
        "operating_threshold": operating_threshold,
        "threshold_policy": "target 92% recall on real validation",
        "validation_metrics": validation_metrics,
        "test_predictions_made_before_this_file": False,
    }
    (OUT / "locked_configuration_before_test.json").write_text(
        json.dumps(lock, indent=2), encoding="utf-8"
    )

    # First and only test materialization/scoring point in this script.
    test_raw = raw.iloc[test_idx].reset_index(drop=True)
    X_test = builder.transform(test_raw).to_numpy(dtype=np.float32)
    y_test = test_raw["is_fraud"].to_numpy(dtype=np.int8)
    test_scores = model.predict(X_test, num_iteration=model.best_iteration)
    test_metrics = metric_bundle(y_test, test_scores, operating_threshold)

    result = {
        "evaluation_scope": {
            "question": "repeat-card, same-population transaction generalization",
            "card_overlap_is_intentional": True,
            "cold_start_claim": False,
            "validation_card1_overlap": card_overlap(train_raw, valid_raw),
            "test_card1_overlap": card_overlap(train_raw, test_raw),
        },
        "evaluation_integrity": {
            "test_used_for_feature_fitting": False,
            "test_used_for_synthetic_generation": False,
            "test_used_for_augmentation_selection": False,
            "test_used_for_optuna": False,
            "test_used_for_early_stopping": False,
            "test_used_for_threshold_selection": False,
            "test_scored_once_after_configuration_lock": True,
        },
        "rows": {
            "real_train": int(len(train_raw)),
            "synthetic_train_addition": int(len(synth_y)),
            "augmented_training": int(len(train_raw) + len(synth_y)),
            "real_validation": int(len(valid_raw)),
            "training_plus_validation": int(len(train_raw) + len(synth_y) + len(valid_raw)),
            "locked_real_test": int(len(test_raw)),
        },
        "optimization": lock,
        "locked_real_test": test_metrics,
        "runtime_seconds": float(time.time() - started),
    }
    model.save_model(str(OUT / "lightgbm_repeat_card.txt"))
    study.trials_dataframe().to_csv(OUT / "optuna_trials.csv", index=False)
    pd.DataFrame(screening).to_csv(OUT / "augmentation_screening.csv", index=False)
    (OUT / "category_maps.json").write_text(
        json.dumps(builder.category_maps, indent=2), encoding="utf-8"
    )
    (OUT / "final_results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
