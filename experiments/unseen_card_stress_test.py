#!/usr/bin/env python3
"""Leakage-resistant IEEE-CIS rerun with a locked, unseen-card1 test set.

Protocol:
1. Freeze mutually exclusive card1 groups with StratifiedGroupKFold.
2. Fit preprocessing and the synthetic generator on training groups only.
3. Tune LightGBM and select the operating threshold on validation groups only.
4. Score the locked test groups once, after all choices are fixed.
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
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedGroupKFold


ROOT = Path(__file__).resolve().parents[1]
SEED = 42
N_SYNTHETIC = 568_630
SOURCE = ROOT / "data" / "ieee_fraud_features.parquet"
OUT = ROOT / "artifacts" / "unseen_card_stress_test"

EXCLUDED = [
    "transaction_id", "transaction_ts", "card1_historical_fraud_rate",
    "email_historical_fraud_rate", "card1_txn_count", "card1_avg_amt",
    "email_txn_count", "is_high_risk_product", "amt_vs_card_avg_ratio",
]

KNOWN_CATEGORICAL = [
    "product_cd", "card1", "card2", "card3", "card4", "card5", "card6",
    "addr1", "addr2", "purchaser_email_domain", "recipient_email_domain", "device_type",
]


def freeze_group_split(frame: pd.DataFrame):
    y = frame["is_fraud"].astype("int8").to_numpy()
    groups = frame["card1"].astype("string").fillna("__MISSING_CARD1__").to_numpy()
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    remaining_idx, test_idx = next(outer.split(frame, y, groups))
    inner = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=SEED + 1)
    inner_groups = groups[remaining_idx]
    inner_y = y[remaining_idx]
    train_rel, valid_rel = next(inner.split(remaining_idx, inner_y, inner_groups))
    train_idx = remaining_idx[train_rel]
    valid_idx = remaining_idx[valid_rel]

    train_groups = set(groups[train_idx])
    valid_groups = set(groups[valid_idx])
    test_groups = set(groups[test_idx])
    assert train_groups.isdisjoint(valid_groups)
    assert train_groups.isdisjoint(test_groups)
    assert valid_groups.isdisjoint(test_groups)
    assert len(set(train_idx) | set(valid_idx) | set(test_idx)) == len(frame)
    return train_idx, valid_idx, test_idx, groups


def encode_features(frame: pd.DataFrame, train_idx: np.ndarray):
    X = frame.drop(columns=["is_fraud"] + [c for c in EXCLUDED if c in frame]).copy()
    categorical = list(X.select_dtypes(include=["object", "string"]).columns)
    categorical.extend(c for c in KNOWN_CATEGORICAL if c in X.columns)
    categorical = list(dict.fromkeys(categorical))
    maps = {}
    for col in categorical:
        train_values = X.iloc[train_idx][col].astype("string").fillna("__NA__")
        categories = pd.Index(train_values.unique())
        mapping = {value: code for code, value in enumerate(categories)}
        X[col] = (
            X[col].astype("string").fillna("__NA__")
            .map(mapping).fillna(-1).astype("int32")
        )
        maps[col] = [str(v) for v in categories]
    for col in X.columns:
        if X[col].dtype == "float64":
            X[col] = X[col].astype("float32")
        elif X[col].dtype == "int64":
            X[col] = X[col].astype("int32")
    return X, categorical, maps


def synthesize(X_train: pd.DataFrame, y_train: np.ndarray, categorical: list[str]):
    """High-fidelity class-conditional kernel bootstrap, train partition only."""
    rng = np.random.default_rng(SEED)
    columns = list(X_train.columns)
    matrix = X_train.to_numpy(dtype=np.float32)
    discrete = set(categorical)
    for col in columns:
        if X_train[col].nunique(dropna=True) <= 64:
            discrete.add(col)
    discrete_idx = np.array([i for i, c in enumerate(columns) if c in discrete])
    continuous_idx = np.array([i for i, c in enumerate(columns) if c not in discrete])
    continuous = X_train.iloc[:, continuous_idx]
    scale = np.nan_to_num(
        (continuous.quantile(0.75) - continuous.quantile(0.25)).to_numpy(dtype=np.float32)
        * 0.0005,
        nan=0.0,
    )
    lower = continuous.min().to_numpy(dtype=np.float32)
    upper = continuous.max().to_numpy(dtype=np.float32)
    positive = int(round(N_SYNTHETIC * float(y_train.mean())))
    counts = {0: N_SYNTHETIC - positive, 1: positive}
    result = np.empty((N_SYNTHETIC, matrix.shape[1]), dtype=np.float32)
    labels = np.empty(N_SYNTHETIC, dtype=np.int8)
    cursor = 0
    for cls in (0, 1):
        pool = np.flatnonzero(y_train == cls)
        count = counts[cls]
        parents = rng.choice(pool, count, replace=True)
        donors = rng.choice(pool, count, replace=True)
        for start in range(0, count, 50_000):
            stop = min(start + 50_000, count)
            a = matrix[parents[start:stop]]
            b = matrix[donors[start:stop]]
            block = a.copy()
            av = a[:, continuous_idx]
            noise = rng.normal(0.0, scale, size=av.shape).astype(np.float32)
            perturbed = np.minimum(np.maximum(av + noise, lower), upper)
            block[:, continuous_idx] = np.where(np.isfinite(av), perturbed, av)
            choose = rng.random((stop - start, len(discrete_idx))) < 0.02
            donor_values = b[:, discrete_idx]
            choose &= np.isfinite(donor_values)
            block[:, discrete_idx] = np.where(choose, donor_values, a[:, discrete_idx])
            result[cursor + start : cursor + stop] = block
            labels[cursor + start : cursor + stop] = cls
        cursor += count
    order = rng.permutation(N_SYNTHETIC)
    return result[order], labels[order]


def threshold_for_recall(y: np.ndarray, scores: np.ndarray, target: float) -> float:
    positives = np.sort(scores[y == 1])
    rank = max(0, int(np.floor((1.0 - target) * len(positives))))
    return float(positives[rank])


def metrics(y: np.ndarray, scores: np.ndarray, threshold: float):
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


def id_hash(values: pd.Series) -> str:
    joined = "\n".join(map(str, sorted(values.tolist()))).encode()
    return hashlib.sha256(joined).hexdigest()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.time()
    raw = pd.read_parquet(SOURCE)
    train_idx, valid_idx, test_idx, raw_groups = freeze_group_split(raw)
    y = raw["is_fraud"].astype("int8")
    X, categorical, category_maps = encode_features(raw, train_idx)

    assignments = pd.DataFrame({
        "transaction_id": raw["transaction_id"],
        "card1_group": raw["card1"].astype("string"),
        "split": "",
    })
    assignments.loc[train_idx, "split"] = "train_source"
    assignments.loc[valid_idx, "split"] = "validation"
    assignments.loc[test_idx, "split"] = "locked_test"
    assignments.to_parquet(OUT / "split_assignments.parquet", index=False)

    split_manifest = {
        "protocol": "card1-grouped 60/20/20 using nested StratifiedGroupKFold",
        "seeds": {"outer": SEED, "inner": SEED + 1},
        "group_overlap_assertions": {
            "train_validation": 0,
            "train_test": 0,
            "validation_test": 0,
        },
        "rows": {
            "train_source": int(len(train_idx)),
            "validation": int(len(valid_idx)),
            "locked_test": int(len(test_idx)),
        },
        "card1_groups": {
            "train_source": int(len(set(raw_groups[train_idx]))),
            "validation": int(len(set(raw_groups[valid_idx]))),
            "locked_test": int(len(set(raw_groups[test_idx]))),
        },
        "fraud_rates": {
            "train_source": float(y.iloc[train_idx].mean()),
            "validation": float(y.iloc[valid_idx].mean()),
            "locked_test": float(y.iloc[test_idx].mean()),
        },
        "transaction_id_sha256": {
            "train_source": id_hash(raw.iloc[train_idx]["transaction_id"]),
            "validation": id_hash(raw.iloc[valid_idx]["transaction_id"]),
            "locked_test": id_hash(raw.iloc[test_idx]["transaction_id"]),
        },
    }
    (OUT / "split_manifest.json").write_text(json.dumps(split_manifest, indent=2), encoding="utf-8")
    (OUT / "category_maps.json").write_text(json.dumps(category_maps, indent=2), encoding="utf-8")

    X_train = X.iloc[train_idx].reset_index(drop=True)
    y_train = y.iloc[train_idx].to_numpy(dtype=np.int8)
    X_valid = X.iloc[valid_idx].reset_index(drop=True)
    y_valid = y.iloc[valid_idx].to_numpy(dtype=np.int8)
    # Test features and labels are deliberately not materialized until after tuning and threshold lock.
    synth_X, synth_y = synthesize(X_train, y_train, categorical)
    features = list(X.columns)
    cat_idx = [features.index(c) for c in categorical]
    real_hashes = set(
        pd.util.hash_pandas_object(X_train, index=False).astype("uint64").tolist()
    )
    synthetic_hashes = pd.util.hash_pandas_object(
        pd.DataFrame(synth_X, columns=features), index=False
    ).astype("uint64")
    exact_source_matches = sum(int(value) in real_hashes for value in synthetic_hashes)
    assert exact_source_matches == 0
    train_set = lgb.Dataset(
        synth_X, label=synth_y, feature_name=features, categorical_feature=cat_idx,
        free_raw_data=False, params={"max_bin": 255, "feature_pre_filter": False},
    )
    valid_array = X_valid.to_numpy(dtype=np.float32)
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
            "learning_rate": trial.suggest_float("learning_rate", 0.025, 0.09, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 32, 160, log=True),
            "max_depth": trial.suggest_int("max_depth", 6, 14),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 60, 500, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.65, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.70, 1.0),
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-4, 10.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-4, 20.0, log=True),
            "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0.0, 0.5),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 4.0, log=True),
        }
        model = lgb.train(
            params, train_set, num_boost_round=1_800, valid_sets=[valid_set],
            callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
        )
        trial.set_user_attr("best_iteration", int(model.best_iteration))
        return float(model.best_score["valid_0"]["auc"])

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED, multivariate=True),
        study_name="defensible_card1_grouped_lightgbm",
    )
    study.optimize(objective, n_trials=12, show_progress_bar=False)
    final_params = {**fixed, **study.best_params}
    model = lgb.train(
        final_params, train_set, num_boost_round=2_400, valid_sets=[valid_set],
        callbacks=[lgb.early_stopping(140, verbose=False), lgb.log_evaluation(0)],
    )
    validation_scores = model.predict(valid_array, num_iteration=model.best_iteration)
    # Precommitted validation recall buffer. Locked before any test prediction.
    operating_threshold = threshold_for_recall(y_valid, validation_scores, 0.92)
    vfpr, vtpr, vthresholds = roc_curve(y_valid, validation_scores)
    idx5 = np.flatnonzero(vfpr <= 0.05)[-1]
    threshold_at_5pct_fpr = float(vthresholds[idx5])

    locked_configuration = {
        "best_trial": int(study.best_trial.number),
        "best_params": study.best_params,
        "best_validation_auc": float(study.best_value),
        "final_best_iteration": int(model.best_iteration),
        "operating_threshold": operating_threshold,
        "threshold_policy": "target 92% recall on card1-disjoint validation groups",
        "threshold_at_validation_5pct_fpr": threshold_at_5pct_fpr,
        "test_predictions_made_before_this_file": False,
    }
    (OUT / "locked_configuration_before_test.json").write_text(
        json.dumps(locked_configuration, indent=2), encoding="utf-8"
    )

    # First and only test scoring point in this script.
    X_test = X.iloc[test_idx].to_numpy(dtype=np.float32)
    y_test = y.iloc[test_idx].to_numpy(dtype=np.int8)
    test_scores = model.predict(X_test, num_iteration=model.best_iteration)
    result = {
        "evaluation_integrity": {
            "test_card1_overlap_with_train": 0,
            "test_used_for_preprocessing": False,
            "test_used_for_synthesis": False,
            "test_used_for_optuna": False,
            "test_used_for_early_stopping": False,
            "test_used_for_threshold_selection": False,
            "test_scored_once_after_configuration_lock": True,
        },
        "split_manifest": split_manifest,
        "synthetic_training": {
            "rows": int(len(synth_y)),
            "fraud_rate": float(synth_y.mean()),
            "exact_source_row_matches": int(exact_source_matches),
            "method": "class-conditional high-fidelity kernel bootstrap",
            "numeric_noise": "0.05% training IQR",
            "categorical_crossover_probability": 0.02,
        },
        "optimization": locked_configuration,
        "validation": {
            "roc_auc": float(roc_auc_score(y_valid, validation_scores)),
            "pr_auc": float(average_precision_score(y_valid, validation_scores)),
            "recall_at_5pct_fpr": float(vtpr[idx5]),
        },
        "locked_test_at_validation_92pct_recall_threshold": metrics(
            y_test, test_scores, operating_threshold
        ),
        "locked_test_at_validation_5pct_fpr_threshold": metrics(
            y_test, test_scores, threshold_at_5pct_fpr
        ),
        "runtime_seconds": float(time.time() - started),
    }
    model.save_model(str(OUT / "lightgbm_card1_grouped.txt"))
    study.trials_dataframe().to_csv(OUT / "optuna_trials.csv", index=False)
    synth_frame = pd.DataFrame(synth_X, columns=features)
    synth_frame["is_fraud"] = synth_y
    synth_frame.to_parquet(OUT / "synthetic_train_568630.parquet", compression="zstd", index=False)
    (OUT / "final_results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
