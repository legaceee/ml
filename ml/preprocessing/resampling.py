"""
Imbalanced Dataset Handling Strategies.

CIC-IDS2017 is imbalanced at two levels: benign flows outnumber attack flows
overall, and within the attack class a handful of categories (DoS Hulk,
PortScan, DDoS) dwarf the rare ones (Heartbleed, SQL Injection, Infiltration).

This module implements the four standard remedies covered in the course and an
experiment harness that compares them on identical train/test splits.

    none                 - train on the raw (imbalanced) training split
    class_weight         - cost-sensitive learning: minority errors are penalised
                           more heavily (sklearn class_weight="balanced")
    random_undersample   - drop majority-class rows until classes are balanced
    smote                - Synthetic Minority Over-sampling TEchnique:
                           interpolate new minority samples between nearest
                           neighbours (Chawla et al., 2002)

IMPORTANT LEAKAGE RULE: resampling is applied to the TRAINING split only. The
test split is never resampled, otherwise metrics would describe an artificial
distribution instead of real traffic.
"""

import time
from typing import Any, Dict, List, Optional

import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from ml.training.evaluate import evaluate_model

STRATEGIES = ["none", "class_weight", "random_undersample", "smote"]


def resample_training_data(
    X_train: np.ndarray,
    y_train: np.ndarray,
    strategy: str,
    random_state: int = 42,
):
    """Return (X_resampled, y_resampled) for the training split only."""
    if strategy in ("none", "class_weight"):
        return X_train, y_train
    if strategy == "random_undersample":
        sampler = RandomUnderSampler(random_state=random_state)
    elif strategy == "smote":
        sampler = SMOTE(random_state=random_state, k_neighbors=5)
    else:
        raise ValueError(f"Unknown resampling strategy: {strategy}")
    return sampler.fit_resample(X_train, y_train)


def _candidate_models(random_state: int, class_weight: Optional[str]) -> Dict[str, Any]:
    return {
        "logistic_regression": LogisticRegression(max_iter=2000, random_state=random_state, class_weight=class_weight),
        "decision_tree": DecisionTreeClassifier(random_state=random_state, class_weight=class_weight),
        "random_forest": RandomForestClassifier(n_estimators=100, random_state=random_state, class_weight=class_weight, n_jobs=-1),
    }


def run_imbalance_study(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    random_state: int = 42,
    strategies: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Train each candidate model under each imbalance strategy and evaluate on the
    untouched test split. Returns a JSON-serialisable summary.
    """
    strategies = strategies or STRATEGIES
    records = []
    class_counts_before = {int(k): int(v) for k, v in zip(*np.unique(y_train, return_counts=True))}

    for strategy in strategies:
        t0 = time.perf_counter()
        X_res, y_res = resample_training_data(X_train, y_train, strategy, random_state)
        resample_time = time.perf_counter() - t0
        class_counts_after = {int(k): int(v) for k, v in zip(*np.unique(y_res, return_counts=True))}
        cw = "balanced" if strategy == "class_weight" else None

        for model_name, model in _candidate_models(random_state, cw).items():
            t0 = time.perf_counter()
            model.fit(X_res, y_res)
            train_time = time.perf_counter() - t0
            ev = evaluate_model(model, X_test, y_test, model_name=model_name, training_time=train_time)
            records.append({
                "strategy": strategy,
                "model": model_name,
                "train_rows_after_resampling": int(len(y_res)),
                "train_class_counts_after": class_counts_after,
                "resample_time_sec": round(resample_time, 3),
                "training_time_sec": round(train_time, 3),
                "accuracy": ev["accuracy"],
                "precision": ev["precision"],
                "recall": ev["recall"],
                "f1": ev["f1"],
                "roc_auc": ev["roc_auc"],
                "pr_auc": ev["pr_auc"],
                "false_positive_rate": ev["false_positive_rate"],
                "false_negative_rate": ev["false_negative_rate"],
                "confusion_matrix": ev["confusion_matrix"],
            })

    # Per-model best strategy by F1 (ties -> higher recall)
    best_by_model: Dict[str, Dict[str, Any]] = {}
    for rec in records:
        cur = best_by_model.get(rec["model"])
        if cur is None or (rec["f1"], rec["recall"]) > (cur["f1"], cur["recall"]):
            best_by_model[rec["model"]] = {"strategy": rec["strategy"], "f1": rec["f1"], "recall": rec["recall"]}

    return {
        "description": "Imbalanced-dataset handling study. Resampling applied to the training split only; "
                       "all rows evaluated on the same untouched test split.",
        "train_class_counts_before": class_counts_before,
        "train_attack_ratio_before": float(np.mean(y_train == 1)),
        "strategies": strategies,
        "records": records,
        "best_strategy_by_model": best_by_model,
    }
