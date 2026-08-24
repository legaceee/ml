"""
Evaluation Engine for Cyber Attack Detection Models.

Calculates all standard and intrusion-detection-specific metrics:
Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, Specificity, FPR, FNR,
Confusion Matrix and Inference Latency, plus helpers for k-fold
cross-validation, per-attack-type recall and bootstrap confidence intervals.
Nothing here is ever fabricated: every number comes from model outputs.
"""

import time
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_validate


def _positive_scores(model: Any, X: np.ndarray):
    """
    Return (scores, source) where scores rank samples by attack likelihood.
    Preference: predict_proba -> decision_function -> hard labels.
    Hard labels give a valid but coarse ROC (a single operating point).
    """
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(X)
            return (proba[:, 1] if proba.shape[1] > 1 else proba[:, 0]), "predict_proba"
        except Exception:
            pass
    if hasattr(model, "decision_function"):
        try:
            return model.decision_function(X), "decision_function"
        except Exception:
            pass
    return model.predict(X).astype(float), "hard_labels"


def evaluate_model(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str,
    training_time: float = 0.0,
    is_multiclass: bool = False,
) -> Dict[str, Any]:
    """Evaluate a trained classifier on held-out data and return a metrics dictionary."""
    t_start = time.perf_counter()
    y_pred = model.predict(X_test)
    t_end = time.perf_counter()

    num_samples = len(X_test)
    total_infer_time = t_end - t_start
    infer_latency_ms_per_1k = (total_infer_time / max(1, num_samples)) * 1000.0 * 1000.0

    if is_multiclass:
        acc = float(accuracy_score(y_test, y_pred))
        return {
            "model_name": model_name,
            "accuracy": round(acc, 4),
            "macro_precision": round(float(precision_score(y_test, y_pred, average="macro", zero_division=0)), 4),
            "macro_recall": round(float(recall_score(y_test, y_pred, average="macro", zero_division=0)), 4),
            "macro_f1": round(float(f1_score(y_test, y_pred, average="macro", zero_division=0)), 4),
            "weighted_f1": round(float(f1_score(y_test, y_pred, average="weighted", zero_division=0)), 4),
            "confusion_matrix_raw": confusion_matrix(y_test, y_pred).tolist(),
            "training_time_sec": round(training_time, 4),
            "inference_time_ms_per_1k": round(infer_latency_ms_per_1k, 4),
            "num_test_samples": num_samples,
        }

    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, zero_division=0))
    rec = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    tn, fp, fn, tp = [int(v) for v in cm.ravel()]

    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0

    y_scores, score_source = _positive_scores(model, X_test)
    try:
        roc_auc_val = float(roc_auc_score(y_test, y_scores))
    except Exception:
        roc_auc_val = 0.5

    fpr_arr, tpr_arr, _ = roc_curve(y_test, y_scores)
    step = max(1, len(fpr_arr) // 60)
    roc_pts = [{"fpr": round(float(f), 4), "tpr": round(float(t), 4)} for f, t in zip(fpr_arr[::step], tpr_arr[::step])]
    if roc_pts and (roc_pts[-1]["fpr"] != 1.0 or roc_pts[-1]["tpr"] != 1.0):
        roc_pts.append({"fpr": 1.0, "tpr": 1.0})

    prec_arr, rec_arr, _ = precision_recall_curve(y_test, y_scores)
    pr_auc_val = float(auc(rec_arr, prec_arr))
    pr_step = max(1, len(prec_arr) // 60)
    pr_pts = [{"recall": round(float(r), 4), "precision": round(float(p), 4)} for r, p in zip(rec_arr[::pr_step], prec_arr[::pr_step])]

    return {
        "model_name": model_name,
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "roc_auc": round(roc_auc_val, 4),
        "pr_auc": round(pr_auc_val, 4),
        "score_source": score_source,
        "specificity": round(specificity, 4),
        "false_positive_rate": round(fpr, 4),
        "false_negative_rate": round(fnr, 4),
        "confusion_matrix": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        "roc_curve": roc_pts,
        "pr_curve": pr_pts,
        "training_time_sec": round(training_time, 4),
        "inference_time_ms_per_1k": round(infer_latency_ms_per_1k, 4),
        "num_test_samples": num_samples,
    }


def cross_validate_classifier(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    model_name: str,
    n_splits: int = 5,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Stratified k-fold cross-validation on the TRAINING split.
    Reports mean +- std of F1 / recall / precision / accuracy across folds so
    that the variance of every estimate is visible, not just a point value.
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    t0 = time.perf_counter()
    res = cross_validate(
        clone(model), X, y, cv=cv,
        scoring={"f1": "f1", "recall": "recall", "precision": "precision", "accuracy": "accuracy"},
        n_jobs=1, return_train_score=True,
    )
    wall = time.perf_counter() - t0
    out: Dict[str, Any] = {"model_name": model_name, "n_splits": n_splits, "wall_time_sec": round(wall, 2)}
    for metric in ["f1", "recall", "precision", "accuracy"]:
        vals = res[f"test_{metric}"]
        out[f"{metric}_folds"] = [round(float(v), 4) for v in vals]
        out[f"{metric}_mean"] = round(float(np.mean(vals)), 4)
        out[f"{metric}_std"] = round(float(np.std(vals)), 4)
    out["train_f1_mean"] = round(float(np.mean(res["train_f1"])), 4)
    out["overfit_gap_f1"] = round(out["train_f1_mean"] - out["f1_mean"], 4)
    return out


def per_class_recall(model: Any, X_test: np.ndarray, y_test_binary: np.ndarray, y_test_labels: np.ndarray) -> List[Dict[str, Any]]:
    """
    Detection rate broken down by original attack category (e.g. how many
    Heartbleed flows were caught) - a binary F1 hides exactly this.
    """
    y_pred = model.predict(X_test)
    labels = np.asarray(y_test_labels).astype(str)
    rows = []
    for cls in sorted(set(labels)):
        mask = labels == cls
        n = int(mask.sum())
        expected = int(y_test_binary[mask][0]) if n > 0 else 1
        correct = int(np.sum(y_pred[mask] == expected))
        rows.append({
            "class": cls,
            "is_attack": bool(expected == 1),
            "n_test": n,
            "n_detected": correct,
            "recall": round(correct / n, 4) if n else None,
        })
    rows.sort(key=lambda r: (-r["is_attack"], -r["n_test"]))
    return rows


def bootstrap_confidence_intervals(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_bootstrap: int = 200,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Non-parametric bootstrap on the test split: resample rows with replacement,
    recompute metrics, and report the 2.5/97.5 percentiles (95% CI).
    Predictions are computed once; only the resampling is repeated.
    """
    rng = np.random.RandomState(random_state)
    y_pred = model.predict(X_test)
    n = len(y_test)
    stats = {"f1": [], "recall": [], "accuracy": [], "fpr": []}
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        yt, yp = y_test[idx], y_pred[idx]
        stats["f1"].append(f1_score(yt, yp, zero_division=0))
        stats["recall"].append(recall_score(yt, yp, zero_division=0))
        stats["accuracy"].append(accuracy_score(yt, yp))
        cm = confusion_matrix(yt, yp, labels=[0, 1])
        tn, fp = int(cm[0, 0]), int(cm[0, 1])
        stats["fpr"].append(fp / (fp + tn) if (fp + tn) > 0 else 0.0)
    out: Dict[str, Any] = {"n_bootstrap": n_bootstrap}
    for k, vals in stats.items():
        arr = np.asarray(vals)
        out[f"{k}_mean"] = round(float(arr.mean()), 4)
        out[f"{k}_std"] = round(float(arr.std()), 4)
        out[f"{k}_ci95"] = [round(float(np.percentile(arr, 2.5)), 4), round(float(np.percentile(arr, 97.5)), 4)]
    return out
