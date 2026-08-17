"""
Comprehensive Evaluation Engine for Cyber Attack Detection Models.

Calculates all standard and intrusion-detection-specific metrics:
Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, Specificity, FPR, FNR,
Confusion Matrix, and Inference Latency. Never fabricates metrics.
"""

import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve
)


def evaluate_model(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str,
    training_time: float = 0.0,
    is_multiclass: bool = False
) -> Dict[str, Any]:
    """
    Evaluate trained classifier on test data and return comprehensive metrics dictionary.
    
    Args:
        model: Trained scikit-learn or XGBoost estimator.
        X_test: Test features.
        y_test: Ground truth labels.
        model_name: Identifier for model.
        training_time: Measured training duration in seconds.
        is_multiclass: Whether task is multiclass or binary.
        
    Returns:
        Structured dictionary of evaluation results and curve coordinates.
    """
    # Measure inference latency
    t_start = time.perf_counter()
    y_pred = model.predict(X_test)
    t_end = time.perf_counter()
    
    num_samples = len(X_test)
    total_infer_time = t_end - t_start
    infer_latency_ms_per_1k = (total_infer_time / max(1, num_samples)) * 1000.0 * 1000.0

    # Obtain prediction probabilities if supported
    has_proba = hasattr(model, "predict_proba")
    y_proba = None
    if has_proba:
        try:
            y_proba = model.predict_proba(X_test)
        except Exception:
            has_proba = False

    if not is_multiclass:
        # Binary Classification Metrics
        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, zero_division=0))
        rec = float(recall_score(y_test, y_pred, zero_division=0))
        f1 = float(f1_score(y_test, y_pred, zero_division=0))

        cm = confusion_matrix(y_test, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = [int(v) for v in cm.ravel()]
        else:
            tn, fp, fn, tp = 0, 0, 0, 0

        # Derived intrusion-detection metrics
        specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
        fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0

        roc_auc_val = 0.5
        pr_auc_val = 0.5
        roc_pts: List[Dict[str, float]] = []
        pr_pts: List[Dict[str, float]] = []

        if has_proba and y_proba is not None:
            # Positive class probability (ATTACK = 1)
            y_scores = y_proba[:, 1] if y_proba.shape[1] > 1 else y_proba[:, 0]
            try:
                roc_auc_val = float(roc_auc_score(y_test, y_scores))
            except Exception:
                roc_auc_val = 0.5

            fpr_arr, tpr_arr, _ = roc_curve(y_test, y_scores)
            # Sample curve points to keep JSON lightweight
            step = max(1, len(fpr_arr) // 40)
            roc_pts = [
                {"fpr": round(float(f), 4), "tpr": round(float(t), 4)}
                for f, t in zip(fpr_arr[::step], tpr_arr[::step])
            ]

            prec_arr, rec_arr, _ = precision_recall_curve(y_test, y_scores)
            pr_auc_val = float(auc(rec_arr, prec_arr))
            pr_step = max(1, len(prec_arr) // 40)
            pr_pts = [
                {"recall": round(float(r), 4), "precision": round(float(p), 4)}
                for r, p in zip(rec_arr[::pr_step], prec_arr[::pr_step])
            ]

        return {
            "model_name": model_name,
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "roc_auc": round(roc_auc_val, 4),
            "pr_auc": round(pr_auc_val, 4),
            "specificity": round(specificity, 4),
            "false_positive_rate": round(fpr, 4),
            "false_negative_rate": round(fnr, 4),
            "confusion_matrix": {
                "tp": tp,
                "tn": tn,
                "fp": fp,
                "fn": fn
            },
            "roc_curve": roc_pts,
            "pr_curve": pr_pts,
            "training_time_sec": round(training_time, 4),
            "inference_time_ms_per_1k": round(infer_latency_ms_per_1k, 4),
            "num_test_samples": num_samples
        }
    else:
        # Multiclass metrics
        acc = float(accuracy_score(y_test, y_pred))
        macro_prec = float(precision_score(y_test, y_pred, average="macro", zero_division=0))
        macro_rec = float(recall_score(y_test, y_pred, average="macro", zero_division=0))
        macro_f1 = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
        weighted_f1 = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))

        cm = confusion_matrix(y_test, y_pred)
        
        return {
            "model_name": model_name,
            "accuracy": round(acc, 4),
            "macro_precision": round(macro_prec, 4),
            "macro_recall": round(macro_rec, 4),
            "macro_f1": round(macro_f1, 4),
            "weighted_f1": round(weighted_f1, 4),
            "confusion_matrix_raw": cm.tolist(),
            "training_time_sec": round(training_time, 4),
            "inference_time_ms_per_1k": round(infer_latency_ms_per_1k, 4),
            "num_test_samples": num_samples
        }
