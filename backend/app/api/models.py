"""
Model Registry API Endpoints.
"""

from typing import Any, Dict, List
from fastapi import APIRouter, HTTPException

from backend.app.schemas.schemas import ModelInfo
from backend.app.services.ml_service import ml_service

router = APIRouter(prefix="/models", tags=["Model Registry"])


@router.get("", response_model=List[ModelInfo])
def get_all_models():
    """
    List all trained and registered models with comprehensive evaluation benchmarks.
    """
    master = ml_service.master_metrics
    if not master or "models" not in master:
        # Fallback to loaded model names
        loaded_names = ml_service.get_loaded_models_list()
        return [
            ModelInfo(
                model_id=name,
                model_name=name.replace("_", " ").title(),
                algorithm_type="Baseline" if "optimized" not in name and "voting" not in name and "stacking" not in name else "Ensemble",
                accuracy=0.0,
                precision=0.0,
                recall=0.0,
                f1_score=0.0,
                roc_auc=0.0,
                false_positive_rate=0.0,
                false_negative_rate=0.0,
                training_time_sec=0.0,
                inference_time_ms=0.0,
                is_synthetic=True
            )
            for name in loaded_names
        ]

    models_data = master["models"]
    is_synth = master.get("is_synthetic", True)
    results = []

    for m_key, m_eval in models_data.items():
        alg_type = "Baseline"
        if "optimized" in m_key:
            alg_type = "Optimized (Optuna)"
        elif any(k in m_key for k in ["voting", "bagging", "adaboost", "gradient_boosting", "stacking", "weighted"]):
            alg_type = "Ensemble"

        results.append(ModelInfo(
            model_id=m_key,
            model_name=m_key.replace("_", " ").title(),
            algorithm_type=alg_type,
            accuracy=m_eval["accuracy"],
            precision=m_eval["precision"],
            recall=m_eval["recall"],
            f1_score=m_eval["f1"],
            roc_auc=m_eval["roc_auc"],
            false_positive_rate=m_eval["false_positive_rate"],
            false_negative_rate=m_eval["false_negative_rate"],
            training_time_sec=m_eval["training_time_sec"],
            inference_time_ms=m_eval["inference_time_ms_per_1k"],
            is_synthetic=is_synth
        ))

    return results


@router.get("/{model_id}", response_model=Dict[str, Any])
def get_model_details(model_id: str):
    """
    Get detailed metrics, configuration, and confusion matrix for a specific model.
    """
    master = ml_service.master_metrics
    if not master or "models" not in master or model_id not in master["models"]:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found in registry.")

    return {
        "model_id": model_id,
        "metrics": master["models"][model_id],
        "is_synthetic": master.get("is_synthetic", True),
        "dataset_name": master.get("dataset_name", "CIC-IDS2017")
    }
