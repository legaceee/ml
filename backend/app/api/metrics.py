"""
Master Metrics & Curve Visualization API Endpoints.
"""

from typing import Any, Dict
from fastapi import APIRouter, HTTPException

from backend.app.services.ml_service import ml_service

router = APIRouter(prefix="/metrics", tags=["Evaluation Metrics"])


@router.get("", response_model=Dict[str, Any])
def get_master_metrics():
    """
    Get full comparison dictionary containing metrics, ROC curves, PR curves,
    and confusion matrices for all evaluated models.
    """
    master = ml_service.master_metrics
    if not master:
        # Attempt to reload if freshly generated
        ml_service._load_metrics()
        master = ml_service.master_metrics

    if not master:
        raise HTTPException(status_code=404, detail="Master benchmark metrics not yet generated.")

    return master


@router.get("/{model_id}", response_model=Dict[str, Any])
def get_single_model_metrics(model_id: str):
    """
    Get evaluation metrics, confusion matrix, and curves for a specified model.
    """
    master = ml_service.master_metrics
    if not master or "models" not in master or model_id not in master["models"]:
        raise HTTPException(status_code=404, detail=f"Metrics for model '{model_id}' not found.")

    return master["models"][model_id]
