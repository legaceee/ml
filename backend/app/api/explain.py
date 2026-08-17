"""
Explainable AI (SHAP) API Endpoints.
"""

import json
from pathlib import Path
from typing import Any, Dict, List
from fastapi import APIRouter, HTTPException

from backend.app.schemas.schemas import LocalExplainRequest, LocalExplainResponse
from backend.app.services.ml_service import ml_service

router = APIRouter(prefix="/explain", tags=["Explainable AI"])


@router.get("/global/{model_id}")
def get_global_shap_importances(model_id: str):
    """
    Get global SHAP feature importance rankings for a given model.
    """
    shap_file = ml_service.metrics_dir / "shap_global.json"
    if shap_file.exists():
        with open(shap_file, "r") as f:
            data = json.load(f)
            return data

    # Fallback to computing on the fly
    explainer = ml_service.get_explainer(model_id)
    if not explainer:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found for SHAP calculation.")

    return {
        "model_name": model_id,
        "global_importance": explainer.explain_global(np.zeros((10, len(explainer.feature_names))), max_display=25)
    }


@router.post("/local", response_model=LocalExplainResponse)
def explain_local_instance(payload: LocalExplainRequest):
    """
    Compute real-time SHAP waterfall/force feature attributions for a single network flow instance.
    """
    model_name = payload.model_name or "optimized_xgboost"
    explainer = ml_service.get_explainer(model_name)
    if not explainer:
        raise HTTPException(status_code=404, detail=f"Explainer could not be initialized for model '{model_name}'.")

    feature_names = ml_service.get_feature_names()
    prep = ml_service.preprocessor_data["preprocessor"]

    # Build row dataframe and scale
    row_dict = {f: float(payload.features.get(f, 0.0)) for f in feature_names}
    df_row = pd.DataFrame([row_dict])
    X_scaled = prep.transform(df_row)

    explanation = explainer.explain_instance(
        instance_1d=X_scaled[0],
        raw_feature_values=row_dict,
        top_k=8
    )

    return {
        "model_name": model_name,
        "base_value": explanation["base_value"],
        "top_attack_drivers": explanation["top_attack_drivers"],
        "top_benign_drivers": explanation["top_benign_drivers"]
    }
