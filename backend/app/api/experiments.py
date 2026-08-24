"""
Experiment Tracking & Research Conclusions API Endpoints.
"""

import json
from pathlib import Path
from typing import Any, Dict, List
from fastapi import APIRouter, HTTPException
import pandas as pd

from backend.app.services.ml_service import ml_service

router = APIRouter(prefix="/experiments", tags=["Experiments & Conclusions"])


@router.get("")
def get_experiment_records():
    """
    Get full experiment history table (results.csv).
    """
    results_csv = ml_service.experiments_dir / "results.csv"
    if not results_csv.exists():
        return {"records": []}

    df = pd.read_csv(results_csv)
    return {"records": df.to_dict(orient="records")}


@router.get("/statistical-stability")
def get_statistical_stability():
    """
    Get multi-seed statistical evaluation stability metrics (mean ± std).
    """
    stab_file = ml_service.metrics_dir / "statistical_stability.json"
    if not stab_file.exists():
        raise HTTPException(status_code=404, detail="Statistical stability metrics not found.")

    with open(stab_file, "r") as f:
        return json.load(f)


@router.get("/conclusions")
def get_research_conclusions():
    """
    Get automatically synthesized empirical answers to Research Questions RQ1-RQ6.
    """
    conc_file = ml_service.experiments_dir / "research_conclusions.json"
    if not conc_file.exists():
        raise HTTPException(status_code=404, detail="Research conclusions not found.")

    with open(conc_file, "r") as f:
        return json.load(f)


@router.get("/optimization-study")
def get_optimization_study():
    """
    Get Optuna hyperparameter optimization trial histories and parameters.
    """
    opt_file = ml_service.metrics_dir / "optimization_study.json"
    if not opt_file.exists():
        raise HTTPException(status_code=404, detail="Optimization study results not found.")

    with open(opt_file, "r") as f:
        return json.load(f)


ARTIFACT_WHITELIST = {
    "cv-results": "cv_results.json",
    "imbalance-study": "imbalance_study.json",
    "per-class-recall": "per_class_recall.json",
    "search-comparison": "optimization_study.json",
    "shap-global": "shap_global.json",
    "pca-summary": "pca_summary.json",
}


@router.get("/artifact/{name}")
def get_named_artifact(name: str):
    """
    Serve one of the metrics JSON artifacts produced by the experiment runner:
    cv-results, imbalance-study, per-class-recall, search-comparison, shap-global, pca-summary.
    """
    fname = ARTIFACT_WHITELIST.get(name)
    if fname is None:
        raise HTTPException(status_code=404, detail=f"Unknown artifact '{name}'. Choose from {sorted(ARTIFACT_WHITELIST)}.")
    path = ml_service.metrics_dir / fname
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Artifact {fname} not generated yet. Run the experiment runner.")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if name == "search-comparison":
        return {"search_comparison": data.get("search_comparison", []), "protocol": data.get("protocol", "")}
    return data


@router.get("/feature-selection")
def get_feature_selection_benchmarks():
    """
    Get feature selection rankings and subsets (Correlation, MI, Tree, PCA).
    """
    fs_file = ml_service.metrics_dir / "feature_selection_summary.json"
    pca_file = ml_service.metrics_dir / "pca_summary.json"

    fs_data = json.load(open(fs_file)) if fs_file.exists() else {}
    pca_data = json.load(open(pca_file)) if pca_file.exists() else {}

    return {
        "feature_selection": fs_data,
        "pca_analysis": pca_data
    }
