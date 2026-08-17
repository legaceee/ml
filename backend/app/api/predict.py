"""
Prediction API Endpoints for Single Flow, CSV Batch Ingestion, and Real Test Sample Evaluation.
"""

from io import StringIO
from typing import Optional
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
import pandas as pd
from sqlalchemy.orm import Session

from backend.app.database.db import get_db
from backend.app.database.models import PredictionLog
from backend.app.schemas.schemas import (
    BatchPredictionSummary,
    PredictionRequest,
    PredictionResponse,
    TestSamplesResponse
)
from backend.app.services.ml_service import ml_service

router = APIRouter(prefix="/predict", tags=["Predictions"])


@router.post("", response_model=PredictionResponse)
def predict_single_flow(
    payload: PredictionRequest,
    db: Session = Depends(get_db)
):
    """
    Classify a single network flow telemetry record and provide SHAP feature attribution.
    """
    try:
        result = ml_service.predict_single(
            features=payload.features,
            model_name=payload.model_name or "optimized_xgboost"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    # Log prediction to database
    try:
        log_entry = PredictionLog(
            model_name=result["model_used"],
            prediction=result["prediction"],
            is_attack=result["is_attack"],
            confidence=result["confidence"],
            latency_ms=result["latency_ms"],
            attack_type=result.get("attack_type"),
            input_features=payload.features,
            top_features=[dict(f) for f in result.get("top_contributing_features", [])]
        )
        db.add(log_entry)
        db.commit()
    except Exception as db_err:
        print(f"[Prediction API] DB logging warning: {db_err}")

    return result


@router.get("/test-samples", response_model=TestSamplesResponse)
def get_test_samples_predictions(
    count: int = Query(default=10, ge=2, le=50),
    model_name: Optional[str] = Query(default="optimized_xgboost")
):
    """
    Run predictions on real, untouched test samples from the X_test split.
    Provides verified ground truth comparison distinguishing ML model health
    from interactive input parameter variations.
    """
    try:
        samples = ml_service.predict_test_samples(
            count=count,
            model_name=model_name or "optimized_xgboost"
        )
        return {
            "total_samples": len(samples),
            "model_used": model_name or "optimized_xgboost",
            "samples": samples
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate test sample predictions: {str(e)}")


@router.post("/batch", response_model=BatchPredictionSummary)
async def predict_batch_csv(
    file: UploadFile = File(...),
    model_name: Optional[str] = Form("optimized_xgboost"),
    db: Session = Depends(get_db)
):
    """
    Process an uploaded CSV file containing network flow records in batch mode.
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Invalid file format. Only CSV files are supported.")

    try:
        contents = await file.read()
        csv_str = contents.decode("utf-8", errors="ignore")
        df = pd.read_csv(StringIO(csv_str), low_memory=False)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse CSV file: {str(e)}")

    if len(df) == 0:
        raise HTTPException(status_code=400, detail="Uploaded CSV file contains no data rows.")

    if len(df) > 50000:
        raise HTTPException(status_code=400, detail="Batch size limit is 50,000 rows per request.")

    try:
        summary = ml_service.predict_batch_df(df, model_name=model_name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")

    return summary
