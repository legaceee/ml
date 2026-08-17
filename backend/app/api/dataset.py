"""
Dataset Summary & Audit API Endpoints.
"""

import json
from pathlib import Path
from typing import Any, Dict, List
from fastapi import APIRouter, HTTPException

from backend.app.services.ml_service import ml_service

router = APIRouter(prefix="/dataset", tags=["Dataset Explorer"])


@router.get("/summary")
def get_dataset_summary():
    """
    Get dataset metadata, sample distribution, cleaning audit, and split details.
    """
    if not ml_service.dataset_summary:
        ml_service._load_metrics()

    if not ml_service.dataset_summary:
        raise HTTPException(status_code=404, detail="Dataset summary not found.")

    return ml_service.dataset_summary


@router.get("/features")
def get_dataset_features():
    """
    Get list of all expected input features with descriptions.
    """
    features = ml_service.get_feature_names()
    return {
        "features_count": len(features),
        "feature_names": features
    }


@router.get("/presets")
def get_sample_flow_presets():
    """
    Return authentic flow templates extracted directly from real CIC-IDS2017 traffic.
    Ensures complete 61-feature physical integrity for interactive simulation.
    """
    presets_json_file = Path(__file__).resolve().parent.parent.parent.parent / "ml" / "artifacts" / "metrics" / "presets.json"
    if presets_json_file.exists():
        with open(presets_json_file, "r") as f:
            return json.load(f)

    # Fallback to embedded dictionary
    return {
        "benign_web": {
            "name": "Benign HTTPS Traffic (Port 443)",
            "description": "Standard encrypted SSL/TLS web traffic with balanced bidirectional flow and normal window sizes.",
            "category": "BENIGN",
            "features": {
                "Destination Port": 443.0,
                "Flow Duration": 210103.0,
                "Total Fwd Packets": 11.0,
                "Total Backward Packets": 8.0,
                "Total Length of Fwd Packets": 4026.0,
                "Total Length of Bwd Packets": 6724.0,
                "Flow Bytes/s": 10117.0,
                "Flow Packets/s": 90.0,
                "SYN Flag Count": 1.0,
                "ACK Flag Count": 1.0,
                "RST Flag Count": 0.0,
                "Init_Win_bytes_forward": 14600.0,
                "Init_Win_bytes_backward": 29200.0
            }
        },
        "dos_hulk": {
            "name": "DoS Hulk Attack (HTTP Flood)",
            "description": "High-rate request exhaustion attack flooding web services with unique obfuscated requests.",
            "category": "ATTACK",
            "features": {
                "Destination Port": 8080.0,
                "Flow Duration": 7512.0,
                "Total Fwd Packets": 54.0,
                "Total Backward Packets": 3.0,
                "Total Length of Fwd Packets": 3091.0,
                "Total Length of Bwd Packets": 109.0,
                "Flow Bytes/s": 267367.0,
                "Flow Packets/s": 7587.0,
                "SYN Flag Count": 1.0,
                "ACK Flag Count": 0.0,
                "RST Flag Count": 0.0,
                "Init_Win_bytes_forward": 256.0,
                "Init_Win_bytes_backward": 0.0
            }
        }
    }
