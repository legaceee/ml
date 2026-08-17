import sys
from pathlib import Path

# Ensure project root is present in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient
import pytest

from backend.app.main import app

client = TestClient(app)



def test_health_endpoint():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "models_loaded_count" in data


def test_models_registry_endpoint():
    response = client.get("/api/models")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) > 0
    assert "model_id" in data[0]


def test_master_metrics_endpoint():
    response = client.get("/api/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "models" in data


def test_predict_single_endpoint():
    payload = {
        "model_name": "optimized_xgboost",
        "features": {
            "Destination Port": 80.0,
            "Flow Duration": 15000.0,
            "Total Fwd Packets": 45.0,
            "Total Backward Packets": 2.0,
            "Flow Bytes/s": 185000.0,
            "SYN Flag Count": 1.0
        }
    }
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert data["prediction"] in ["ATTACK", "BENIGN"]
    assert "confidence" in data
    assert "top_contributing_features" in data


def test_research_conclusions_endpoint():
    response = client.get("/api/experiments/conclusions")
    assert response.status_code == 200
    data = response.json()
    assert "rq1_individual_baselines" in data
    assert "rq4_ensemble_superiority" in data
    assert "executive_summary" in data
