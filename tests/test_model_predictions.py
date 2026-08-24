"""
Integration tests for the trained artifacts and the prediction service.

These tests need the artifacts produced by
    python -m ml.preprocessing.pipeline
    python -m ml.training.run_all_experiments
and are skipped automatically when they are absent.

Verifies:
1. Every serialized model loads.
2. Feature alignment between splits and preprocessor.
3. Probability invariants (shape, sum-to-1, classes_ contains 0 and 1).
4. Both classes present in the test split.
5. Real test-split rows are classified correctly by the tuned XGBoost model.
6. MLService.predict_single on real-flow presets.
7. /api/predict/test-samples endpoint.
8. Input validation rejects malformed payloads.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd
import pytest

ML_DIR = PROJECT_ROOT / "ml"
SPLITS_FILE = ML_DIR / "data" / "splits" / "dataset_splits.joblib"
PREP_FILE = ML_DIR / "artifacts" / "preprocessors" / "preprocessor.joblib"
MODELS_DIR = ML_DIR / "artifacts" / "models"

pytestmark = pytest.mark.skipif(
    not (SPLITS_FILE.exists() and PREP_FILE.exists() and (MODELS_DIR / "optimized_xgboost.joblib").exists()),
    reason="Trained artifacts not present - run the pipeline and experiment runner first.",
)


@pytest.fixture(scope="module")
def artifacts_and_data():
    return {
        "splits": joblib.load(SPLITS_FILE),
        "prep_data": joblib.load(PREP_FILE),
        "models_dir": MODELS_DIR,
    }


@pytest.fixture(scope="module")
def api_client():
    from fastapi.testclient import TestClient
    from backend.app.main import app
    return TestClient(app)


def test_models_load_successfully(artifacts_and_data):
    model_files = list(artifacts_and_data["models_dir"].glob("*.joblib"))
    assert len(model_files) > 0, "No serialized model files found."
    for mf in model_files:
        assert joblib.load(mf) is not None, f"Model {mf.stem} loaded as None."


def test_feature_count_and_alignment(artifacts_and_data):
    splits = artifacts_and_data["splits"]
    prep_data = artifacts_and_data["prep_data"]
    n = len(prep_data["feature_names"])
    assert n > 0
    assert splits["feature_names"] == prep_data["feature_names"]
    assert splits["X_train"].shape[1] == n
    assert splits["X_test"].shape[1] == n
    assert list(splits["X_test_df_raw"].columns) == prep_data["feature_names"]


def test_model_classes_and_probability_invariants(artifacts_and_data):
    splits = artifacts_and_data["splits"]
    X_test = splits["X_test"][:20]
    for m_name in ["optimized_xgboost", "optimized_random_forest", "logistic_regression", "voting_soft", "stacking"]:
        m_file = artifacts_and_data["models_dir"] / f"{m_name}.joblib"
        if not m_file.exists():
            continue
        model = joblib.load(m_file)
        if hasattr(model, "classes_"):
            assert set(np.asarray(model.classes_).tolist()) == {0, 1}, f"{m_name} classes_ != {{0,1}}"
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_test)
            assert probs.shape == (20, 2), f"Expected (20, 2), got {probs.shape} for {m_name}"
            assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5), f"Probabilities do not sum to 1 in {m_name}"


def test_both_classes_in_evaluation(artifacts_and_data):
    y_test = artifacts_and_data["splits"]["y_test"]
    unique_classes, counts = np.unique(y_test, return_counts=True)
    assert set(unique_classes.tolist()) == {0, 1}
    assert counts.min() > 100


def test_real_test_samples_predictions(artifacts_and_data):
    splits = artifacts_and_data["splits"]
    X_test_raw = splits["X_test_df_raw"]
    y_test = splits["y_test"]
    prep = artifacts_and_data["prep_data"]["preprocessor"]
    model = joblib.load(artifacts_and_data["models_dir"] / "optimized_xgboost.joblib")

    benign_idx = np.where(y_test == 0)[0][:200]
    attack_idx = np.where(y_test == 1)[0][:200]

    benign_probs = model.predict_proba(prep.transform(X_test_raw.iloc[benign_idx]))
    attack_probs = model.predict_proba(prep.transform(X_test_raw.iloc[attack_idx]))

    assert np.mean(benign_probs[:, 0] >= 0.5) >= 0.95, "Too many real benign rows flagged as attack."
    assert np.mean(attack_probs[:, 1] >= 0.5) >= 0.95, "Too many real attack rows missed."


def test_ml_service_single_prediction_benign_and_attack():
    from backend.app.api.dataset import get_sample_flow_presets
    from backend.app.services.ml_service import ml_service

    presets = get_sample_flow_presets()
    assert "benign_web" in presets
    assert "dos_hulk" in presets

    res_benign = ml_service.predict_single(presets["benign_web"]["features"], "optimized_xgboost")
    assert res_benign["prediction"] == "BENIGN"
    assert abs(res_benign["probability_benign"] + res_benign["probability_attack"] - 1.0) < 1e-4

    res_attack = ml_service.predict_single(presets["dos_hulk"]["features"], "optimized_xgboost")
    assert res_attack["prediction"] == "ATTACK"
    assert abs(res_attack["probability_benign"] + res_attack["probability_attack"] - 1.0) < 1e-4


def test_test_samples_endpoint(api_client):
    response = api_client.get("/api/predict/test-samples?count=8&model_name=optimized_xgboost")
    assert response.status_code == 200
    data = response.json()
    assert data["total_samples"] == 8
    actuals = [s["actual"] for s in data["samples"]]
    assert "BENIGN" in actuals and "ATTACK" in actuals
    for s in data["samples"]:
        assert 0.0 <= s["benign_probability"] <= 1.0
        assert 0.0 <= s["attack_probability"] <= 1.0
        assert abs(s["benign_probability"] + s["attack_probability"] - 1.0) < 1e-3
    # A strong model should get the large majority of real rows right
    assert sum(s["is_correct"] for s in data["samples"]) >= 7


def test_input_validation_rejection(api_client):
    payload_invalid_str = {"model_name": "optimized_xgboost", "features": {"Destination Port": 80.0, "Flow Duration": "not_a_number"}}
    assert api_client.post("/api/predict", json=payload_invalid_str).status_code == 422
    raw_json = '{"model_name": "optimized_xgboost", "features": {"Destination Port": "invalid_val"}}'
    assert api_client.post("/api/predict", content=raw_json, headers={"Content-Type": "application/json"}).status_code == 422
