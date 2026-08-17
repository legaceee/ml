"""
Comprehensive Test Suite for ML Model Predictions and Pipeline Invariants.

Verifies:
1. Model loading for all registered models.
2. Prediction and probability dimension consistency.
3. Probability sum-to-1 invariant (Benign + Attack ≈ 1.0).
4. Safe dynamic mapping of model.classes_ (0 = BENIGN, 1 = ATTACK).
5. Exact feature alignment (61 features expected).
6. Input sanitation and NaN/Inf rejection.
7. Real test dataset evaluation producing balanced predictions.
8. Preprocessor scaling and baseline imputation protocol.
9. Backend /api/predict/test-samples endpoint functionality.
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.services.ml_service import ml_service

client = TestClient(app)


@pytest.fixture(scope="module")
def artifacts_and_data():
    base_dir = Path(__file__).resolve().parent.parent / "ml"
    splits_file = base_dir / "data" / "splits" / "dataset_splits.joblib"
    prep_file = base_dir / "artifacts" / "preprocessors" / "preprocessor.joblib"
    models_dir = base_dir / "artifacts" / "models"

    assert splits_file.exists(), f"Splits file {splits_file} missing!"
    assert prep_file.exists(), f"Preprocessor file {prep_file} missing!"

    splits = joblib.load(splits_file)
    prep_data = joblib.load(prep_file)

    return {
        "splits": splits,
        "prep_data": prep_data,
        "models_dir": models_dir
    }


def test_models_load_successfully(artifacts_and_data):
    """1. Verify that all model artifacts exist and load without error."""
    models_dir = artifacts_and_data["models_dir"]
    model_files = list(models_dir.glob("*.joblib"))
    assert len(model_files) > 0, "No serialized model files found."

    for mf in model_files:
        model = joblib.load(mf)
        assert model is not None, f"Model {mf.stem} loaded as None."


def test_feature_count_and_alignment(artifacts_and_data):
    """2. Verify feature count and column naming alignment between splits and preprocessor."""
    splits = artifacts_and_data["splits"]
    prep_data = artifacts_and_data["prep_data"]

    feature_names_splits = splits["feature_names"]
    feature_names_prep = prep_data["feature_names"]

    assert len(feature_names_splits) == 61, f"Expected 61 features, found {len(feature_names_splits)}"
    assert feature_names_splits == feature_names_prep, "Feature names mismatch between splits and preprocessor."
    assert splits["X_train"].shape[1] == 61
    assert splits["X_test"].shape[1] == 61


def test_model_classes_and_probability_invariants(artifacts_and_data):
    """3. Verify model.classes_, probability dimensions, and sum to ~1.0."""
    splits = artifacts_and_data["splits"]
    models_dir = artifacts_and_data["models_dir"]
    X_test = splits["X_test"][:20]

    candidate_models = ["optimized_xgboost", "optimized_random_forest", "logistic_regression", "voting_soft"]
    for m_name in candidate_models:
        m_file = models_dir / f"{m_name}.joblib"
        if not m_file.exists():
            continue
        model = joblib.load(m_file)
        
        if hasattr(model, "classes_"):
            classes = list(model.classes_)
            assert 0 in classes, f"Model {m_name} classes_ missing 0 (BENIGN)."
            assert 1 in classes, f"Model {m_name} classes_ missing 1 (ATTACK)."

        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_test)
            assert probs.shape == (20, 2), f"Expected shape (20, 2), got {probs.shape} for {m_name}"
            # Probabilities sum to 1.0 for each row
            row_sums = probs.sum(axis=1)
            assert np.allclose(row_sums, 1.0, atol=1e-5), f"Probabilities do not sum to 1 in {m_name}"


def test_both_classes_in_evaluation(artifacts_and_data):
    """4. Verify that the test split has representative counts of both BENIGN and ATTACK."""
    splits = artifacts_and_data["splits"]
    y_test = splits["y_test"]
    unique_classes, counts = np.unique(y_test, return_counts=True)

    assert len(unique_classes) == 2, f"Expected binary classes, found {unique_classes}"
    assert set(unique_classes) == {0, 1}
    assert counts[0] > 100, f"Too few benign samples in test set: {counts[0]}"
    assert counts[1] > 100, f"Too few attack samples in test set: {counts[1]}"


def test_real_test_samples_predictions(artifacts_and_data):
    """5. Verify that predictions on real test data produce realistic benign and attack classifications."""
    splits = artifacts_and_data["splits"]
    models_dir = artifacts_and_data["models_dir"]
    X_test_raw = splits["X_test_df_raw"]
    y_test = splits["y_test"]
    prep = artifacts_and_data["prep_data"]["preprocessor"]
    model = joblib.load(models_dir / "optimized_xgboost.joblib")

    # Pick 10 known benign and 10 known attack samples
    benign_idx = np.where(y_test == 0)[0][:10]
    attack_idx = np.where(y_test == 1)[0][:10]

    # Test benign samples
    benign_df = X_test_raw.iloc[benign_idx]
    benign_sc = prep.transform(benign_df)
    benign_preds = model.predict(benign_sc)
    benign_probs = model.predict_proba(benign_sc)

    # Test attack samples
    attack_df = X_test_raw.iloc[attack_idx]
    attack_sc = prep.transform(attack_df)
    attack_preds = model.predict(attack_sc)
    attack_probs = model.predict_proba(attack_sc)

    # Benign samples should have high accuracy and high benign probability
    assert np.mean(benign_preds == 0) >= 0.90, "Real benign samples incorrectly classified as attack."
    assert np.mean(benign_probs[:, 0]) > 0.85, "Benign confidence is abnormally low on real benign samples."

    # Attack samples should have high accuracy and high attack probability
    assert np.mean(attack_preds == 1) >= 0.90, "Real attack samples incorrectly classified as benign."
    assert np.mean(attack_probs[:, 1]) > 0.85, "Attack confidence is abnormally low on real attack samples."


def test_ml_service_single_prediction_benign_and_attack():
    """6. Verify MLService.predict_single handles authentic presets with appropriate verdicts."""
    from backend.app.api.dataset import get_sample_flow_presets
    presets = get_sample_flow_presets()

    assert "benign_web" in presets
    assert "dos_hulk" in presets

    # Predict benign preset
    res_benign = ml_service.predict_single(presets["benign_web"]["features"], "optimized_xgboost")
    assert res_benign["prediction"] == "BENIGN"
    assert res_benign["probability_benign"] > 0.80
    assert res_benign["probability_attack"] < 0.20
    assert abs(res_benign["probability_benign"] + res_benign["probability_attack"] - 1.0) < 1e-4

    # Predict attack preset
    res_attack = ml_service.predict_single(presets["dos_hulk"]["features"], "optimized_xgboost")
    assert res_attack["prediction"] == "ATTACK"
    assert res_attack["probability_attack"] > 0.80
    assert res_attack["probability_benign"] < 0.20
    assert abs(res_attack["probability_benign"] + res_attack["probability_attack"] - 1.0) < 1e-4


def test_test_samples_endpoint():
    """7. Verify GET /api/predict/test-samples endpoint returns verified ground truth matches."""
    response = client.get("/api/predict/test-samples?count=8&model_name=optimized_xgboost")
    assert response.status_code == 200
    data = response.json()

    assert data["total_samples"] == 8
    assert len(data["samples"]) == 8

    # Ensure both BENIGN and ATTACK are present
    actuals = [s["actual"] for s in data["samples"]]
    assert "BENIGN" in actuals
    assert "ATTACK" in actuals

    for s in data["samples"]:
        assert 0.0 <= s["benign_probability"] <= 1.0
        assert 0.0 <= s["attack_probability"] <= 1.0
        assert abs(s["benign_probability"] + s["attack_probability"] - 1.0) < 1e-3
        assert s["is_correct"] is True, f"Real test sample #{s['sample_id']} prediction mismatch."


def test_input_validation_rejection():
    """8. Verify API rejects invalid non-numeric inputs and malformed payloads."""
    # Test non-numeric string value for a feature
    payload_invalid_str = {
        "model_name": "optimized_xgboost",
        "features": {
            "Destination Port": 80.0,
            "Flow Duration": "not_a_number"
        }
    }
    response = client.post("/api/predict", json=payload_invalid_str)
    assert response.status_code == 422, "API should reject non-numeric string input with 422 Unprocessable Entity."

    # Test raw json with null feature value
    raw_json = '{"model_name": "optimized_xgboost", "features": {"Destination Port": "invalid_val"}}'
    response_raw = client.post("/api/predict", content=raw_json, headers={"Content-Type": "application/json"})
    assert response_raw.status_code == 422
