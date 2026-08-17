import sys
from pathlib import Path

# Ensure project root is present in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pytest
from sklearn.datasets import make_classification

from ml.models.baselines import get_baseline_models
from ml.models.ensembles import build_ensemble_models
from ml.training.evaluate import evaluate_model



@pytest.fixture
def dummy_dataset():
    X, y = make_classification(
        n_samples=200,
        n_features=10,
        n_informative=6,
        n_redundant=2,
        random_state=42
    )
    return X[:140], y[:140], X[140:], y[140:]


def test_baseline_models_execution(dummy_dataset):
    X_train, y_train, X_test, y_test = dummy_dataset
    models = get_baseline_models(random_state=42)

    for name in ["logistic_regression", "random_forest", "xgboost"]:
        model = models[name]
        model.fit(X_train, y_train)
        eval_res = evaluate_model(model, X_test, y_test, model_name=name)

        assert "f1" in eval_res
        assert "recall" in eval_res
        assert "accuracy" in eval_res
        assert 0.0 <= eval_res["f1"] <= 1.0
        assert 0.0 <= eval_res["recall"] <= 1.0


def test_ensemble_models_execution(dummy_dataset):
    X_train, y_train, X_test, y_test = dummy_dataset
    ensembles = build_ensemble_models(random_state=42)

    for name in ["voting_soft", "bagging_extra_trees", "stacking"]:
        model = ensembles[name]
        model.fit(X_train, y_train)
        eval_res = evaluate_model(model, X_test, y_test, model_name=name)

        assert eval_res["accuracy"] > 0.60
        assert eval_res["roc_auc"] > 0.60
