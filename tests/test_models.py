import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pytest
from sklearn.datasets import make_classification

from ml.models.baselines import get_baseline_models
from ml.models.ensembles import build_ensemble_models
from ml.models.optimization import HyperparameterOptimizer
from ml.preprocessing.resampling import resample_training_data, run_imbalance_study
from ml.training.evaluate import bootstrap_confidence_intervals, cross_validate_classifier, evaluate_model, per_class_recall


@pytest.fixture(scope="module")
def dummy_dataset():
    X, y = make_classification(n_samples=600, n_features=12, n_informative=6, n_redundant=2, weights=[0.8, 0.2], random_state=42)
    return X[:450], y[:450], X[450:], y[450:]


def test_baseline_models_execution(dummy_dataset):
    X_train, y_train, X_test, y_test = dummy_dataset
    models = get_baseline_models(random_state=42)
    assert set(models) == {"logistic_regression", "decision_tree", "random_forest", "extra_trees",
                           "support_vector_machine", "k_nearest_neighbors", "xgboost"}
    for name in ["logistic_regression", "random_forest", "xgboost", "k_nearest_neighbors"]:
        model = models[name]
        model.fit(X_train, y_train)
        ev = evaluate_model(model, X_test, y_test, model_name=name)
        for k in ("f1", "recall", "precision", "accuracy", "roc_auc", "pr_auc", "false_positive_rate"):
            assert 0.0 <= ev[k] <= 1.0
        assert ev["confusion_matrix"]["tp"] + ev["confusion_matrix"]["fn"] == int((y_test == 1).sum())


def test_ensemble_models_execution(dummy_dataset):
    X_train, y_train, X_test, y_test = dummy_dataset
    ensembles = build_ensemble_models(random_state=42)
    assert set(ensembles) == {"voting_hard", "voting_soft", "bagging_extra_trees", "adaboost", "gradient_boosting", "stacking", "weighted_ensemble"}
    for name in ["voting_hard", "voting_soft", "bagging_extra_trees", "stacking"]:
        model = ensembles[name]
        model.fit(X_train, y_train)
        ev = evaluate_model(model, X_test, y_test, model_name=name)
        assert ev["accuracy"] > 0.60
        assert ev["roc_auc"] > 0.60
    # Hard voting has no predict_proba: AUC must come from hard labels, not a 0.5 placeholder
    ev_hard = evaluate_model(ensembles["voting_hard"], X_test, y_test, model_name="voting_hard")
    assert ev_hard["score_source"] == "hard_labels"


def test_weighted_ensemble_uses_validation_split(dummy_dataset):
    X_train, y_train, X_test, y_test = dummy_dataset
    ens = build_ensemble_models(random_state=42)["weighted_ensemble"]
    ens.fit(X_train[:350], y_train[:350], X_val=X_train[350:], y_val=y_train[350:])
    assert abs(ens.normalized_weights_.sum() - 1.0) < 1e-9
    assert ens.validation_f1_ is not None
    probs = ens.predict_proba(X_test)
    assert probs.shape == (len(X_test), 2)
    assert np.allclose(probs.sum(axis=1), 1.0)


def test_cross_validation_and_bootstrap(dummy_dataset):
    X_train, y_train, X_test, y_test = dummy_dataset
    model = get_baseline_models(random_state=42)["random_forest"]
    cv = cross_validate_classifier(model, X_train, y_train, "random_forest", n_splits=3)
    assert len(cv["f1_folds"]) == 3
    assert 0.0 <= cv["f1_mean"] <= 1.0
    assert cv["f1_std"] >= 0.0
    model.fit(X_train, y_train)
    ci = bootstrap_confidence_intervals(model, X_test, y_test, n_bootstrap=20)
    lo, hi = ci["f1_ci95"]
    assert lo <= ci["f1_mean"] <= hi


def test_per_class_recall(dummy_dataset):
    X_train, y_train, X_test, y_test = dummy_dataset
    model = get_baseline_models(random_state=42)["decision_tree"].fit(X_train, y_train)
    labels = np.where(y_test == 1, "DoS Hulk", "BENIGN")
    rows = per_class_recall(model, X_test, y_test, labels)
    assert {r["class"] for r in rows} == {"BENIGN", "DoS Hulk"}
    for r in rows:
        assert r["n_detected"] <= r["n_test"]
        assert 0.0 <= r["recall"] <= 1.0


def test_resampling_strategies(dummy_dataset):
    X_train, y_train, _, _ = dummy_dataset
    n_min = int((y_train == 1).sum())
    n_maj = int((y_train == 0).sum())
    Xs, ys = resample_training_data(X_train, y_train, "smote")
    assert (ys == 1).sum() == (ys == 0).sum() == n_maj
    Xu, yu = resample_training_data(X_train, y_train, "random_undersample")
    assert (yu == 1).sum() == (yu == 0).sum() == n_min
    Xn, yn = resample_training_data(X_train, y_train, "none")
    assert len(yn) == len(y_train)
    with pytest.raises(ValueError):
        resample_training_data(X_train, y_train, "bogus")


def test_imbalance_study_shape(dummy_dataset):
    X_train, y_train, X_test, y_test = dummy_dataset
    res = run_imbalance_study(X_train, y_train, X_test, y_test, strategies=["none", "class_weight"])
    assert len(res["records"]) == 2 * 3
    assert set(res["best_strategy_by_model"]) == {"logistic_regression", "decision_tree", "random_forest"}


def test_search_methods_share_protocol(dummy_dataset):
    X_train, y_train, _, _ = dummy_dataset
    from ml.models import optimization as opt
    opt.RF_GRID = {"n_estimators": [20, 40], "max_depth": [3]}
    optimizer = HyperparameterOptimizer(n_trials=3, cv_folds=3, random_state=42)
    _, grid = optimizer.grid_search_random_forest(X_train, y_train)
    _, rnd = optimizer.random_search_random_forest(X_train, y_train, n_iter=3)
    _, tpe = optimizer.optimize_random_forest(X_train, y_train)
    assert grid["n_trials"] == 2 and rnd["n_trials"] == 3 and tpe["n_trials"] == 3
    for r in (grid, rnd, tpe):
        assert 0.0 <= r["best_cv_f1"] <= 1.0
        assert r["wall_time_sec"] >= 0
        assert len(r["trials_history"]) == r["n_trials"]
