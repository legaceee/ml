"""
Ensemble Learning Architectures for Cyber Attack Detection.

Implements the four ensemble families covered in the course plus one custom
blend:

1. Voting      - hard (majority label) and soft (average probability)
2. Bagging     - Extra-Trees (bootstrap + random feature subsets; Random Forest
                 is also bagging and is covered by the tuned RF)
3. Boosting    - AdaBoost, Gradient Boosting (XGBoost is covered separately)
4. Stacking    - heterogeneous base learners + logistic-regression meta-learner
                 trained on 5-fold out-of-fold predictions
5. Weighted    - soft-vote whose weights are tuned on the VALIDATION split

Base learners deliberately span three hypothesis families: trees (RF, XGB),
linear (Logistic Regression) and max-margin (RBF-SVM). Diversity of inductive
bias is what makes an ensemble worth more than its parts.

Cost note: an RBF-SVM with ``probability=True`` runs an internal 5-fold Platt
calibration (~6x the cost of a plain fit). Hard voting and stacking do not
need calibrated probabilities (stacking can use ``decision_function``), so
those ensembles receive a plain SVC.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.svm import SVC
from xgboost import XGBClassifier


class WeightedAveragingEnsemble(BaseEstimator, ClassifierMixin):
    """Weighted probability blend whose weights are chosen on validation data."""

    def __init__(self, estimators: List[Tuple[str, Any]], weights: Optional[List[float]] = None, n_candidates: int = 60, random_state: int = 42):
        self.estimators = estimators
        self.weights = weights
        self.n_candidates = n_candidates
        self.random_state = random_state
        self.fitted_estimators_: List[Any] = []
        self.normalized_weights_: np.ndarray = np.array([])
        self.classes_ = np.array([0, 1])
        self.validation_f1_: Optional[float] = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None):
        self.fitted_estimators_ = []
        for _, est in self.estimators:
            est = clone(est)
            est.fit(X_train, y_train)
            self.fitted_estimators_.append(est)

        n = len(self.fitted_estimators_)
        if self.weights is not None:
            w = np.array(self.weights, dtype=np.float64)
            self.normalized_weights_ = w / np.sum(w)
        elif X_val is not None and y_val is not None:
            self.normalized_weights_ = self._optimize_weights(X_val, y_val)
        else:
            self.normalized_weights_ = np.ones(n) / n
        return self

    def _optimize_weights(self, X_val: np.ndarray, y_val: np.ndarray) -> np.ndarray:
        probas = [est.predict_proba(X_val)[:, 1] for est in self.fitted_estimators_]
        n = len(probas)
        rng = np.random.RandomState(self.random_state)
        candidates = [np.ones(n) / n] + [np.eye(n)[i] for i in range(n)] + [rng.dirichlet(np.ones(n)) for _ in range(self.n_candidates)]

        best_f1, best_w = -1.0, np.ones(n) / n
        for w in candidates:
            blended = np.zeros(len(X_val))
            for i, p in enumerate(probas):
                blended += w[i] * p
            score = f1_score(y_val, (blended >= 0.5).astype(int), zero_division=0)
            if score > best_f1:
                best_f1, best_w = score, w
        self.validation_f1_ = float(best_f1)
        return best_w

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        blended_pos = np.zeros(len(X))
        for i, est in enumerate(self.fitted_estimators_):
            blended_pos += self.normalized_weights_[i] * est.predict_proba(X)[:, 1]
        return np.column_stack([1.0 - blended_pos, blended_pos])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def build_ensemble_models(
    random_state: int = 42,
    optimized_rf: Optional[Any] = None,
    optimized_xgb: Optional[Any] = None,
) -> Dict[str, Any]:
    """Build the suite of Voting, Bagging, Boosting, Stacking and Weighted ensembles."""
    rf = clone(optimized_rf) if optimized_rf is not None else RandomForestClassifier(
        n_estimators=100, max_depth=12, random_state=random_state, class_weight="balanced", n_jobs=-1
    )
    xgb = clone(optimized_xgb) if optimized_xgb is not None else XGBClassifier(
        n_estimators=100, max_depth=6, eval_metric="logloss", random_state=random_state, n_jobs=-1, tree_method="hist"
    )
    lr = LogisticRegression(max_iter=2000, random_state=random_state, class_weight="balanced")
    svc_proba = SVC(kernel="rbf", probability=True, random_state=random_state, class_weight="balanced", cache_size=1000)
    svc_plain = SVC(kernel="rbf", probability=False, random_state=random_state, class_weight="balanced", cache_size=1000)

    def learners(svc):
        return [("rf", clone(rf)), ("xgb", clone(xgb)), ("lr", clone(lr)), ("svc", clone(svc))]

    ensembles = {
        "voting_hard": VotingClassifier(estimators=learners(svc_plain), voting="hard", n_jobs=1),
        "voting_soft": VotingClassifier(estimators=learners(svc_proba), voting="soft", n_jobs=1),
        "bagging_extra_trees": ExtraTreesClassifier(
            n_estimators=150, bootstrap=True, random_state=random_state, class_weight="balanced", n_jobs=-1
        ),
        "adaboost": AdaBoostClassifier(n_estimators=100, learning_rate=0.5, random_state=random_state),
        "gradient_boosting": GradientBoostingClassifier(n_estimators=150, learning_rate=0.1, max_depth=4, random_state=random_state),
        "stacking": StackingClassifier(
            estimators=learners(svc_plain),
            final_estimator=LogisticRegression(max_iter=2000, random_state=random_state),
            cv=5, n_jobs=1, passthrough=False,
        ),
        "weighted_ensemble": WeightedAveragingEnsemble(estimators=learners(svc_proba), random_state=random_state),
    }
    return ensembles


ENSEMBLE_FAMILY = {
    "voting_hard": "Voting",
    "voting_soft": "Voting",
    "bagging_extra_trees": "Bagging",
    "adaboost": "Boosting",
    "gradient_boosting": "Boosting",
    "stacking": "Stacking",
    "weighted_ensemble": "Weighted Voting",
}
