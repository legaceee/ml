"""
Ensemble Learning Architectures for Cyber Attack Detection.

Implements:
1. Voting Classifiers (Hard & Soft Voting)
2. Bagging (Random Forest, Extra Trees, Bagged Trees)
3. Boosting (AdaBoost, Gradient Boosting, XGBoost)
4. Stacking Classifier (Base Estimators + Out-of-Fold Meta Learner)
5. Weighted Soft Voting Ensemble with Validation Optimization
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
    VotingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.svm import SVC
from xgboost import XGBClassifier


class WeightedAveragingEnsemble(BaseEstimator, ClassifierMixin):
    """
    Weighted Probability Ensemble where model weights are tuned on validation data.
    """
    def __init__(self, estimators: List[Tuple[str, Any]], weights: Optional[List[float]] = None):
        self.estimators = estimators
        self.weights = weights
        self.fitted_estimators_: List[Any] = []
        self.normalized_weights_: np.ndarray = np.array([])

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None):
        # Fit base estimators on train
        self.fitted_estimators_ = []
        for name, est in self.estimators:
            est.fit(X_train, y_train)
            self.fitted_estimators_.append(est)

        if self.weights is not None:
            w = np.array(self.weights, dtype=np.float64)
            self.normalized_weights_ = w / np.sum(w)
        elif X_val is not None and y_val is not None:
            # Optimize weights via validation search
            self.normalized_weights_ = self._optimize_weights(X_val, y_val)
        else:
            # Uniform weights
            n = len(self.fitted_estimators_)
            self.normalized_weights_ = np.ones(n) / n

        return self

    def _optimize_weights(self, X_val: np.ndarray, y_val: np.ndarray) -> np.ndarray:
        probas = [est.predict_proba(X_val)[:, 1] for est in self.fitted_estimators_]
        best_f1 = -1.0
        best_weights = np.ones(len(self.fitted_estimators_)) / len(self.fitted_estimators_)

        # Grid search over candidate simplex weights
        rng = np.random.RandomState(42)
        candidate_weights = [best_weights]
        for _ in range(50):
            sample = rng.dirichlet(np.ones(len(self.fitted_estimators_)))
            candidate_weights.append(sample)

        for w in candidate_weights:
            blended = np.zeros(len(X_val))
            for i, p in enumerate(probas):
                blended += w[i] * p
            preds = (blended >= 0.5).astype(int)
            score = f1_score(y_val, preds, zero_division=0)
            if score > best_f1:
                best_f1 = score
                best_weights = w

        return best_weights

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        blended_pos = np.zeros(len(X))
        for i, est in enumerate(self.fitted_estimators_):
            p = est.predict_proba(X)[:, 1]
            blended_pos += self.normalized_weights_[i] * p
        blended_neg = 1.0 - blended_pos
        return np.column_stack([blended_neg, blended_pos])

    def predict(self, X: np.ndarray) -> np.ndarray:
        prob = self.predict_proba(X)[:, 1]
        return (prob >= 0.5).astype(int)


def build_ensemble_models(
    random_state: int = 42,
    optimized_rf: Optional[Any] = None,
    optimized_xgb: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Build suite of Voting, Bagging, Boosting, and Stacking ensemble classifiers.
    """
    rf = optimized_rf if optimized_rf is not None else RandomForestClassifier(
        n_estimators=100, max_depth=12, random_state=random_state, class_weight="balanced", n_jobs=-1
    )
    xgb = optimized_xgb if optimized_xgb is not None else XGBClassifier(
        n_estimators=100, max_depth=6, eval_metric="logloss", random_state=random_state, n_jobs=-1
    )
    lr = LogisticRegression(max_iter=1000, random_state=random_state, class_weight="balanced")
    svc = SVC(kernel="rbf", probability=True, random_state=random_state, class_weight="balanced")

    base_learners = [
        ("rf", rf),
        ("xgb", xgb),
        ("lr", lr),
        ("svc", svc)
    ]

    ensembles = {
        # 1. Voting Classifiers
        "voting_hard": VotingClassifier(
            estimators=base_learners,
            voting="hard",
            n_jobs=-1
        ),
        "voting_soft": VotingClassifier(
            estimators=base_learners,
            voting="soft",
            n_jobs=-1
        ),

        # 2. Bagging Classifiers
        "bagging_extra_trees": ExtraTreesClassifier(
            n_estimators=120,
            max_depth=14,
            random_state=random_state,
            class_weight="balanced",
            n_jobs=-1
        ),

        # 3. Boosting Classifiers
        "adaboost": AdaBoostClassifier(
            n_estimators=80,
            learning_rate=0.1,
            random_state=random_state
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=random_state
        ),

        # 4. Stacking Classifier (Meta Learner with Out-Of-Fold Cross-Validation)
        "stacking": StackingClassifier(
            estimators=[
                ("rf", rf),
                ("xgb", xgb),
                ("lr", lr),
                ("svc", svc)
            ],
            final_estimator=LogisticRegression(max_iter=1000, random_state=random_state),
            cv=5,
            n_jobs=-1,
            passthrough=False
        ),

        # 5. Weighted Soft Ensemble
        "weighted_ensemble": WeightedAveragingEnsemble(
            estimators=[
                ("rf", rf),
                ("xgb", xgb),
                ("lr", lr),
                ("svc", svc)
            ]
        )
    }

    return ensembles
