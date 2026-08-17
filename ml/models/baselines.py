"""
Baseline Machine Learning Classifiers for Intrusion Detection.

Implements standard unoptimized baselines across linear, tree, distance,
and boosting paradigms to establish benchmark comparisons.
"""

from typing import Any, Dict
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier


def get_baseline_models(random_state: int = 42, class_weight: str = "balanced") -> Dict[str, Any]:
    """
    Instantiate standard baseline models with default/balanced configurations.
    """
    models = {
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            random_state=random_state,
            class_weight=class_weight
        ),
        "decision_tree": DecisionTreeClassifier(
            random_state=random_state,
            class_weight=class_weight
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=100,
            random_state=random_state,
            class_weight=class_weight,
            n_jobs=-1
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=100,
            random_state=random_state,
            class_weight=class_weight,
            n_jobs=-1
        ),
        "support_vector_machine": SVC(
            kernel="rbf",
            probability=True,
            random_state=random_state,
            class_weight=class_weight
        ),
        "k_nearest_neighbors": KNeighborsClassifier(
            n_neighbors=5,
            n_jobs=-1
        ),
        "xgboost": XGBClassifier(
            n_estimators=100,
            random_state=random_state,
            eval_metric="logloss",
            n_jobs=-1
        )
    }
    return models
