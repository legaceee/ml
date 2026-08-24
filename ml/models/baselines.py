"""
Baseline Machine Learning Classifiers for Intrusion Detection.

Seven algorithms spanning the hypothesis families covered in the course:
linear (Logistic Regression), tree (Decision Tree), bagged trees (Random
Forest, Extra Trees), max-margin (RBF-SVM), instance-based (KNN) and
gradient boosting (XGBoost). Hyperparameters are library defaults except
``class_weight="balanced"`` (cost-sensitive learning for the imbalanced
target), so these numbers are the honest "before tuning" reference.
"""

from typing import Any, Dict, Optional

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


def get_baseline_models(random_state: int = 42, class_weight: Optional[str] = "balanced") -> Dict[str, Any]:
    """Instantiate the baseline models with default hyperparameters."""
    return {
        "logistic_regression": LogisticRegression(max_iter=2000, random_state=random_state, class_weight=class_weight),
        "decision_tree": DecisionTreeClassifier(random_state=random_state, class_weight=class_weight),
        "random_forest": RandomForestClassifier(n_estimators=100, random_state=random_state, class_weight=class_weight, n_jobs=-1),
        "extra_trees": ExtraTreesClassifier(n_estimators=100, random_state=random_state, class_weight=class_weight, n_jobs=-1),
        "support_vector_machine": SVC(kernel="rbf", probability=True, random_state=random_state, class_weight=class_weight, cache_size=1000),
        "k_nearest_neighbors": KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
        "xgboost": XGBClassifier(n_estimators=100, random_state=random_state, eval_metric="logloss", n_jobs=-1, tree_method="hist"),
    }


BASELINE_FAMILY = {
    "logistic_regression": "Linear",
    "decision_tree": "Tree",
    "random_forest": "Bagging (trees)",
    "extra_trees": "Bagging (trees)",
    "support_vector_machine": "Max-margin (RBF kernel)",
    "k_nearest_neighbors": "Instance-based",
    "xgboost": "Gradient boosting",
}
