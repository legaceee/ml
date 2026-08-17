"""
Feature Selection and Ranking Engine for Network Intrusion Detection.

Implements:
1. Correlation Filtering (Collinearity Pruning)
2. Mutual Information Scoring
3. Tree-based Feature Importance (Random Forest / XGBoost)
4. SelectKBest (ANOVA F-value / Mutual Info)
5. Recursive Feature Elimination (RFE)

All selectors are strictly fitted on Training splits to prevent leakage.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import (
    RFE,
    SelectKBest,
    f_classif,
    mutual_info_classif
)


class FeatureSelector:
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.selected_features_: Dict[str, List[str]] = {}
        self.feature_scores_: Dict[str, Dict[str, float]] = {}

    def correlation_filter(
        self,
        X_train: pd.DataFrame,
        threshold: float = 0.90
    ) -> List[str]:
        """
        Identify and remove highly correlated redundant features from training data.
        
        Args:
            X_train: Training DataFrame with named columns.
            threshold: Pearson correlation threshold above which one feature is dropped.
            
        Returns:
            List of retained feature names.
        """
        corr_matrix = X_train.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
        retained = [col for col in X_train.columns if col not in to_drop]
        
        self.selected_features_["correlation_filter"] = retained
        return retained

    def mutual_information(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        feature_names: List[str],
        top_k: Optional[int] = 30
    ) -> List[str]:
        """
        Compute Mutual Information scores between features and target.
        """
        mi_scores = mutual_info_classif(X_train, y_train, random_state=self.random_state)
        scores_dict = {name: float(score) for name, score in zip(feature_names, mi_scores)}
        self.feature_scores_["mutual_information"] = scores_dict
        
        sorted_features = sorted(scores_dict.items(), key=lambda x: x[1], reverse=True)
        if top_k:
            selected = [f[0] for f in sorted_features[:top_k]]
        else:
            selected = [f[0] for f in sorted_features]
            
        self.selected_features_[f"mutual_info_top_{top_k}"] = selected
        return selected

    def tree_importance(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        feature_names: List[str],
        top_k: Optional[int] = 30
    ) -> List[str]:
        """
        Rank features using Random Forest Gini impurity importance.
        """
        rf = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=self.random_state, n_jobs=-1)
        rf.fit(X_train, y_train)
        
        importances = rf.feature_importances_
        scores_dict = {name: float(imp) for name, imp in zip(feature_names, importances)}
        self.feature_scores_["tree_importance"] = scores_dict
        
        sorted_features = sorted(scores_dict.items(), key=lambda x: x[1], reverse=True)
        if top_k:
            selected = [f[0] for f in sorted_features[:top_k]]
        else:
            selected = [f[0] for f in sorted_features]
            
        self.selected_features_[f"tree_top_{top_k}"] = selected
        return selected

    def select_k_best(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        feature_names: List[str],
        k: int = 30,
        score_func=f_classif
    ) -> List[str]:
        """
        Select top K features using SelectKBest (ANOVA F-value or Mutual Info).
        """
        k_val = min(k, len(feature_names))
        selector = SelectKBest(score_func=score_func, k=k_val)
        selector.fit(X_train, y_train)
        
        mask = selector.get_support()
        selected = [feature_names[i] for i, val in enumerate(mask) if val]
        self.selected_features_[f"select_k_best_{k}"] = selected
        return selected

    def rfe_selection(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        feature_names: List[str],
        n_features_to_select: int = 25
    ) -> List[str]:
        """
        Recursive Feature Elimination using Random Forest.
        """
        estimator = RandomForestClassifier(n_estimators=50, max_depth=8, random_state=self.random_state, n_jobs=-1)
        rfe = RFE(estimator=estimator, n_features_to_select=n_features_to_select, step=5)
        rfe.fit(X_train, y_train)
        
        selected = [feature_names[i] for i, val in enumerate(rfe.support_) if val]
        self.selected_features_[f"rfe_{n_features_to_select}"] = selected
        return selected

    def get_summary(self) -> Dict:
        """Return comprehensive dictionary of all feature selection trials."""
        return {
            "feature_scores": self.feature_scores_,
            "selected_subsets": self.selected_features_
        }
