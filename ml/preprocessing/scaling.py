"""
Feature Imputation and Scaling Pipeline for Network Flow Features.

Ensures strict separation between fit (Train-only) and transform (Validation/Test/Inference).
"""

from typing import List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, RobustScaler


class NetworkFlowPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, scaler_type: str = "standard", strategy: str = "median"):
        """
        Args:
            scaler_type: 'standard', 'robust', or 'none'
            strategy: Imputation strategy ('median', 'mean')
        """
        self.scaler_type = scaler_type
        self.strategy = strategy
        self.imputer = SimpleImputer(strategy=strategy)
        
        if scaler_type == "standard":
            self.scaler = StandardScaler()
        elif scaler_type == "robust":
            self.scaler = RobustScaler()
        else:
            self.scaler = None
            
        self.feature_names_in_: List[str] = []

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Optional[np.ndarray] = None):
        """Fit imputer and scaler on Training data only."""
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = list(X.columns)
            X_arr = X.values
        else:
            X_arr = np.asarray(X)
            
        # Fit imputer
        X_imputed = self.imputer.fit_transform(X_arr)
        
        # Fit scaler
        if self.scaler is not None:
            self.scaler.fit(X_imputed)
            
        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Transform data using pre-fitted parameters without refitting."""
        if isinstance(X, pd.DataFrame):
            # Ensure columns align if provided as DataFrame
            if self.feature_names_in_:
                # Match training columns, fill missing with 0
                missing_cols = set(self.feature_names_in_) - set(X.columns)
                if missing_cols:
                    X = X.copy()
                    for col in missing_cols:
                        X[col] = 0.0
                X = X[self.feature_names_in_]
            X_arr = X.values
        else:
            X_arr = np.asarray(X)

        X_imputed = self.imputer.transform(X_arr)
        if self.scaler is not None:
            return self.scaler.transform(X_imputed)
        return X_imputed

    def fit_transform(self, X: Union[pd.DataFrame, np.ndarray], y: Optional[np.ndarray] = None) -> np.ndarray:
        return self.fit(X, y).transform(X)

    def get_feature_names(self) -> List[str]:
        return self.feature_names_in_
