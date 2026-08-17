"""
Dimensionality Reduction Module: PCA Analysis for Network Flow Features.

Computes Principal Component Analysis, explained variance ratios,
and component loadings strictly fitted on Training split.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


class PCAReducer:
    def __init__(self, n_components: Optional[int] = None, variance_threshold: float = 0.95, random_state: int = 42):
        self.n_components = n_components
        self.variance_threshold = variance_threshold
        self.random_state = random_state
        self.pca: Optional[PCA] = None
        self.explained_variance_ratio_: np.ndarray = np.array([])
        self.cumulative_variance_: np.ndarray = np.array([])
        self.n_components_retained_: int = 0

    def fit(self, X_train: np.ndarray) -> "PCAReducer":
        """Fit PCA model strictly on training data."""
        # First fit full PCA to analyze variance spectrum
        full_pca = PCA(random_state=self.random_state)
        full_pca.fit(X_train)
        
        self.explained_variance_ratio_ = full_pca.explained_variance_ratio_
        self.cumulative_variance_ = np.cumsum(self.explained_variance_ratio_)
        
        if self.n_components is not None:
            k = min(self.n_components, X_train.shape[1])
        else:
            # Find minimum components required to exceed variance_threshold
            k = int(np.argmax(self.cumulative_variance_ >= self.variance_threshold) + 1)
            
        self.n_components_retained_ = k
        self.pca = PCA(n_components=k, random_state=self.random_state)
        self.pca.fit(X_train)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Project features onto principal component subspace."""
        if self.pca is None:
            raise ValueError("PCAReducer is not fitted yet.")
        return self.pca.transform(X)

    def fit_transform(self, X_train: np.ndarray) -> np.ndarray:
        return self.fit(X_train).transform(X_train)

    def get_variance_summary(self) -> Dict:
        """Return variance metrics dictionary."""
        return {
            "n_components_retained": int(self.n_components_retained_),
            "variance_explained_ratio": [float(v) for v in self.pca.explained_variance_ratio_] if self.pca else [],
            "total_variance_explained": float(np.sum(self.pca.explained_variance_ratio_)) if self.pca else 0.0,
            "cumulative_variance_curve": [float(v) for v in self.cumulative_variance_[:30]]
        }
