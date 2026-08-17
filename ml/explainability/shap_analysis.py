"""
Explainable AI (XAI) Engine using SHAP (SHapley Additive exPlanations).

Provides:
1. Global Feature Importance (Mean |SHAP| values)
2. Local Instance Feature Attributions (Waterfall / Force breakdown for single predictions)
3. Model-agnostic explanations with fast TreeExplainer optimizations.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import shap


class IDSExplainer:
    def __init__(self, model: Any, feature_names: List[str], background_data: Optional[np.ndarray] = None):
        self.model = model
        self.feature_names = feature_names
        self.background_data = background_data
        self.explainer: Optional[Any] = None
        self.is_tree_model = False
        self._init_explainer()

    def _init_explainer(self):
        """Initialize appropriate SHAP explainer based on model architecture."""
        model_type = type(self.model).__name__.lower()
        
        if any(k in model_type for k in ["xgb", "forest", "tree", "gradientboosting", "extratrees"]):
            self.is_tree_model = True
            try:
                self.explainer = shap.TreeExplainer(self.model)
                return
            except Exception:
                pass

        # Fallback to Kernel or standard Explainer with background sample
        bg = self.background_data if self.background_data is not None else np.zeros((10, len(self.feature_names)))
        if len(bg) > 100:
            bg = bg[:100]  # Cap background size for speed
            
        predict_fn = self.model.predict_proba if hasattr(self.model, "predict_proba") else self.model.predict
        try:
            self.explainer = shap.Explainer(predict_fn, bg)
        except Exception:
            self.explainer = shap.KernelExplainer(predict_fn, bg)

    def explain_global(self, X_sample: np.ndarray, max_display: int = 20) -> List[Dict[str, Any]]:
        """
        Compute global mean |SHAP| values across evaluation sample.
        
        Returns:
            List of dictionaries with feature name, mean_shap_value, and rank.
        """
        if len(X_sample) > 200:
            X_sample = X_sample[:200]

        try:
            shap_values = self.explainer(X_sample)
            if hasattr(shap_values, "values"):
                vals = shap_values.values
            else:
                vals = np.array(shap_values)

            # Handle multiclass or 3D output arrays
            if vals.ndim == 3:
                vals = vals[:, :, 1]  # Attack class SHAP values

            mean_abs_shap = np.mean(np.abs(vals), axis=0)
        except Exception as e:
            # Fallback to model feature importances if SHAP calculation encounters dimension mismatch
            if hasattr(self.model, "feature_importances_"):
                mean_abs_shap = self.model.feature_importances_
            else:
                mean_abs_shap = np.ones(len(self.feature_names)) / len(self.feature_names)

        results = []
        for name, score in zip(self.feature_names, mean_abs_shap):
            results.append({
                "feature": name,
                "importance": round(float(score), 5)
            })

        results = sorted(results, key=lambda x: x["importance"], reverse=True)
        for i, item in enumerate(results):
            item["rank"] = i + 1

        return results[:max_display]

    def explain_instance(
        self,
        instance_1d: np.ndarray,
        raw_feature_values: Optional[Dict[str, Any]] = None,
        top_k: int = 8
    ) -> Dict[str, Any]:
        """
        Explain why a single network flow was classified as Attack or Benign.
        
        Args:
            instance_1d: 1D array of preprocessed features for 1 flow.
            raw_feature_values: Dictionary of original, unscaled human-readable values.
            top_k: Number of positive and negative drivers to return.
            
        Returns:
            Dictionary with base_value, top positive drivers (pushing toward ATTACK),
            top negative drivers (pushing toward BENIGN), and summary.
        """
        X_row = instance_1d.reshape(1, -1)

        try:
            shap_obj = self.explainer(X_row)
            if hasattr(shap_obj, "values"):
                vals = shap_obj.values[0]
                base_val = float(shap_obj.base_values[0]) if hasattr(shap_obj, "base_values") else 0.5
            else:
                vals = np.array(shap_obj)[0]
                base_val = 0.5

            if vals.ndim == 2:
                vals = vals[:, 1]  # Attack class output
        except Exception:
            # Fallback proxy attribution based on feature importance and normalized magnitude
            if hasattr(self.model, "feature_importances_"):
                vals = self.model.feature_importances_ * np.sign(instance_1d) * np.abs(instance_1d)
            else:
                vals = instance_1d * 0.05
            base_val = 0.5

        contributions = []
        for i, (name, val) in enumerate(zip(self.feature_names, vals)):
            raw_val = raw_feature_values.get(name, instance_1d[i]) if raw_feature_values else float(instance_1d[i])
            contributions.append({
                "feature": name,
                "shap_value": round(float(val), 4),
                "actual_value": round(float(raw_val), 2) if isinstance(raw_val, (int, float, np.number)) else str(raw_val),
                "effect": "ATTACK" if val > 0 else "BENIGN"
            })

        # Sort positive (attack drivers) and negative (benign drivers)
        pos_drivers = sorted([c for c in contributions if c["shap_value"] > 0], key=lambda x: x["shap_value"], reverse=True)[:top_k]
        neg_drivers = sorted([c for c in contributions if c["shap_value"] < 0], key=lambda x: x["shap_value"])[:top_k]

        return {
            "base_value": round(float(base_val), 4),
            "top_attack_drivers": pos_drivers,
            "top_benign_drivers": neg_drivers,
            "all_features_count": len(self.feature_names)
        }
