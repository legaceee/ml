"""
Data Cleaning Module for Network Intrusion Detection Datasets.

Handles infinite values, NaNs, duplicate records, zero-variance columns,
and column header sanitization with comprehensive audit logging.
"""

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd


class DataCleaner:
    def __init__(self, drop_duplicates: bool = True, drop_constants: bool = True):
        self.drop_duplicates = drop_duplicates
        self.drop_constants = drop_constants
        self.constant_columns: List[str] = []
        self.audit_log: Dict = {}

    def clean_dataframe(self, df: pd.DataFrame, is_train: bool = True) -> Tuple[pd.DataFrame, Dict]:
        """
        Clean DataFrame by sanitizing headers, handling inf/nan, and removing constants.
        
        Args:
            df: Input raw or semi-processed DataFrame.
            is_train: If True, identifies and records constant columns to drop.
                      If False, drops the previously identified constant columns.
                      
        Returns:
            Tuple of (cleaned_df, cleaning_audit_dict)
        """
        initial_rows, initial_cols = df.shape
        cleaned = df.copy()

        # 1. Clean column names
        cleaned.columns = cleaned.columns.str.strip().str.replace(r"\s+", " ", regex=True)

        # 2. Standardize string representations of infinity and NaNs
        cleaned = cleaned.replace([np.inf, -np.inf, "Infinity", "infinity", "inf", "-inf", "NaN", "nan"], np.nan)

        # 3. Separate Label column if present
        label_col = None
        for candidate in ["Label", "label", "class", "Class"]:
            if candidate in cleaned.columns:
                label_col = candidate
                break

        # 4. Handle duplicates
        duplicates_removed = 0
        if self.drop_duplicates and is_train:
            pre_dup_count = len(cleaned)
            cleaned = cleaned.drop_duplicates()
            duplicates_removed = pre_dup_count - len(cleaned)

        # 5. Handle infinite and NaN counts before imputing
        inf_nan_counts = cleaned.isna().sum().sum()

        # 6. Numerical column conversion
        feature_cols = [c for c in cleaned.columns if c != label_col]
        for col in feature_cols:
            cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")

        # 7. Constant Column Detection & Removal
        if is_train and self.drop_constants:
            # Columns with only 1 unique value or zero variance
            constants = []
            for col in feature_cols:
                nunique = cleaned[col].nunique(dropna=True)
                if nunique <= 1:
                    constants.append(col)
            self.constant_columns = constants

        # Drop identified constant columns
        if self.constant_columns:
            drop_targets = [c for c in self.constant_columns if c in cleaned.columns]
            cleaned = cleaned.drop(columns=drop_targets)

        # 8. Compile audit summary
        final_rows, final_cols = cleaned.shape
        self.audit_log = {
            "initial_shape": (initial_rows, initial_cols),
            "final_shape": (final_rows, final_cols),
            "duplicates_removed": duplicates_removed,
            "nan_inf_cells_found": int(inf_nan_counts),
            "constant_columns_dropped": self.constant_columns,
            "retained_features_count": final_cols - (1 if label_col in cleaned.columns else 0)
        }

        return cleaned, self.audit_log
