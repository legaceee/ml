"""
Master Preprocessing Pipeline with Strict Zero-Leakage Splitting.

Splits dataset into Stratified Train (70%), Validation (15%), and Test (15%)
BEFORE fitting any scalers, imputers, or transformers.
"""

import json
import os
from pathlib import Path
from typing import Dict, Optional, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ml.data.loader import DatasetLoader
from ml.preprocessing.cleaning import DataCleaner
from ml.preprocessing.encoding import LabelEncoderIDS
from ml.preprocessing.scaling import NetworkFlowPreprocessor


class PipelineOrchestrator:
    def __init__(
        self,
        base_dir: Optional[str] = None,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        random_seed: int = 42,
        mode: str = "binary"
    ):
        if base_dir is None:
            self.base_dir = Path(__file__).resolve().parent.parent
        else:
            self.base_dir = Path(base_dir)

        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.random_seed = random_seed
        self.mode = mode

        self.splits_dir = self.base_dir / "data" / "splits"
        self.artifacts_dir = self.base_dir / "artifacts"
        self.preprocessors_dir = self.artifacts_dir / "preprocessors"
        self.metrics_dir = self.artifacts_dir / "metrics"

        self.splits_dir.mkdir(parents=True, exist_ok=True)
        self.preprocessors_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

        self.cleaner = DataCleaner(drop_duplicates=True, drop_constants=True)
        self.encoder = LabelEncoderIDS(mode=mode)
        self.preprocessor = NetworkFlowPreprocessor(scaler_type="standard", strategy="median")

    def run(self, filename: Optional[str] = None, sample_size: Optional[int] = None) -> Dict:
        """
        Execute full end-to-end preprocessing workflow with leakage-free splitting.
        """
        print("=" * 60)
        print(">>> Starting Preprocessing Pipeline (Leakage Prevention Protocol)")
        print("=" * 60)

        # 1. Load Raw Dataset
        loader = DatasetLoader(raw_data_dir=str(self.base_dir / "data" / "raw"))
        raw_df, audit_meta = loader.load_dataset(filename=filename, sample_size=sample_size, random_state=self.random_seed)
        print(f"[1/6] Raw dataset loaded: {raw_df.shape[0]} rows, {raw_df.shape[1]} columns. Synthetic: {audit_meta['is_synthetic']}")

        # 2. Data Cleaning
        cleaned_df, clean_audit = self.cleaner.clean_dataframe(raw_df, is_train=True)
        print(f"[2/6] Data cleaned. Deduplicated rows: {clean_audit['duplicates_removed']}, Constant columns dropped: {len(clean_audit['constant_columns_dropped'])}")

        # 3. Label Extraction & Target Encoding
        label_col = None
        for cand in ["Label", "label", "class", "Class"]:
            if cand in cleaned_df.columns:
                label_col = cand
                break
        if label_col is None:
            raise ValueError("No label column found in cleaned dataset.")

        y_raw = cleaned_df[label_col].values
        X_df = cleaned_df.drop(columns=[label_col])
        y = self.encoder.transform(y_raw)
        
        feature_names = list(X_df.columns)
        print(f"[3/6] Target encoded. Total Features: {len(feature_names)}. Class counts: {dict(pd.Series(y).value_counts())}")

        # 4. Stratified Train / Validation / Test Splitting
        # First split: Train vs Temp (Val + Test)
        test_val_ratio = self.val_ratio + self.test_ratio
        X_train_df, X_temp_df, y_train, y_temp = train_test_split(
            X_df, y,
            test_size=test_val_ratio,
            stratify=y,
            random_state=self.random_seed
        )

        # Second split: Validation vs Test
        val_relative_ratio = self.val_ratio / test_val_ratio
        X_val_df, X_test_df, y_val, y_test = train_test_split(
            X_temp_df, y_temp,
            test_size=(1.0 - val_relative_ratio),
            stratify=y_temp,
            random_state=self.random_seed
        )

        print(f"[4/6] Stratified Splits created:")
        print(f"      Train: {len(X_train_df)} rows ({len(X_train_df)/len(X_df)*100:.1f}%)")
        print(f"      Val:   {len(X_val_df)} rows ({len(X_val_df)/len(X_df)*100:.1f}%)")
        print(f"      Test:  {len(X_test_df)} rows ({len(X_test_df)/len(X_df)*100:.1f}%)")

        # 5. Fit Preprocessor strictly on Train ONLY
        self.preprocessor.fit(X_train_df)
        X_train_scaled = self.preprocessor.transform(X_train_df)
        X_val_scaled = self.preprocessor.transform(X_val_df)
        X_test_scaled = self.preprocessor.transform(X_test_df)
        print(f"[5/6] Preprocessor fitted exclusively on Train split; Val and Test transformed.")

        # 6. Save Splits and Preprocessing Artifacts
        split_data = {
            "X_train": X_train_scaled,
            "y_train": y_train,
            "X_val": X_val_scaled,
            "y_val": y_val,
            "X_test": X_test_scaled,
            "y_test": y_test,
            "feature_names": feature_names,
            "class_names": self.encoder.get_class_names(),
            "X_train_df_raw": X_train_df,  # unscaled for tree models if needed
            "X_val_df_raw": X_val_df,
            "X_test_df_raw": X_test_df
        }

        splits_file = self.splits_dir / "dataset_splits.joblib"
        joblib.dump(split_data, splits_file)
        print(f"[6/6] Saved dataset splits to: {splits_file}")

        preprocessor_file = self.preprocessors_dir / "preprocessor.joblib"
        joblib.dump({
            "cleaner": self.cleaner,
            "encoder": self.encoder,
            "preprocessor": self.preprocessor,
            "feature_names": feature_names,
            "mode": self.mode
        }, preprocessor_file)
        print(f"      Saved preprocessor pipeline to: {preprocessor_file}")

        # Save summary metadata JSON for API & UI
        summary = {
            "dataset_info": audit_meta,
            "cleaning_audit": {
                "initial_shape": list(clean_audit["initial_shape"]),
                "final_shape": list(clean_audit["final_shape"]),
                "duplicates_removed": clean_audit["duplicates_removed"],
                "nan_inf_cells_found": clean_audit["nan_inf_cells_found"],
                "constant_columns_dropped": clean_audit["constant_columns_dropped"]
            },
            "split_info": {
                "train_count": len(X_train_df),
                "val_count": len(X_val_df),
                "test_count": len(X_test_df),
                "train_attack_ratio": float(np.mean(y_train == 1)),
                "val_attack_ratio": float(np.mean(y_val == 1)),
                "test_attack_ratio": float(np.mean(y_test == 1))
            },
            "features": feature_names,
            "num_features": len(feature_names),
            "classes": self.encoder.get_class_names()
        }

        summary_file = self.metrics_dir / "dataset_summary.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)

        print(f">>> Preprocessing complete. Metadata stored at: {summary_file}")
        return summary


if __name__ == "__main__":
    pipeline = PipelineOrchestrator()
    summary = pipeline.run()
