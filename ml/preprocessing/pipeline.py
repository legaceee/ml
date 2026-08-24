"""
Master Preprocessing Pipeline with Strict Zero-Leakage Splitting.

Order of operations (the order matters for leakage):
1. Load raw CSV.
2. Clean: strip headers, coerce Inf/NaN, drop exact duplicate rows, drop
   constant columns. Duplicates are dropped BEFORE splitting so that the same
   flow can never sit in both train and test.
3. Encode the label (binary: BENIGN=0, ATTACK=1). The original attack
   category is kept alongside for per-attack-type analysis.
4. Stratified split into Train (70%) / Validation (15%) / Test (15%).
5. Fit the median imputer + StandardScaler on TRAIN ONLY, then transform all
   three splits with those frozen parameters.
6. Persist splits, preprocessor and an audit summary.
"""

import json
from pathlib import Path
from typing import Dict, Optional

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

    @staticmethod
    def _class_counts(labels: np.ndarray) -> Dict[str, int]:
        vals, counts = np.unique(np.asarray(labels).astype(str), return_counts=True)
        return {str(k): int(v) for k, v in sorted(zip(vals, counts), key=lambda kv: -kv[1])}

    def run(self, filename: Optional[str] = None, sample_size: Optional[int] = None) -> Dict:
        """Execute the full preprocessing workflow with leakage-free splitting."""
        print("=" * 60)
        print(">>> Starting Preprocessing Pipeline (Leakage Prevention Protocol)")
        print("=" * 60)

        # 1. Load Raw Dataset
        loader = DatasetLoader(raw_data_dir=str(self.base_dir / "data" / "raw"))
        raw_df, audit_meta = loader.load_dataset(filename=filename, sample_size=sample_size, random_state=self.random_seed)
        print(f"[1/6] Raw dataset loaded: {raw_df.shape[0]:,} rows, {raw_df.shape[1]} columns. Synthetic: {audit_meta['is_synthetic']}")

        # 2. Data Cleaning (before the split so duplicates cannot straddle splits)
        cleaned_df, clean_audit = self.cleaner.clean_dataframe(raw_df, is_train=True)
        print(f"[2/6] Data cleaned. Duplicates removed: {clean_audit['duplicates_removed']:,}, "
              f"Inf/NaN cells: {clean_audit['nan_inf_cells_found']:,}, Constant columns dropped: {len(clean_audit['constant_columns_dropped'])}")

        # 3. Label Extraction & Target Encoding
        label_col = None
        for cand in ["Label", "label", "class", "Class"]:
            if cand in cleaned_df.columns:
                label_col = cand
                break
        if label_col is None:
            raise ValueError("No label column found in cleaned dataset.")

        y_raw = cleaned_df[label_col].astype(str).str.strip().values
        X_df = cleaned_df.drop(columns=[label_col])
        y = self.encoder.transform(y_raw)
        feature_names = list(X_df.columns)
        print(f"[3/6] Target encoded ({self.mode}). Features: {len(feature_names)}. "
              f"Class counts: {dict(zip(*np.unique(y, return_counts=True)))}")

        # 4. Stratified Train / Validation / Test Splitting.
        #    Stratify on the ORIGINAL category so rare attacks are spread across splits.
        strat_key = y_raw if np.min(np.unique(y_raw, return_counts=True)[1]) >= 3 else y
        test_val_ratio = self.val_ratio + self.test_ratio
        idx_all = np.arange(len(X_df))
        idx_train, idx_temp = train_test_split(idx_all, test_size=test_val_ratio, stratify=strat_key, random_state=self.random_seed)
        strat_temp = strat_key[idx_temp]
        if np.min(np.unique(strat_temp, return_counts=True)[1]) < 2:
            strat_temp = y[idx_temp]
        val_relative_ratio = self.val_ratio / test_val_ratio
        idx_val, idx_test = train_test_split(idx_temp, test_size=(1.0 - val_relative_ratio), stratify=strat_temp, random_state=self.random_seed)

        X_train_df, X_val_df, X_test_df = X_df.iloc[idx_train], X_df.iloc[idx_val], X_df.iloc[idx_test]
        y_train, y_val, y_test = y[idx_train], y[idx_val], y[idx_test]
        y_train_raw, y_val_raw, y_test_raw = y_raw[idx_train], y_raw[idx_val], y_raw[idx_test]

        print("[4/6] Stratified splits created:")
        print(f"      Train: {len(X_train_df):,} rows ({len(X_train_df)/len(X_df)*100:.1f}%)")
        print(f"      Val:   {len(X_val_df):,} rows ({len(X_val_df)/len(X_df)*100:.1f}%)")
        print(f"      Test:  {len(X_test_df):,} rows ({len(X_test_df)/len(X_df)*100:.1f}%)")

        # 5. Fit Preprocessor strictly on Train ONLY
        self.preprocessor.fit(X_train_df)
        X_train_scaled = self.preprocessor.transform(X_train_df)
        X_val_scaled = self.preprocessor.transform(X_val_df)
        X_test_scaled = self.preprocessor.transform(X_test_df)
        print("[5/6] Imputer + scaler fitted on Train only; Val and Test transformed with frozen parameters.")

        # 6. Save Splits and Preprocessing Artifacts
        split_data = {
            "X_train": X_train_scaled,
            "y_train": y_train,
            "X_val": X_val_scaled,
            "y_val": y_val,
            "X_test": X_test_scaled,
            "y_test": y_test,
            "y_train_labels": y_train_raw,
            "y_val_labels": y_val_raw,
            "y_test_labels": y_test_raw,
            "feature_names": feature_names,
            "class_names": self.encoder.get_class_names(),
            "X_train_df_raw": X_train_df.reset_index(drop=True),
            "X_val_df_raw": X_val_df.reset_index(drop=True),
            "X_test_df_raw": X_test_df.reset_index(drop=True),
            "is_synthetic": bool(audit_meta["is_synthetic"]),
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
                "train_count": int(len(X_train_df)),
                "val_count": int(len(X_val_df)),
                "test_count": int(len(X_test_df)),
                "train_attack_ratio": float(np.mean(y_train == 1)),
                "val_attack_ratio": float(np.mean(y_val == 1)),
                "test_attack_ratio": float(np.mean(y_test == 1)),
                "train_class_counts": self._class_counts(y_train_raw),
                "val_class_counts": self._class_counts(y_val_raw),
                "test_class_counts": self._class_counts(y_test_raw),
            },
            "preprocessing": {
                "imputation": "median (fitted on train split only)",
                "scaling": "StandardScaler z-score (fitted on train split only)",
                "duplicate_policy": "exact duplicate rows removed before splitting",
                "random_seed": self.random_seed,
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
