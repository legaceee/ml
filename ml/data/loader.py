"""
Dataset Loader for CIC-IDS2017 and Compatible Network Intrusion Datasets.

Provides chunked, memory-efficient loading, automatic header stripping,
column verification, and sample auditing.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np


class DatasetLoader:
    def __init__(self, raw_data_dir: Optional[str] = None):
        if raw_data_dir is None:
            self.raw_data_dir = Path(__file__).resolve().parent / "raw"
        else:
            self.raw_data_dir = Path(raw_data_dir)
            
    def list_available_files(self) -> List[Path]:
        """Return list of CSV files in raw directory."""
        if not self.raw_data_dir.exists():
            return []
        return list(self.raw_data_dir.glob("*.csv"))

    def load_dataset(
        self,
        filename: Optional[str] = None,
        sample_size: Optional[int] = None,
        chunk_size: Optional[int] = None,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Load dataset with automatic column trimming, memory optimization, and audit info.
        
        Args:
            filename: Name of the CSV file. If None, auto-selects first available or dev sample.
            sample_size: If specified, downsamples dataset for rapid prototyping.
            chunk_size: If specified, reads in chunks to reduce memory footprint.
            random_state: Random state for deterministic sampling.
            
        Returns:
            Tuple of (pd.DataFrame, audit_metadata_dict)
        """
        if filename is None:
            files = self.list_available_files()
            if not files:
                # Auto-generate dev sample if no files exist
                from ml.data.generate_dev_sample import generate_synthetic_dataset
                dev_path = self.raw_data_dir / "cicids2017_dev_sample.csv"
                generate_synthetic_dataset(num_samples=6000, random_seed=random_state, output_path=str(dev_path))
                filepath = dev_path
            else:
                # Prioritize full dataset if present, otherwise dev sample
                full_files = [f for f in files if "dev_sample" not in f.name]
                filepath = full_files[0] if full_files else files[0]
        else:
            filepath = self.raw_data_dir / filename
            if not filepath.exists():
                raise FileNotFoundError(f"Dataset file not found at {filepath}")
                
        is_synthetic = "dev_sample" in filepath.name.lower() or "synthetic" in filepath.name.lower()
        
        # Read dataset
        if chunk_size:
            chunks = []
            for chunk in pd.read_csv(filepath, chunksize=chunk_size, low_memory=False):
                # Standardize column whitespace
                chunk.columns = chunk.columns.str.strip()
                chunks.append(chunk)
            df = pd.concat(chunks, axis=0, ignore_index=True)
        else:
            df = pd.read_csv(filepath, low_memory=False)
            df.columns = df.columns.str.strip()

        # Audit information
        raw_rows = len(df)
        raw_cols = len(df.columns)
        
        if sample_size and sample_size < raw_rows:
            df = df.sample(n=sample_size, random_state=random_state).reset_index(drop=True)
            
        label_col = "Label" if "Label" in df.columns else df.columns[-1]
        class_dist = df[label_col].value_counts().to_dict()
        
        audit_metadata = {
            "source_file": filepath.name,
            "is_synthetic": is_synthetic,
            "total_rows_loaded": len(df),
            "original_rows": raw_rows,
            "num_features": raw_cols - 1,
            "label_column": label_col,
            "class_distribution": class_dist,
            "memory_usage_mb": float(df.memory_usage(deep=True).sum() / (1024 * 1024))
        }
        
        return df, audit_metadata


if __name__ == "__main__":
    loader = DatasetLoader()
    df, meta = loader.load_dataset()
    print("Dataset loaded successfully:")
    for k, v in meta.items():
        print(f"  {k}: {v}")
