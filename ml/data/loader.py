"""
Dataset Loader for CIC-IDS2017 and Compatible Network Intrusion Datasets.

Resolution order when no explicit filename is given:
1. ``raw/cicids2017_sample.csv``  - the class-capped stratified sample built by
   ``ml/data/build_dataset.py`` from the real CIC-IDS2017 corpus (preferred).
2. ``raw/cicids2017_full/*.csv``  - the raw day files; a sample is built on the fly.
3. Any other ``raw/*.csv`` that is not a dev fixture.
4. ``raw/cicids2017_dev_sample.csv`` - a clearly-labelled *synthetic* fixture,
   generated automatically so that the code base can be smoke-tested without
   the real data. Results on the fixture are NOT research results.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


class DatasetLoader:
    def __init__(self, raw_data_dir: Optional[str] = None):
        if raw_data_dir is None:
            self.raw_data_dir = Path(__file__).resolve().parent / "raw"
        else:
            self.raw_data_dir = Path(raw_data_dir)

    def list_available_files(self) -> List[Path]:
        """Return list of CSV files directly inside the raw directory."""
        if not self.raw_data_dir.exists():
            return []
        return sorted(self.raw_data_dir.glob("*.csv"))

    def _resolve_default_file(self, random_state: int) -> Path:
        sample_file = self.raw_data_dir / "cicids2017_sample.csv"
        if sample_file.exists():
            return sample_file

        full_dir = self.raw_data_dir / "cicids2017_full"
        if full_dir.exists() and any(full_dir.glob("*.csv")):
            from ml.data.build_dataset import build_and_save
            return build_and_save(random_state=random_state, full_dir=full_dir, sample_file=sample_file)

        files = self.list_available_files()
        real_files = [f for f in files if "dev_sample" not in f.name and "synthetic" not in f.name.lower()]
        if real_files:
            return real_files[0]

        from ml.data.generate_dev_sample import generate_synthetic_dataset
        dev_path = self.raw_data_dir / "cicids2017_dev_sample.csv"
        if not dev_path.exists():
            generate_synthetic_dataset(num_samples=6000, random_seed=random_state, output_path=str(dev_path))
        return dev_path

    def load_dataset(
        self,
        filename: Optional[str] = None,
        sample_size: Optional[int] = None,
        chunk_size: Optional[int] = None,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Load dataset with automatic column trimming and an audit dictionary.

        Args:
            filename: CSV name inside the raw dir. If None, see module docstring.
            sample_size: Optional uniform random down-sample for rapid prototyping.
            chunk_size: Optional chunked reading to reduce peak memory.
            random_state: Seed for deterministic sampling.
        """
        if filename is None:
            filepath = self._resolve_default_file(random_state)
        else:
            filepath = self.raw_data_dir / filename
            if not filepath.exists():
                raise FileNotFoundError(f"Dataset file not found at {filepath}")

        is_synthetic = "dev_sample" in filepath.name.lower() or "synthetic" in filepath.name.lower()

        read_kwargs = dict(low_memory=False, encoding="latin-1")
        if chunk_size:
            chunks = []
            for chunk in pd.read_csv(filepath, chunksize=chunk_size, **read_kwargs):
                chunk.columns = chunk.columns.str.strip()
                chunks.append(chunk)
            df = pd.concat(chunks, axis=0, ignore_index=True)
        else:
            df = pd.read_csv(filepath, **read_kwargs)
            df.columns = df.columns.str.strip()

        raw_rows = len(df)
        raw_cols = len(df.columns)

        if sample_size and sample_size < raw_rows:
            df = df.sample(n=sample_size, random_state=random_state).reset_index(drop=True)

        label_col = "Label" if "Label" in df.columns else df.columns[-1]
        df[label_col] = df[label_col].astype(str).str.strip()
        class_dist = df[label_col].value_counts().to_dict()

        manifest_file = self.raw_data_dir / "cicids2017_sample_manifest.json"
        manifest = None
        if filepath.name == "cicids2017_sample.csv" and manifest_file.exists():
            import json
            with open(manifest_file, "r") as f:
                manifest = json.load(f)

        audit_metadata = {
            "source_file": filepath.name,
            "is_synthetic": is_synthetic,
            "total_rows_loaded": len(df),
            "original_rows": raw_rows,
            "num_features": raw_cols - 1,
            "label_column": label_col,
            "class_distribution": {str(k): int(v) for k, v in class_dist.items()},
            "memory_usage_mb": float(df.memory_usage(deep=True).sum() / (1024 * 1024)),
            "sampling_manifest": manifest,
        }

        return df, audit_metadata


if __name__ == "__main__":
    loader = DatasetLoader()
    df, meta = loader.load_dataset()
    print("Dataset loaded successfully:")
    for k, v in meta.items():
        print(f"  {k}: {v}")
