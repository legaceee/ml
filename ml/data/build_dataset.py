"""
CIC-IDS2017 Dataset Builder.

The full CIC-IDS2017 corpus (8 CSV files, ~2.83 million flows, ~885 MB) is too
large to train 15+ models (including RBF-SVM, KNN and 5-fold stacking) on a
single workstation in reasonable time. This module:

1. Downloads the 8 official "MachineLearningCSV" day-files from a public mirror
   (Hugging Face: c01dsnap/CIC-IDS2017 - identical content to the UNB release).
2. Merges them into one frame and records the TRUE class distribution.
3. Draws a *class-capped stratified sample*:
     - Every attack class is kept in full if it has <= `attack_cap` rows
       (rare classes such as Heartbleed (11), SQL Injection (21) and
       Infiltration (36) are therefore never lost).
     - Large attack classes are randomly down-sampled to `attack_cap`.
     - BENIGN is randomly down-sampled to `benign_rows`.
   The result is still imbalanced (roughly 3:1 benign:attack overall, and
   thousands:1 for the rarest attacks), which keeps the "imbalanced dataset
   handling" question meaningful while making the experiments tractable.

Usage:
    python -m ml.data.build_dataset --download            # fetch the 8 CSVs
    python -m ml.data.build_dataset --build               # build the sample CSV
    python -m ml.data.build_dataset --download --build    # both
"""

import argparse
import json
import sys
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

RAW_DIR = Path(__file__).resolve().parent / "raw"
FULL_DIR = RAW_DIR / "cicids2017_full"
SAMPLE_FILE = RAW_DIR / "cicids2017_sample.csv"
MANIFEST_FILE = RAW_DIR / "cicids2017_sample_manifest.json"

MIRROR_BASE = "https://huggingface.co/datasets/c01dsnap/CIC-IDS2017/resolve/main/"

DAY_FILES: List[str] = [
    "Monday-WorkingHours.pcap_ISCX.csv",
    "Tuesday-WorkingHours.pcap_ISCX.csv",
    "Wednesday-workingHours.pcap_ISCX.csv",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
    "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
]


def _normalise_label(label: str) -> str:
    """Fix the mojibake en-dash in the 'Web Attack' labels and strip spaces."""
    label = str(label).strip()
    # The original CSVs contain a Windows-1252 en dash (0x96) that pandas reads as '\x96' or 'ï¿½'
    for bad in ["ï¿½", "\x96", "�", "–", "—", "�"]:
        label = label.replace(bad, "-")
    label = label.replace(" - ", " - ").replace("  ", " ")
    label = " ".join(label.split())
    return label


def download_full_dataset(dest_dir: Path = FULL_DIR, force: bool = False) -> List[Path]:
    """Download the 8 day-files if they are not already present."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    downloaded = []
    for fname in DAY_FILES:
        target = dest_dir / fname
        if target.exists() and target.stat().st_size > 1_000_000 and not force:
            print(f"  [skip] {fname} already present ({target.stat().st_size / 1e6:.1f} MB)")
            downloaded.append(target)
            continue
        url = MIRROR_BASE + fname
        print(f"  [get ] {fname} ...", end="", flush=True)
        urllib.request.urlretrieve(url, target)
        print(f" {target.stat().st_size / 1e6:.1f} MB")
        downloaded.append(target)
    return downloaded


def load_full_dataset(full_dir: Path = FULL_DIR) -> pd.DataFrame:
    """Read and concatenate all day-files (only the columns are stripped here)."""
    files = sorted(full_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {full_dir}. Run with --download first.")
    frames = []
    for f in files:
        print(f"  reading {f.name} ...", end="", flush=True)
        df = pd.read_csv(f, low_memory=False, encoding="latin-1")
        df.columns = df.columns.str.strip()
        df["Label"] = df["Label"].map(_normalise_label)
        df["source_day_file"] = f.name
        frames.append(df)
        print(f" {len(df):,} rows")
    full = pd.concat(frames, axis=0, ignore_index=True)
    return full


def build_sample(
    full: pd.DataFrame,
    benign_rows: int = 80_000,
    attack_cap: int = 3_000,
    random_state: int = 42,
) -> pd.DataFrame:
    """Class-capped stratified sample (see module docstring)."""
    rng = np.random.RandomState(random_state)
    parts = []
    for label, group in full.groupby("Label"):
        cap = benign_rows if label == "BENIGN" else attack_cap
        if len(group) > cap:
            idx = rng.choice(group.index.values, size=cap, replace=False)
            parts.append(full.loc[idx])
        else:
            parts.append(group)
    sample = pd.concat(parts, axis=0)
    sample = sample.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    return sample


def build_and_save(
    benign_rows: int = 80_000,
    attack_cap: int = 3_000,
    random_state: int = 42,
    full_dir: Path = FULL_DIR,
    sample_file: Path = SAMPLE_FILE,
    manifest_file: Path = MANIFEST_FILE,
) -> Path:
    print(">>> Loading full CIC-IDS2017 corpus")
    full = load_full_dataset(full_dir)
    full_dist = full["Label"].value_counts().to_dict()
    print(f">>> Full corpus: {len(full):,} rows, {full.shape[1] - 2} features, {len(full_dist)} classes")

    print(f">>> Building class-capped stratified sample (benign_rows={benign_rows}, attack_cap={attack_cap})")
    sample = build_sample(full, benign_rows=benign_rows, attack_cap=attack_cap, random_state=random_state)
    sample_dist = sample["Label"].value_counts().to_dict()

    sample.drop(columns=["source_day_file"]).to_csv(sample_file, index=False)
    manifest: Dict = {
        "source": "CIC-IDS2017 MachineLearningCSV (UNB) via Hugging Face mirror c01dsnap/CIC-IDS2017",
        "day_files": [f.name for f in sorted(full_dir.glob('*.csv'))],
        "full_corpus_rows": int(len(full)),
        "full_corpus_class_distribution": {k: int(v) for k, v in full_dist.items()},
        "sampling_strategy": {
            "type": "class-capped stratified random sample",
            "benign_rows": benign_rows,
            "attack_cap_per_class": attack_cap,
            "random_state": random_state,
            "rationale": "Keep every rare attack class in full while bounding total size so that "
                         "RBF-SVM, KNN and 5-fold stacking remain tractable on a single workstation.",
        },
        "sample_rows": int(len(sample)),
        "sample_class_distribution": {k: int(v) for k, v in sample_dist.items()},
        "sample_attack_ratio": float((sample["Label"] != "BENIGN").mean()),
    }
    with open(manifest_file, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f">>> Sample written: {sample_file} ({len(sample):,} rows, attack ratio {manifest['sample_attack_ratio']:.3f})")
    print(f">>> Manifest written: {manifest_file}")
    return sample_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--download", action="store_true", help="Download the 8 CIC-IDS2017 day CSVs")
    parser.add_argument("--build", action="store_true", help="Build the class-capped stratified sample CSV")
    parser.add_argument("--benign-rows", type=int, default=80_000)
    parser.add_argument("--attack-cap", type=int, default=3_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.download and not args.build:
        parser.print_help()
        sys.exit(0)
    if args.download:
        print(">>> Downloading CIC-IDS2017 day files")
        download_full_dataset()
    if args.build:
        build_and_save(benign_rows=args.benign_rows, attack_cap=args.attack_cap, random_state=args.seed)
