"""
Label Encoding Module for Binary and Multiclass Cyber Attack Classification.

Provides standardized conversions between categorical attack strings and integer targets,
with zero risk of label target leakage.
"""

from typing import Dict, List, Tuple, Union
import numpy as np
import pandas as pd


class LabelEncoderIDS:
    # Standardized CIC-IDS2017 Multiclass Mapping
    STANDARD_MULTICLASS_MAP = {
        "BENIGN": 0,
        "DoS Hulk": 1,
        "PortScan": 2,
        "DDoS": 3,
        "FTP-Patator": 4,
        "SSH-Patator": 5,
        "Web Attack \u2013 Brute Force": 6,
        "Web Attack - Brute Force": 6,
        "Web Attack \u2013 XSS": 6,
        "Web Attack \u2013 Sql Injection": 6,
        "Bot": 7,
        "Infiltration": 8,
        "Heartbleed": 9,
        "DoS GoldenEye": 1,
        "DoS slowloris": 1,
        "DoS Slowhttptest": 1
    }

    MULTICLASS_NAMES = [
        "BENIGN",
        "DoS",
        "PortScan",
        "DDoS",
        "FTP-Patator",
        "SSH-Patator",
        "Web Attack",
        "Bot",
        "Infiltration",
        "Heartbleed"
    ]

    def __init__(self, mode: str = "binary"):
        """
        Args:
            mode: Either 'binary' (0=BENIGN, 1=ATTACK) or 'multiclass'
        """
        assert mode in ["binary", "multiclass"], "Mode must be 'binary' or 'multiclass'"
        self.mode = mode
        self.classes_: List[str] = ["BENIGN", "ATTACK"] if mode == "binary" else self.MULTICLASS_NAMES

    def transform(self, labels: Union[pd.Series, np.ndarray, List[str]]) -> np.ndarray:
        """Convert string labels to integer target array."""
        if isinstance(labels, (pd.Series, list)):
            labels = np.array(labels)
            
        labels = np.array([str(lbl).strip() for lbl in labels])

        if self.mode == "binary":
            # 0 for BENIGN, 1 for all attack classes
            return np.where(labels == "BENIGN", 0, 1).astype(np.int64)
        else:
            encoded = []
            for lbl in labels:
                # Check direct match or map to standard category
                val = self.STANDARD_MULTICLASS_MAP.get(lbl, None)
                if val is not None:
                    encoded.append(val)
                elif lbl == "BENIGN":
                    encoded.append(0)
                else:
                    encoded.append(1)  # Default generic attack bucket
            return np.array(encoded, dtype=np.int64)

    def inverse_transform(self, encoded_targets: Union[np.ndarray, List[int]]) -> List[str]:
        """Convert integer targets back to string class names."""
        encoded_targets = np.array(encoded_targets)
        if self.mode == "binary":
            return ["BENIGN" if t == 0 else "ATTACK" for t in encoded_targets]
        else:
            return [
                self.MULTICLASS_NAMES[t] if 0 <= t < len(self.MULTICLASS_NAMES) else "Unknown"
                for t in encoded_targets
            ]

    def get_class_names(self) -> List[str]:
        """Return list of human-readable class names."""
        return self.classes_
