import sys
from pathlib import Path

# Ensure project root is present in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import pytest

from ml.preprocessing.cleaning import DataCleaner
from ml.preprocessing.encoding import LabelEncoderIDS
from ml.preprocessing.scaling import NetworkFlowPreprocessor



def test_data_cleaner_inf_and_constants():
    df = pd.DataFrame({
        "Destination Port": [80, 443, 8080, 80],
        "Flow Duration": [1000, np.inf, 3000, 4000],
        "Flow Bytes/s": [500.0, np.nan, 700.0, 800.0],
        "Constant Col": [0, 0, 0, 0],
        "Label": ["BENIGN", "DoS Hulk", "BENIGN", "PortScan"]
    })

    cleaner = DataCleaner(drop_duplicates=True, drop_constants=True)
    cleaned_df, audit = cleaner.clean_dataframe(df, is_train=True)

    assert "Constant Col" not in cleaned_df.columns
    assert "Destination Port" in cleaned_df.columns
    assert audit["constant_columns_dropped"] == ["Constant Col"]
    assert audit["nan_inf_cells_found"] > 0


def test_label_encoder_binary_and_multiclass():
    encoder_bin = LabelEncoderIDS(mode="binary")
    labels = ["BENIGN", "DoS Hulk", "PortScan", "BENIGN"]
    y_bin = encoder_bin.transform(labels)
    np.testing.assert_array_equal(y_bin, np.array([0, 1, 1, 0]))

    inv = encoder_bin.inverse_transform(y_bin)
    assert inv == ["BENIGN", "ATTACK", "ATTACK", "BENIGN"]

    encoder_multi = LabelEncoderIDS(mode="multiclass")
    y_multi = encoder_multi.transform(labels)
    assert y_multi[0] == 0  # BENIGN
    assert y_multi[1] == 1  # DoS
    assert y_multi[2] == 2  # PortScan


def test_preprocessor_train_only_fitting():
    X_train = pd.DataFrame({
        "f1": [10.0, 20.0, 30.0, 40.0],
        "f2": [100.0, 200.0, np.nan, 400.0]
    })
    X_test = pd.DataFrame({
        "f1": [15.0, 25.0],
        "f2": [150.0, 250.0]
    })

    prep = NetworkFlowPreprocessor(scaler_type="standard", strategy="median")
    prep.fit(X_train)
    X_train_trans = prep.transform(X_train)
    X_test_trans = prep.transform(X_test)

    assert X_train_trans.shape == (4, 2)
    assert X_test_trans.shape == (2, 2)
    assert not np.isnan(X_train_trans).any()
    assert not np.isnan(X_test_trans).any()
