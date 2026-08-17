"""
Pydantic Schemas for Cyber Attack Detection API.

Validates incoming single flow telemetry, CSV batch uploads,
test sample benchmarks, and formats typed responses.
"""

import math
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


class FeatureAttribution(BaseModel):
    feature: str
    shap_value: float
    actual_value: Any
    effect: str


class PredictionRequest(BaseModel):
    model_name: Optional[str] = Field(default="optimized_xgboost", description="Selected ML model identifier")
    features: Dict[str, float] = Field(
        default_factory=lambda: {
            "Destination Port": 80.0,
            "Flow Duration": 12000.0,
            "Total Fwd Packets": 4.0,
            "Total Backward Packets": 4.0,
            "Total Length of Fwd Packets": 240.0,
            "Total Length of Bwd Packets": 520.0,
            "Flow Bytes/s": 63333.3,
            "Flow Packets/s": 666.6,
            "SYN Flag Count": 0.0,
            "ACK Flag Count": 1.0,
            "RST Flag Count": 0.0,
            "Init_Win_bytes_forward": 29200.0,
            "Init_Win_bytes_backward": 28960.0
        },
        description="Dictionary mapping network flow feature names to their numeric values"
    )

    @field_validator("features")
    @classmethod
    def validate_features(cls, v: Dict[str, Any]) -> Dict[str, float]:
        cleaned = {}
        for key, val in v.items():
            if val is None:
                continue
            try:
                num = float(val)
                if math.isnan(num) or math.isinf(num):
                    raise ValueError(f"Feature '{key}' contains invalid non-finite value ({val}).")
                cleaned[str(key).strip()] = num
            except (ValueError, TypeError) as e:
                raise ValueError(f"Invalid numeric value for feature '{key}': {val}")
        return cleaned


class PredictionResponse(BaseModel):
    prediction: str = Field(description="ATTACK or BENIGN")
    is_attack: bool
    confidence: float
    probability_attack: float
    probability_benign: float
    model_used: str
    model_version: str = "1.0.0"
    attack_type: Optional[str] = None
    latency_ms: float
    top_contributing_features: List[FeatureAttribution] = []
    debug_info: Optional[Dict[str, Any]] = None
    disclaimer: str = "ML-based cyber attack predictions are probabilistic and provide defensive guidance."


class TestSampleItem(BaseModel):
    sample_id: int
    actual: str
    actual_numeric: int
    predicted: str
    is_attack: bool
    is_correct: bool
    benign_probability: float
    attack_probability: float
    confidence: float
    model_used: str
    key_telemetry: Dict[str, Any]
    features: Dict[str, float]


class TestSamplesResponse(BaseModel):
    total_samples: int
    model_used: str
    samples: List[TestSampleItem]


class BatchPredictionSummary(BaseModel):
    total_records: int
    attacks_detected: int
    benign_count: int
    attack_percentage: float
    model_used: str
    processing_time_sec: float
    predictions_sample: List[Dict[str, Any]]
    class_breakdown: Dict[str, int]


class ModelInfo(BaseModel):
    model_id: str
    model_name: str
    algorithm_type: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float
    false_positive_rate: float
    false_negative_rate: float
    training_time_sec: float
    inference_time_ms: float
    is_synthetic: bool


class ConfusionMatrixData(BaseModel):
    tp: int
    tn: int
    fp: int
    fn: int


class MetricDetail(BaseModel):
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    pr_auc: float
    specificity: float
    false_positive_rate: float
    false_negative_rate: float
    confusion_matrix: ConfusionMatrixData
    roc_curve: List[Dict[str, float]]
    pr_curve: List[Dict[str, float]]
    training_time_sec: float
    inference_time_ms_per_1k: float


class GlobalFeatureImportance(BaseModel):
    rank: int
    feature: str
    importance: float


class ExplainResponse(BaseModel):
    model_name: str
    global_importance: List[GlobalFeatureImportance]


class LocalExplainRequest(BaseModel):
    model_name: Optional[str] = "optimized_xgboost"
    features: Dict[str, float]


class LocalExplainResponse(BaseModel):
    model_name: str
    base_value: float
    top_attack_drivers: List[FeatureAttribution]
    top_benign_drivers: List[FeatureAttribution]
