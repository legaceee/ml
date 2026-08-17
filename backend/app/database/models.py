"""
SQLAlchemy ORM Models for Cyber Attack Detection System.

Stores prediction audits, registered model metadata, and experiment records.
"""

from datetime import datetime
from sqlalchemy import Boolean, Column, DateTime, Float, Integer, JSON, String, Text
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class PredictionLog(Base):
    __tablename__ = "prediction_logs"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    model_name = Column(String(100), index=True)
    prediction = Column(String(50))
    is_attack = Column(Boolean)
    confidence = Column(Float)
    latency_ms = Column(Float)
    attack_type = Column(String(100), nullable=True)
    input_features = Column(JSON)
    top_features = Column(JSON)


class ModelRegistryEntry(Base):
    __tablename__ = "model_registry"

    id = Column(Integer, primary_key=True, index=True)
    model_id = Column(String(100), unique=True, index=True)
    model_name = Column(String(150))
    version = Column(String(20), default="1.0.0")
    algorithm_type = Column(String(100))  # Baseline, Optimized, Ensemble
    dataset_name = Column(String(100))
    is_synthetic = Column(Boolean, default=True)
    features_count = Column(Integer)
    accuracy = Column(Float)
    precision = Column(Float)
    recall = Column(Float)
    f1_score = Column(Float)
    roc_auc = Column(Float)
    false_positive_rate = Column(Float)
    false_negative_rate = Column(Float)
    training_time_sec = Column(Float)
    inference_time_ms = Column(Float)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class ExperimentRecord(Base):
    __tablename__ = "experiment_records"

    id = Column(Integer, primary_key=True, index=True)
    experiment_id = Column(String(100), unique=True, index=True)
    model_name = Column(String(100))
    feature_method = Column(String(100))
    optimization = Column(String(100))
    ensemble = Column(String(100))
    accuracy = Column(Float)
    precision = Column(Float)
    recall = Column(Float)
    f1_score = Column(Float)
    roc_auc = Column(Float)
    training_time_sec = Column(Float)
    inference_time_ms = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)
