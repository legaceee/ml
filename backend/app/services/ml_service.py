"""
Machine Learning Core Service for Backend API.

Manages model lifecycle, preprocessor transformations, caching,
real-time inference, and SHAP explainability.
"""

import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple
import joblib
import numpy as np
import pandas as pd

from ml.explainability.shap_analysis import IDSExplainer


class MLService:
    _instance: Optional["MLService"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MLService, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        self.base_dir = Path(__file__).resolve().parent.parent.parent.parent / "ml"
        self.artifacts_dir = self.base_dir / "artifacts"
        self.models_dir = self.artifacts_dir / "models"
        self.preprocessors_dir = self.artifacts_dir / "preprocessors"
        self.metrics_dir = self.artifacts_dir / "metrics"
        self.experiments_dir = self.artifacts_dir / "experiments"
        self.splits_file = self.base_dir / "data" / "splits" / "dataset_splits.joblib"

        self.preprocessor_data: Optional[Dict] = None
        self.models: Dict[str, Any] = {}
        self.explainers: Dict[str, IDSExplainer] = {}
        self.master_metrics: Optional[Dict] = None
        self.dataset_summary: Optional[Dict] = None
        self.splits_data: Optional[Dict] = None
        self.benign_baseline: Dict[str, float] = {}

        self._load_preprocessor()
        self._load_available_models()
        self._load_metrics()
        self._load_splits_and_baseline()

    def _load_preprocessor(self):
        prep_file = self.preprocessors_dir / "preprocessor.joblib"
        if prep_file.exists():
            self.preprocessor_data = joblib.load(prep_file)

    def _load_available_models(self):
        if not self.models_dir.exists():
            return
        for m_file in self.models_dir.glob("*.joblib"):
            model_name = m_file.stem
            try:
                self.models[model_name] = joblib.load(m_file)
            except Exception as e:
                print(f"[MLService] Could not load model {model_name}: {e}")

    def _load_metrics(self):
        m_file = self.metrics_dir / "master_comparison.json"
        if m_file.exists():
            with open(m_file, "r") as f:
                self.master_metrics = json.load(f)

        s_file = self.metrics_dir / "dataset_summary.json"
        if s_file.exists():
            with open(s_file, "r") as f:
                self.dataset_summary = json.load(f)

    def _load_splits_and_baseline(self):
        if self.splits_file.exists():
            try:
                self.splits_data = joblib.load(self.splits_file)
                if "X_train_df_raw" in self.splits_data and "y_train" in self.splits_data:
                    X_train_raw = self.splits_data["X_train_df_raw"]
                    y_train = self.splits_data["y_train"]
                    benign_mask = (y_train == 0)
                    if np.sum(benign_mask) > 0:
                        self.benign_baseline = X_train_raw[benign_mask].median().to_dict()
                    else:
                        self.benign_baseline = X_train_raw.median().to_dict()
            except Exception as e:
                print(f"[MLService] Warning loading dataset splits: {e}")

        # Fallback if baseline empty
        if not self.benign_baseline and self.preprocessor_data:
            feat_names = self.get_feature_names()
            self.benign_baseline = {f: 0.0 for f in feat_names}

    def get_feature_names(self) -> List[str]:
        if self.preprocessor_data and "feature_names" in self.preprocessor_data:
            return self.preprocessor_data["feature_names"]
        return []

    def get_loaded_models_list(self) -> List[str]:
        return list(self.models.keys())

    def get_explainer(self, model_name: str) -> Optional[IDSExplainer]:
        if model_name in self.explainers:
            return self.explainers[model_name]

        model = self.models.get(model_name)
        if model is None:
            return None

        feature_names = self.get_feature_names()
        explainer = IDSExplainer(model=model, feature_names=feature_names)
        self.explainers[model_name] = explainer
        return explainer

    def prepare_feature_vector(self, features: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """
        Build a mathematically consistent 61-feature vector from input parameters.
        Uses baseline telemetry profile for unprovided flow attributes and computes
        correlated flow statistics (packet rates, byte rates, segment sizes).
        """
        expected_features = self.get_feature_names()
        
        # Start from benign baseline profile
        row_dict = self.benign_baseline.copy() if self.benign_baseline else {f: 0.0 for f in expected_features}
        
        # Overlay explicitly provided user features
        for f, val in features.items():
            if f in row_dict:
                try:
                    num_val = float(val)
                    if not (np.isnan(num_val) or np.isinf(num_val)):
                        row_dict[f] = num_val
                except (ValueError, TypeError):
                    continue

        # If primary interactive parameters were provided, update physically derived flow features
        fwd_pkts = max(1.0, float(row_dict.get("Total Fwd Packets", 1)))
        bwd_pkts = float(row_dict.get("Total Backward Packets", 0))
        tot_pkts = fwd_pkts + bwd_pkts
        
        fwd_len = float(row_dict.get("Total Length of Fwd Packets", 0))
        bwd_len = float(row_dict.get("Total Length of Bwd Packets", 0))
        tot_bytes = fwd_len + bwd_len
        
        dur_us = max(1.0, float(row_dict.get("Flow Duration", 1000)))
        dur_sec = dur_us / 1e6

        # Flow throughput & packet rates if not explicitly customized
        if "Flow Bytes/s" not in features or float(features.get("Flow Bytes/s", 0)) <= 0:
            row_dict["Flow Bytes/s"] = tot_bytes / dur_sec
        if "Flow Packets/s" not in features or float(features.get("Flow Packets/s", 0)) <= 0:
            row_dict["Flow Packets/s"] = tot_pkts / dur_sec

        row_dict["Fwd Packets/s"] = fwd_pkts / dur_sec
        row_dict["Bwd Packets/s"] = bwd_pkts / dur_sec

        # Packet length means & segment sizes
        fwd_mean = fwd_len / fwd_pkts
        bwd_mean = bwd_len / max(1.0, bwd_pkts) if bwd_pkts > 0 else 0.0
        pkt_mean = tot_bytes / max(1.0, tot_pkts)

        if "Fwd Packet Length Mean" in row_dict and "Fwd Packet Length Mean" not in features:
            row_dict["Fwd Packet Length Mean"] = fwd_mean
        if "Bwd Packet Length Mean" in row_dict and "Bwd Packet Length Mean" not in features:
            row_dict["Bwd Packet Length Mean"] = bwd_mean
        if "Packet Length Mean" in row_dict and "Packet Length Mean" not in features:
            row_dict["Packet Length Mean"] = pkt_mean
        if "Average Packet Size" in row_dict and "Average Packet Size" not in features:
            row_dict["Average Packet Size"] = pkt_mean
        if "Avg Fwd Segment Size" in row_dict and "Avg Fwd Segment Size" not in features:
            row_dict["Avg Fwd Segment Size"] = fwd_mean
        if "Avg Bwd Segment Size" in row_dict and "Avg Bwd Segment Size" not in features:
            row_dict["Avg Bwd Segment Size"] = bwd_mean

        # Subflow packet and byte mirrors
        if "Subflow Fwd Packets" in row_dict and "Subflow Fwd Packets" not in features:
            row_dict["Subflow Fwd Packets"] = fwd_pkts
        if "Subflow Fwd Bytes" in row_dict and "Subflow Fwd Bytes" not in features:
            row_dict["Subflow Fwd Bytes"] = fwd_len
        if "Subflow Bwd Packets" in row_dict and "Subflow Bwd Packets" not in features:
            row_dict["Subflow Bwd Packets"] = bwd_pkts
        if "Subflow Bwd Bytes" in row_dict and "Subflow Bwd Bytes" not in features:
            row_dict["Subflow Bwd Bytes"] = bwd_len

        if "Down/Up Ratio" in row_dict and "Down/Up Ratio" not in features:
            row_dict["Down/Up Ratio"] = bwd_pkts / fwd_pkts

        # Build clean aligned DataFrame
        final_row = {f: float(row_dict.get(f, 0.0)) for f in expected_features}
        df_row = pd.DataFrame([final_row])
        return df_row, final_row

    def predict_single(
        self,
        features: Dict[str, Any],
        model_name: str = "optimized_xgboost"
    ) -> Dict[str, Any]:
        """
        Execute single flow inference with verified class indices and SHAP attribution.
        """
        if not self.models:
            self._load_available_models()
        if not self.preprocessor_data:
            self._load_preprocessor()
        if not self.benign_baseline:
            self._load_splits_and_baseline()

        if model_name not in self.models:
            if self.models:
                model_name = list(self.models.keys())[0]
            else:
                raise RuntimeError("No ML models currently loaded in registry.")

        model = self.models[model_name]
        expected_features = self.get_feature_names()

        # Build feature vector DataFrame using verified imputation protocol
        df_row, full_feature_row = self.prepare_feature_vector(features)

        # Preprocess features using pre-fitted preprocessor without refitting
        prep = self.preprocessor_data["preprocessor"]
        X_scaled = prep.transform(df_row)

        # Determine class indices dynamically from model.classes_
        classes = list(model.classes_) if hasattr(model, "classes_") else [0, 1]
        benign_index = classes.index(0) if 0 in classes else 0
        attack_index = classes.index(1) if 1 in classes else (1 if len(classes) > 1 else 0)

        # Run inference
        t0 = time.perf_counter()
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_scaled)[0]
            prob_benign = float(proba[benign_index])
            prob_attack = float(proba[attack_index]) if len(proba) > 1 else float(1.0 - prob_benign)
        else:
            pred_raw = int(model.predict(X_scaled)[0])
            prob_attack = 1.0 if pred_raw == 1 else 0.0
            prob_benign = 1.0 - prob_attack

        latency_ms = (time.perf_counter() - t0) * 1000.0

        # Sanity check probability ranges and sum
        assert 0.0 <= prob_benign <= 1.0001, f"Invalid benign probability: {prob_benign}"
        assert 0.0 <= prob_attack <= 1.0001, f"Invalid attack probability: {prob_attack}"
        total_p = prob_benign + prob_attack
        if total_p > 0:
            prob_benign = prob_benign / total_p
            prob_attack = prob_attack / total_p

        is_attack = prob_attack >= 0.5
        prediction_label = "ATTACK" if is_attack else "BENIGN"
        confidence = prob_attack if is_attack else prob_benign

        # Determine heuristic attack category if attack
        attack_type = None
        if is_attack:
            dest_port = float(full_feature_row.get("Destination Port", 0))
            syn_flag = float(full_feature_row.get("SYN Flag Count", 0))
            flow_bytes = float(full_feature_row.get("Flow Bytes/s", 0))
            flow_dur = float(full_feature_row.get("Flow Duration", 0))

            if dest_port == 21:
                attack_type = "FTP-Patator"
            elif dest_port == 22:
                attack_type = "SSH-Patator"
            elif dest_port in [80, 443, 8080] and flow_bytes > 100000:
                attack_type = "DoS Hulk"
            elif syn_flag == 1 and flow_dur < 1000:
                attack_type = "PortScan"
            elif flow_bytes > 150000:
                attack_type = "DDoS"
            else:
                attack_type = "Malicious Infiltration"

        # Generate SHAP feature attributions
        top_contributions = []
        try:
            explainer = self.get_explainer(model_name)
            if explainer:
                local_exp = explainer.explain_instance(
                    instance_1d=X_scaled[0],
                    raw_feature_values=full_feature_row,
                    top_k=6
                )
                target_drivers = local_exp["top_attack_drivers"] if is_attack else local_exp["top_benign_drivers"]
                top_contributions = target_drivers
        except Exception as e:
            print(f"[MLService] SHAP explanation warning: {e}")

        return {
            "prediction": prediction_label,
            "is_attack": is_attack,
            "confidence": round(confidence, 4),
            "probability_attack": round(prob_attack, 4),
            "probability_benign": round(prob_benign, 4),
            "model_used": model_name,
            "model_version": "1.0.0",
            "attack_type": attack_type,
            "latency_ms": round(latency_ms, 3),
            "top_contributing_features": top_contributions,
            "debug_info": {
                "expected_features_count": len(expected_features),
                "received_features_count": len(features),
                "model_classes": [str(c) for c in classes],
                "raw_benign_prob": float(proba[benign_index]) if hasattr(model, "predict_proba") else prob_benign,
                "raw_attack_prob": float(proba[attack_index]) if hasattr(model, "predict_proba") else prob_attack
            }
        }

    def predict_test_samples(
        self,
        count: int = 10,
        model_name: str = "optimized_xgboost"
    ) -> List[Dict[str, Any]]:
        """
        Execute predictions on real untouched test samples from X_test split.
        Returns actual vs predicted values, probabilities, and telemetry overview.
        """
        if not self.splits_data:
            self._load_splits_and_baseline()
        if not self.splits_data or "X_test_df_raw" not in self.splits_data:
            raise RuntimeError("Dataset test split is not loaded.")

        if not self.models:
            self._load_available_models()
        if model_name not in self.models:
            model_name = list(self.models.keys())[0] if self.models else "optimized_xgboost"

        model = self.models[model_name]
        prep = self.preprocessor_data["preprocessor"]

        X_test_df = self.splits_data["X_test_df_raw"]
        y_test = self.splits_data["y_test"]
        feature_names = self.get_feature_names()

        classes = list(model.classes_) if hasattr(model, "classes_") else [0, 1]
        benign_index = classes.index(0) if 0 in classes else 0
        attack_index = classes.index(1) if 1 in classes else (1 if len(classes) > 1 else 0)

        # Select a balanced sample of real test records
        benign_indices = np.where(y_test == 0)[0]
        attack_indices = np.where(y_test == 1)[0]
        
        half = max(1, count // 2)
        selected_idx = []
        for i in range(min(half, len(benign_indices))):
            selected_idx.append(int(benign_indices[i]))
        for i in range(min(count - len(selected_idx), len(attack_indices))):
            selected_idx.append(int(attack_indices[i]))

        results = []
        for idx in selected_idx:
            sample_dict = X_test_df.iloc[idx].to_dict()
            df_row = pd.DataFrame([{f: float(sample_dict.get(f, 0.0)) for f in feature_names}])
            X_sc = prep.transform(df_row)

            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X_sc)[0]
                prob_benign = float(proba[benign_index])
                prob_attack = float(proba[attack_index]) if len(proba) > 1 else float(1.0 - prob_benign)
            else:
                pred_raw = int(model.predict(X_sc)[0])
                prob_attack = 1.0 if pred_raw == 1 else 0.0
                prob_benign = 1.0 - prob_attack

            actual_label = "BENIGN" if y_test[idx] == 0 else "ATTACK"
            is_attack = prob_attack >= 0.5
            predicted_label = "ATTACK" if is_attack else "BENIGN"
            confidence = prob_attack if is_attack else prob_benign

            results.append({
                "sample_id": idx + 1,
                "actual": actual_label,
                "actual_numeric": int(y_test[idx]),
                "predicted": predicted_label,
                "is_attack": is_attack,
                "is_correct": (actual_label == predicted_label),
                "benign_probability": round(prob_benign, 4),
                "attack_probability": round(prob_attack, 4),
                "confidence": round(confidence, 4),
                "model_used": model_name,
                "key_telemetry": {
                    "destination_port": int(sample_dict.get("Destination Port", 0)),
                    "flow_duration_us": float(sample_dict.get("Flow Duration", 0)),
                    "total_fwd_packets": int(sample_dict.get("Total Fwd Packets", 0)),
                    "total_bwd_packets": int(sample_dict.get("Total Backward Packets", 0)),
                    "flow_bytes_s": round(float(sample_dict.get("Flow Bytes/s", 0)), 2),
                    "syn_flag": int(sample_dict.get("SYN Flag Count", 0)),
                    "ack_flag": int(sample_dict.get("ACK Flag Count", 0))
                },
                "features": {k: round(float(v), 4) for k, v in sample_dict.items() if k in feature_names}
            })

        return results

    def predict_batch_df(
        self,
        df: pd.DataFrame,
        model_name: str = "optimized_xgboost"
    ) -> Dict[str, Any]:
        """
        Process batch of network flows from uploaded CSV DataFrame.
        """
        if not self.models:
            self._load_available_models()
        if not self.preprocessor_data:
            self._load_preprocessor()

        if model_name not in self.models:
            model_name = list(self.models.keys())[0] if self.models else "optimized_xgboost"

        model = self.models[model_name]
        expected_features = self.get_feature_names()

        # Clean column names in uploaded dataframe
        df_clean = df.copy()
        df_clean.columns = df_clean.columns.str.strip().str.replace(r"\s+", " ", regex=True)

        # Build feature matrix aligning with expected columns
        for col in expected_features:
            if col not in df_clean.columns:
                df_clean[col] = self.benign_baseline.get(col, 0.0)

        X_df = df_clean[expected_features]
        prep = self.preprocessor_data["preprocessor"]
        X_scaled = prep.transform(X_df)

        classes = list(model.classes_) if hasattr(model, "classes_") else [0, 1]
        attack_index = classes.index(1) if 1 in classes else (1 if len(classes) > 1 else 0)

        t0 = time.perf_counter()
        if hasattr(model, "predict_proba"):
            probas = model.predict_proba(X_scaled)
            probs_attack = probas[:, attack_index] if probas.shape[1] > attack_index else (1.0 - probas[:, 0])
        else:
            preds = model.predict(X_scaled)
            probs_attack = np.where(preds == 1, 1.0, 0.0)

        proc_time = time.perf_counter() - t0

        is_attacks = probs_attack >= 0.5
        predictions_labels = np.where(is_attacks, "ATTACK", "BENIGN")
        confidences = np.where(is_attacks, probs_attack, 1.0 - probs_attack)

        total = len(df)
        attacks_count = int(np.sum(is_attacks))
        benign_count = total - attacks_count
        attack_pct = round((attacks_count / max(1, total)) * 100, 2)

        # Sample first 20 records for UI preview
        preview_records = []
        for i in range(min(20, total)):
            preview_records.append({
                "row_index": i + 1,
                "prediction": str(predictions_labels[i]),
                "confidence": round(float(confidences[i]), 4),
                "prob_attack": round(float(probs_attack[i]), 4),
                "dest_port": float(df_clean.iloc[i].get("Destination Port", 0)),
                "flow_duration": float(df_clean.iloc[i].get("Flow Duration", 0)),
                "flow_bytes_s": float(df_clean.iloc[i].get("Flow Bytes/s", 0))
            })

        return {
            "total_records": total,
            "attacks_detected": attacks_count,
            "benign_count": benign_count,
            "attack_percentage": attack_pct,
            "model_used": model_name,
            "processing_time_sec": round(proc_time, 4),
            "predictions_sample": preview_records,
            "class_breakdown": {
                "BENIGN": benign_count,
                "ATTACK": attacks_count
            }
        }


ml_service = MLService()
