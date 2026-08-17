export interface FeatureAttribution {
  feature: string;
  shap_value: number;
  actual_value: string | number;
  effect: "ATTACK" | "BENIGN";
}

export interface PredictionResult {
  prediction: "ATTACK" | "BENIGN";
  is_attack: boolean;
  confidence: number;
  probability_attack: number;
  probability_benign: number;
  model_used: string;
  model_version: string;
  attack_type?: string;
  latency_ms: number;
  top_contributing_features: FeatureAttribution[];
  debug_info?: {
    expected_features_count: number;
    received_features_count: number;
    model_classes: string[];
    raw_benign_prob: number;
    raw_attack_prob: number;
  };
  disclaimer: string;
}

export interface TestSampleItem {
  sample_id: number;
  actual: "ATTACK" | "BENIGN";
  actual_numeric: number;
  predicted: "ATTACK" | "BENIGN";
  is_attack: boolean;
  is_correct: boolean;
  benign_probability: number;
  attack_probability: number;
  confidence: number;
  model_used: string;
  key_telemetry: {
    destination_port: number;
    flow_duration_us: number;
    total_fwd_packets: number;
    total_bwd_packets: number;
    flow_bytes_s: number;
    syn_flag: number;
    ack_flag: number;
  };
  features: Record<string, number>;
}

export interface TestSamplesResponse {
  total_samples: number;
  model_used: string;
  samples: TestSampleItem[];
}

export interface BatchResultItem {
  row_index: number;
  prediction: "ATTACK" | "BENIGN";
  confidence: number;
  prob_attack: number;
  dest_port: number;
  flow_duration: number;
  flow_bytes_s: number;
}

export interface BatchPredictionSummary {
  total_records: number;
  attacks_detected: number;
  benign_count: number;
  attack_percentage: number;
  model_used: string;
  processing_time_sec: number;
  predictions_sample: BatchResultItem[];
  class_breakdown: {
    BENIGN: number;
    ATTACK: number;
  };
}

export interface ModelInfo {
  model_id: string;
  model_name: string;
  algorithm_type: string;
  accuracy: number;
  precision: number;
  recall: number;
  f1_score: number;
  roc_auc: number;
  false_positive_rate: number;
  false_negative_rate: number;
  training_time_sec: number;
  inference_time_ms: number;
  is_synthetic: boolean;
}

export interface ConfusionMatrix {
  tp: number;
  tn: number;
  fp: number;
  fn: number;
}

export interface CurvePoint {
  fpr?: number;
  tpr?: number;
  recall?: number;
  precision?: number;
}

export interface ModelEvaluationDetail {
  model_name: string;
  accuracy: number;
  precision: number;
  recall: number;
  f1: number;
  roc_auc: number;
  pr_auc: number;
  specificity: number;
  false_positive_rate: number;
  false_negative_rate: number;
  confusion_matrix: ConfusionMatrix;
  roc_curve: { fpr: number; tpr: number }[];
  pr_curve: { recall: number; precision: number }[];
  training_time_sec: number;
  inference_time_ms_per_1k: number;
  num_test_samples: number;
}

export interface MasterComparison {
  is_synthetic: boolean;
  dataset_name: string;
  timestamp: string;
  models: Record<string, ModelEvaluationDetail>;
  statistical_stability: Record<string, {
    f1_mean: number;
    f1_std: number;
    recall_mean: number;
    recall_std: number;
    accuracy_mean: number;
    accuracy_std: number;
    fpr_mean: number;
    fpr_std: number;
  }>;
}

export interface ResearchConclusions {
  is_synthetic_dataset: boolean;
  rq1_individual_baselines: {
    question: string;
    best_model: string;
    f1_score: number;
    recall: number;
    accuracy: number;
    finding: string;
  };
  rq2_feature_selection: {
    question: string;
    finding: string;
  };
  rq3_optimization_impact: {
    question: string;
    best_optimized: string;
    baseline_f1: number;
    optimized_f1: number;
    gain_percentage: string;
    finding: string;
  };
  rq4_ensemble_superiority: {
    question: string;
    best_ensemble: string;
    ensemble_f1: number;
    optimized_f1: number;
    gain_over_optimized_pct: string;
    finding: string;
  };
  rq5_optimal_tradeoff: {
    question: string;
    best_tradeoff_model: string;
    highest_recall_model: string;
    highest_recall_value: number;
    finding: string;
  };
  rq6_explainability: {
    question: string;
    top_global_features: string[];
    finding: string;
  };
  executive_summary: string;
}

export interface DatasetSummary {
  dataset_info: {
    source_file: string;
    is_synthetic: boolean;
    total_rows_loaded: number;
    original_rows: number;
    num_features: number;
    label_column: string;
    class_distribution: Record<string, number>;
    memory_usage_mb: number;
  };
  cleaning_audit: {
    initial_shape: [number, number];
    final_shape: [number, number];
    duplicates_removed: number;
    nan_inf_cells_found: number;
    constant_columns_dropped: string[];
  };
  split_info: {
    train_count: number;
    val_count: number;
    test_count: number;
    train_attack_ratio: number;
    val_attack_ratio: number;
    test_attack_ratio: number;
  };
  features: string[];
  num_features: number;
  classes: string[];
}
