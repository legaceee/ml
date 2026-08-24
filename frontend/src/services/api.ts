import {
  BatchPredictionSummary,
  DatasetSummary,
  MasterComparison,
  ModelInfo,
  PredictionResult,
  ResearchConclusions,
  TestSamplesResponse
} from "../types";

const API_BASE = "/api";

export async function fetchHealth(): Promise<{ status: string; models_loaded_count: number; is_dataset_synthetic: boolean }> {
  const res = await fetch(`${API_BASE}/health`);
  if (!res.ok) throw new Error("Health check failed");
  return res.json();
}

export async function predictSingle(features: Record<string, number>, modelName: string = "optimized_xgboost"): Promise<PredictionResult> {
  const res = await fetch(`${API_BASE}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model_name: modelName, features })
  });
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.detail || "Prediction failed");
  }
  return res.json();
}

export async function fetchTestSamples(count: number = 10, modelName: string = "optimized_xgboost"): Promise<TestSamplesResponse> {
  const res = await fetch(`${API_BASE}/predict/test-samples?count=${count}&model_name=${modelName}`);
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.detail || "Failed to load real test samples");
  }
  return res.json();
}

export async function predictBatchCSV(file: File, modelName: string = "optimized_xgboost"): Promise<BatchPredictionSummary> {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("model_name", modelName);

  const res = await fetch(`${API_BASE}/predict/batch`, {
    method: "POST",
    body: formData
  });
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.detail || "Batch prediction failed");
  }
  return res.json();
}

export async function fetchModels(): Promise<ModelInfo[]> {
  const res = await fetch(`${API_BASE}/models`);
  if (!res.ok) throw new Error("Failed to load models");
  return res.json();
}

export async function fetchMasterMetrics(): Promise<MasterComparison> {
  const res = await fetch(`${API_BASE}/metrics`);
  if (!res.ok) throw new Error("Failed to load metrics");
  return res.json();
}

export async function fetchResearchConclusions(): Promise<ResearchConclusions> {
  const res = await fetch(`${API_BASE}/experiments/conclusions`);
  if (!res.ok) throw new Error("Failed to load research conclusions");
  return res.json();
}

export async function fetchExperimentRecords(): Promise<{ records: any[] }> {
  const res = await fetch(`${API_BASE}/experiments`);
  if (!res.ok) throw new Error("Failed to load experiment records");
  return res.json();
}

export async function fetchOptimizationStudy(): Promise<any> {
  const res = await fetch(`${API_BASE}/experiments/optimization-study`);
  if (!res.ok) throw new Error("Failed to load optimization study");
  return res.json();
}

export async function fetchFeatureSelection(): Promise<any> {
  const res = await fetch(`${API_BASE}/experiments/feature-selection`);
  if (!res.ok) throw new Error("Failed to load feature selection data");
  return res.json();
}

export async function fetchGlobalExplain(modelName: string): Promise<any> {
  const res = await fetch(`${API_BASE}/explain/global/${modelName}`);
  if (!res.ok) throw new Error("Failed to load global SHAP values");
  return res.json();
}

export async function fetchDatasetSummary(): Promise<DatasetSummary> {
  const res = await fetch(`${API_BASE}/dataset/summary`);
  if (!res.ok) throw new Error("Failed to load dataset summary");
  return res.json();
}

export async function fetchPresets(): Promise<Record<string, any>> {
  const res = await fetch(`${API_BASE}/dataset/presets`);
  if (!res.ok) throw new Error("Failed to load presets");
  return res.json();
}

export async function fetchArtifact(name: "cv-results" | "imbalance-study" | "per-class-recall" | "search-comparison" | "shap-global" | "pca-summary"): Promise<any> {
  const res = await fetch(`${API_BASE}/experiments/artifact/${name}`);
  if (!res.ok) throw new Error(`Failed to fetch artifact ${name}`);
  return res.json();
}
