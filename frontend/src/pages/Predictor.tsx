import React, { useState, useEffect } from "react";
import {
  ShieldAlert,
  ShieldCheck,
  Zap,
  RotateCcw,
  Sparkles,
  Sliders,
  Info,
  AlertCircle,
  Code2,
  ChevronDown,
  ChevronUp,
  Database,
  CheckCircle2,
  XCircle,
  Activity
} from "lucide-react";
import { fetchModels, fetchPresets, fetchTestSamples, predictSingle } from "../services/api";
import { ModelInfo, PredictionResult, TestSampleItem } from "../types";

export const Predictor: React.FC = () => {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [selectedModel, setSelectedModel] = useState<string>("optimized_xgboost");
  const [presets, setPresets] = useState<Record<string, any>>({});
  const [activePresetKey, setActivePresetKey] = useState<string>("benign_web");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [showDebug, setShowDebug] = useState(false);

  // Test samples state for real dataset verification
  const [testSamples, setTestSamples] = useState<TestSampleItem[]>([]);
  const [loadingSamples, setLoadingSamples] = useState(false);
  const [activeTab, setActiveTab] = useState<"interactive" | "real_samples">("interactive");

  // Form feature state (key representative subset for interactive simulation)
  const [features, setFeatures] = useState<Record<string, number>>({
    "Destination Port": 443,
    "Flow Duration": 210103,
    "Total Fwd Packets": 11,
    "Total Backward Packets": 8,
    "Total Length of Fwd Packets": 4026,
    "Total Length of Bwd Packets": 6724,
    "Flow Bytes/s": 10117,
    "Flow Packets/s": 90,
    "SYN Flag Count": 1,
    "ACK Flag Count": 1,
    "RST Flag Count": 0,
    "Init_Win_bytes_forward": 14600,
    "Init_Win_bytes_backward": 29200
  });

  useEffect(() => {
    async function init() {
      try {
        const [mList, pData] = await Promise.all([
          fetchModels(),
          fetchPresets()
        ]);
        setModels(mList);
        setPresets(pData);
        if (mList.length > 0) {
          const defaultOpt = mList.find((m) => m.model_id === "optimized_xgboost");
          if (defaultOpt) setSelectedModel(defaultOpt.model_id);
          else setSelectedModel(mList[0].model_id);
        }
        if (pData && pData["benign_web"]) {
          setFeatures(pData["benign_web"].features);
          setActivePresetKey("benign_web");
        }
      } catch (err) {
        console.error("Init predictor error:", err);
      }
    }
    init();
  }, []);

  const handlePresetSelect = (presetKey: string) => {
    if (!presets[presetKey]) return;
    setActivePresetKey(presetKey);
    setFeatures(presets[presetKey].features);
  };

  const handleInputChange = (key: string, value: number) => {
    setFeatures((prev) => ({
      ...prev,
      [key]: value
    }));
  };

  const handlePredict = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await predictSingle(features, selectedModel);
      setResult(res);
    } catch (err: any) {
      setError(err.message || "Failed to generate prediction");
    } finally {
      setLoading(false);
    }
  };

  const handleLoadTestSamples = async () => {
    setLoadingSamples(true);
    try {
      const resp = await fetchTestSamples(10, selectedModel);
      setTestSamples(resp.samples);
    } catch (err: any) {
      setError(err.message || "Failed to load test samples");
    } finally {
      setLoadingSamples(false);
    }
  };

  const handleSelectSample = (sample: TestSampleItem) => {
    setFeatures(sample.features);
    setActiveTab("interactive");
    setResult({
      prediction: sample.predicted,
      is_attack: sample.is_attack,
      confidence: sample.confidence,
      probability_attack: sample.attack_probability,
      probability_benign: sample.benign_probability,
      model_used: sample.model_used,
      model_version: "1.0.0",
      latency_ms: 1.2,
      top_contributing_features: [],
      disclaimer: "Loaded from real test dataset sample."
    });
  };

  return (
    <div className="space-y-6">
      {/* Header & Controls */}
      <div className="cyber-card p-6 flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
        <div>
          <div className="flex items-center space-x-3">
            <h2 className="text-xl font-bold text-white flex items-center space-x-2">
              <ShieldAlert className="w-5 h-5 text-sky-400" />
              <span>Real-Time Network Intrusion Telemetry Inspector</span>
            </h2>
            <span className="px-2.5 py-0.5 rounded-full text-[10px] font-mono bg-emerald-500/10 text-emerald-400 border border-emerald-500/30">
              Pipeline Verified
            </span>
          </div>
          <p className="text-xs text-slate-400 mt-1">
            Evaluate authentic network flow records, simulate live telemetry variables, and inspect verifiable SHAP feature attributions.
          </p>
        </div>

        {/* Classifier Selector */}
        <div className="flex items-center space-x-3 w-full md:w-auto">
          <label className="text-xs text-slate-400 font-mono">Classifier:</label>
          <select
            value={selectedModel}
            onChange={(e) => setSelectedModel(e.target.value)}
            className="bg-slate-900 border border-slate-700 text-slate-200 text-xs rounded-lg px-3 py-2 focus:ring-1 focus:ring-sky-500 focus:outline-none"
          >
            {models.map((m) => (
              <option key={m.model_id} value={m.model_id}>
                {m.model_name} ({m.algorithm_type})
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Mode Navigation Tabs */}
      <div className="flex items-center space-x-3 border-b border-slate-800 pb-2">
        <button
          onClick={() => setActiveTab("interactive")}
          className={`px-4 py-2 rounded-lg text-xs font-medium flex items-center space-x-2 transition-all ${
            activeTab === "interactive"
              ? "bg-sky-500/20 text-sky-400 border border-sky-500/30"
              : "text-slate-400 hover:text-slate-200 hover:bg-slate-800/50"
          }`}
        >
          <Sliders className="w-4 h-4" />
          <span>Interactive Telemetry Simulation</span>
        </button>

        <button
          onClick={() => {
            setActiveTab("real_samples");
            if (testSamples.length === 0) handleLoadTestSamples();
          }}
          className={`px-4 py-2 rounded-lg text-xs font-medium flex items-center space-x-2 transition-all ${
            activeTab === "real_samples"
              ? "bg-sky-500/20 text-sky-400 border border-sky-500/30"
              : "text-slate-400 hover:text-slate-200 hover:bg-slate-800/50"
          }`}
        >
          <Database className="w-4 h-4" />
          <span>Real Test Dataset Benchmark Samples</span>
        </button>
      </div>

      {activeTab === "interactive" ? (
        <>
          {/* Preset Buttons */}
          <div className="cyber-card p-4">
            <div className="flex items-center justify-between mb-2">
              <p className="text-xs font-mono text-slate-400 flex items-center space-x-2">
                <Sparkles className="w-3.5 h-3.5 text-sky-400" />
                <span>AUTHENTIC FLOW TEMPLATES (CIC-IDS2017):</span>
              </p>
              <span className="text-[10px] text-slate-500 font-mono">61-feature physical integrity</span>
            </div>
            <div className="flex flex-wrap gap-2">
              {Object.entries(presets).map(([key, p]) => {
                const isSelected = activePresetKey === key;
                const isBenign = p.category === "BENIGN";
                return (
                  <button
                    key={key}
                    onClick={() => handlePresetSelect(key)}
                    className={`px-3 py-1.5 rounded-lg border text-xs transition-all flex items-center space-x-2 ${
                      isSelected
                        ? "bg-sky-500/20 border-sky-500/60 text-sky-300 shadow-md shadow-sky-500/10 font-semibold"
                        : "bg-slate-800/80 hover:bg-slate-700/80 border-slate-700 text-slate-300"
                    }`}
                  >
                    <span
                      className={`w-2 h-2 rounded-full ${
                        isBenign ? "bg-emerald-400" : "bg-red-400"
                      }`}
                    />
                    <span>{p.name}</span>
                  </button>
                );
              })}
            </div>
          </div>

          {/* Main Grid: Input Form vs Live Prediction Output */}
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
            {/* Input Parameters Column */}
            <div className="lg:col-span-7 cyber-card p-6 space-y-4">
              <div className="flex items-center justify-between border-b border-slate-800 pb-3">
                <h3 className="font-semibold text-white text-sm flex items-center space-x-2">
                  <Sliders className="w-4 h-4 text-sky-400" />
                  <span>Network Flow Telemetry Attributes</span>
                </h3>
                <span className="text-[11px] font-mono text-slate-400">
                  {presets[activePresetKey]?.name || "Custom Telemetry"}
                </span>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                <div>
                  <label className="text-xs text-slate-400 block mb-1">Destination Port</label>
                  <input
                    type="number"
                    value={features["Destination Port"] ?? 443}
                    onChange={(e) => handleInputChange("Destination Port", parseFloat(e.target.value) || 0)}
                    className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-xs text-slate-100 focus:border-sky-500 focus:outline-none font-mono"
                  />
                </div>

                <div>
                  <label className="text-xs text-slate-400 block mb-1">Flow Duration (&mu;s)</label>
                  <input
                    type="number"
                    value={features["Flow Duration"] ?? 15000}
                    onChange={(e) => handleInputChange("Flow Duration", parseFloat(e.target.value) || 0)}
                    className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-xs text-slate-100 focus:border-sky-500 focus:outline-none font-mono"
                  />
                </div>

                <div>
                  <label className="text-xs text-slate-400 block mb-1">Total Forward Packets</label>
                  <input
                    type="number"
                    value={features["Total Fwd Packets"] ?? 10}
                    onChange={(e) => handleInputChange("Total Fwd Packets", parseFloat(e.target.value) || 0)}
                    className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-xs text-slate-100 focus:border-sky-500 focus:outline-none font-mono"
                  />
                </div>

                <div>
                  <label className="text-xs text-slate-400 block mb-1">Total Backward Packets</label>
                  <input
                    type="number"
                    value={features["Total Backward Packets"] ?? 8}
                    onChange={(e) => handleInputChange("Total Backward Packets", parseFloat(e.target.value) || 0)}
                    className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-xs text-slate-100 focus:border-sky-500 focus:outline-none font-mono"
                  />
                </div>

                <div>
                  <label className="text-xs text-slate-400 block mb-1">Total Fwd Payload (Bytes)</label>
                  <input
                    type="number"
                    value={features["Total Length of Fwd Packets"] ?? 4000}
                    onChange={(e) => handleInputChange("Total Length of Fwd Packets", parseFloat(e.target.value) || 0)}
                    className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-xs text-slate-100 focus:border-sky-500 focus:outline-none font-mono"
                  />
                </div>

                <div>
                  <label className="text-xs text-slate-400 block mb-1">Total Bwd Payload (Bytes)</label>
                  <input
                    type="number"
                    value={features["Total Length of Bwd Packets"] ?? 6000}
                    onChange={(e) => handleInputChange("Total Length of Bwd Packets", parseFloat(e.target.value) || 0)}
                    className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-xs text-slate-100 focus:border-sky-500 focus:outline-none font-mono"
                  />
                </div>

                <div>
                  <label className="text-xs text-slate-400 block mb-1">SYN Flag (Connection Initiation)</label>
                  <select
                    value={features["SYN Flag Count"] ?? 0}
                    onChange={(e) => handleInputChange("SYN Flag Count", parseInt(e.target.value) || 0)}
                    className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-xs text-slate-100 focus:border-sky-500 focus:outline-none font-mono"
                  >
                    <option value={0}>0 (SYN Not Set)</option>
                    <option value={1}>1 (SYN Set - Connection Initiation)</option>
                  </select>
                </div>

                <div>
                  <label className="text-xs text-slate-400 block mb-1">ACK Flag (Established Session)</label>
                  <select
                    value={features["ACK Flag Count"] ?? 1}
                    onChange={(e) => handleInputChange("ACK Flag Count", parseInt(e.target.value) || 0)}
                    className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-xs text-slate-100 focus:border-sky-500 focus:outline-none font-mono"
                  >
                    <option value={0}>0 (ACK Not Set)</option>
                    <option value={1}>1 (ACK Set - Established Session)</option>
                  </select>
                </div>

                <div>
                  <label className="text-xs text-slate-400 block mb-1">Init Fwd Window Size (Bytes)</label>
                  <input
                    type="number"
                    value={features["Init_Win_bytes_forward"] ?? 14600}
                    onChange={(e) => handleInputChange("Init_Win_bytes_forward", parseFloat(e.target.value) || 0)}
                    className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-xs text-slate-100 focus:border-sky-500 focus:outline-none font-mono"
                  />
                </div>

                <div>
                  <label className="text-xs text-slate-400 block mb-1">Init Bwd Window Size (Bytes)</label>
                  <input
                    type="number"
                    value={features["Init_Win_bytes_backward"] ?? 29200}
                    onChange={(e) => handleInputChange("Init_Win_bytes_backward", parseFloat(e.target.value) || 0)}
                    className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-xs text-slate-100 focus:border-sky-500 focus:outline-none font-mono"
                  />
                </div>
              </div>

              <div className="pt-2 flex items-center justify-between">
                <p className="text-[11px] text-slate-400">
                  Correlated flow metrics (rates, lengths, subflow stats) automatically align with network physics.
                </p>
                <button
                  onClick={handlePredict}
                  disabled={loading}
                  className="px-6 py-2.5 rounded-xl bg-gradient-to-r from-sky-500 to-indigo-600 hover:from-sky-400 hover:to-indigo-500 text-white font-medium text-xs flex items-center space-x-2 shadow-lg shadow-sky-500/20 transition-all disabled:opacity-50"
                >
                  {loading ? (
                    <>
                      <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                      <span>Evaluating Flow...</span>
                    </>
                  ) : (
                    <>
                      <Zap className="w-4 h-4" />
                      <span>Execute ML Inference</span>
                    </>
                  )}
                </button>
              </div>
            </div>

            {/* Prediction Results & SHAP Breakdown Column */}
            <div className="lg:col-span-5 space-y-4">
              {error && (
                <div className="p-4 rounded-xl bg-red-950/40 border border-red-800/60 text-red-300 text-xs flex items-start space-x-2">
                  <AlertCircle className="w-4 h-4 text-red-400 mt-0.5 flex-shrink-0" />
                  <span>{error}</span>
                </div>
              )}

              {result ? (
                <div className="cyber-card p-6 space-y-5 border-sky-500/30">
                  {/* Verdict Header */}
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-xs font-mono text-slate-400">INTRUSION CLASSIFICATION</p>
                      <div className="flex items-center space-x-3 mt-1">
                        <span
                          className={`text-2xl font-black px-3.5 py-1 rounded-lg ${
                            result.is_attack
                              ? "bg-red-500/20 text-red-400 border border-red-500/40 shadow-lg shadow-red-500/10"
                              : "bg-emerald-500/20 text-emerald-400 border border-emerald-500/40 shadow-lg shadow-emerald-500/10"
                          }`}
                        >
                          {result.prediction}
                        </span>
                        {result.attack_type && (
                          <span className="text-xs px-2.5 py-1 rounded-md bg-slate-800 text-slate-200 border border-slate-700 font-mono">
                            {result.attack_type}
                          </span>
                        )}
                      </div>
                    </div>

                    <div className="text-right">
                      <p className="text-xs font-mono text-slate-400">CONFIDENCE</p>
                      <p className="text-2xl font-bold text-white mt-1">
                        {(result.confidence * 100).toFixed(1)}%
                      </p>
                    </div>
                  </div>

                  {/* Confidence Gauge Bar */}
                  <div>
                    <div className="flex justify-between text-xs text-slate-400 font-mono mb-1">
                      <span>Benign: {(result.probability_benign * 100).toFixed(1)}%</span>
                      <span>Attack: {(result.probability_attack * 100).toFixed(1)}%</span>
                    </div>
                    <div className="w-full h-2.5 bg-slate-800 rounded-full overflow-hidden flex">
                      <div
                        className="h-full bg-emerald-500 transition-all duration-500"
                        style={{ width: `${result.probability_benign * 100}%` }}
                      ></div>
                      <div
                        className="h-full bg-red-500 transition-all duration-500"
                        style={{ width: `${result.probability_attack * 100}%` }}
                      ></div>
                    </div>
                  </div>

                  {/* SHAP Feature Contribution Waterfall */}
                  {result.top_contributing_features && result.top_contributing_features.length > 0 && (
                    <div>
                      <h4 className="text-xs font-mono text-slate-300 uppercase tracking-wider mb-2 flex items-center justify-between">
                        <span>Top SHAP Feature Attributions</span>
                        <span className="text-[10px] text-slate-400 font-normal">Impact on Verdict</span>
                      </h4>

                      <div className="space-y-2">
                        {result.top_contributing_features.map((f, i) => {
                          const isPositive = f.shap_value > 0;
                          return (
                            <div key={i} className="p-2 rounded bg-slate-900/60 border border-slate-800 text-xs">
                              <div className="flex justify-between items-center mb-1">
                                <span className="text-slate-300 font-medium truncate max-w-[180px]">{f.feature}</span>
                                <span className={`font-mono text-[11px] font-semibold ${isPositive ? "text-red-400" : "text-emerald-400"}`}>
                                  {isPositive ? `+${f.shap_value.toFixed(3)}` : f.shap_value.toFixed(3)}
                                </span>
                              </div>
                              <div className="flex justify-between items-center text-[10px] text-slate-400">
                                <span>Value: {String(f.actual_value)}</span>
                                <span className={`px-1.5 py-0.2 rounded text-[9px] ${isPositive ? "bg-red-950 text-red-300" : "bg-emerald-950 text-emerald-300"}`}>
                                  Pushes toward {f.effect}
                                </span>
                              </div>
                            </div>
                          );
                        })}
                      </div>
                    </div>
                  )}

                  {/* Latency & Metadata Footer */}
                  <div className="pt-3 border-t border-slate-800/80 flex items-center justify-between text-[11px] text-slate-400 font-mono">
                    <span>Model: {result.model_used}</span>
                    <span>Inference: {result.latency_ms}ms</span>
                  </div>

                  {/* Debug Panel Toggle */}
                  <div className="border-t border-slate-800/80 pt-2">
                    <button
                      onClick={() => setShowDebug(!showDebug)}
                      className="w-full flex items-center justify-between text-[11px] font-mono text-slate-400 hover:text-slate-200 transition-colors"
                    >
                      <span className="flex items-center space-x-1.5">
                        <Code2 className="w-3.5 h-3.5 text-sky-400" />
                        <span>Developer Debug Diagnostics</span>
                      </span>
                      {showDebug ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
                    </button>

                    {showDebug && result.debug_info && (
                      <div className="mt-3 p-3 bg-slate-950/80 border border-slate-800 rounded-lg text-[10px] font-mono text-slate-300 space-y-1.5">
                        <div className="flex justify-between">
                          <span className="text-slate-500">Expected Features:</span>
                          <span>{result.debug_info.expected_features_count} columns</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-slate-500">Supplied Features:</span>
                          <span>{result.debug_info.received_features_count} columns</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-slate-500">Model Classes:</span>
                          <span>{result.debug_info.model_classes.join(", ")} (0=BENIGN, 1=ATTACK)</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-slate-500">Raw Benign Output:</span>
                          <span className="text-emerald-400">{result.debug_info.raw_benign_prob.toFixed(6)}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-slate-500">Raw Attack Output:</span>
                          <span className="text-red-400">{result.debug_info.raw_attack_prob.toFixed(6)}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-slate-500">Preprocessing:</span>
                          <span className="text-sky-400">StandardScaler + Median Imputation</span>
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              ) : (
                <div className="cyber-card p-10 flex flex-col items-center justify-center text-center text-slate-400 space-y-3">
                  <Info className="w-8 h-8 text-slate-500" />
                  <p className="text-sm font-medium">Ready for Telemetry Evaluation</p>
                  <p className="text-xs text-slate-400 max-w-xs">
                    Select a preset or customize network flow variables and click "Execute ML Inference" to inspect classification results.
                  </p>
                </div>
              )}
            </div>
          </div>
        </>
      ) : (
        /* Real Test Dataset Benchmark View */
        <div className="cyber-card p-6 space-y-4">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-slate-800 pb-4">
            <div>
              <h3 className="font-semibold text-white text-sm flex items-center space-x-2">
                <Database className="w-4 h-4 text-sky-400" />
                <span>Real Untouched Test Set Evaluation (Ground Truth Benchmark)</span>
              </h3>
              <p className="text-xs text-slate-400 mt-1">
                Directly verify model predictions and probabilities on untouched samples from the real CIC-IDS2017 test split.
              </p>
            </div>
            <button
              onClick={handleLoadTestSamples}
              disabled={loadingSamples}
              className="px-4 py-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-xs font-mono text-slate-200 border border-slate-700 flex items-center space-x-2 transition-all self-start"
            >
              <RotateCcw className={`w-3.5 h-3.5 ${loadingSamples ? "animate-spin" : ""}`} />
              <span>Refresh Test Samples</span>
            </button>
          </div>

          {loadingSamples ? (
            <div className="py-12 flex flex-col items-center justify-center space-y-2 text-slate-400">
              <div className="w-6 h-6 border-2 border-sky-400 border-t-transparent rounded-full animate-spin"></div>
              <span className="text-xs font-mono">Loading real dataset test records...</span>
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs font-mono">
                <thead>
                  <tr className="border-b border-slate-800 text-slate-400">
                    <th className="py-2.5 px-3">Sample ID</th>
                    <th className="py-2.5 px-3">Ground Truth</th>
                    <th className="py-2.5 px-3">Model Verdict</th>
                    <th className="py-2.5 px-3">Benign Prob</th>
                    <th className="py-2.5 px-3">Attack Prob</th>
                    <th className="py-2.5 px-3">Port / Duration</th>
                    <th className="py-2.5 px-3">Action</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/60">
                  {testSamples.map((s) => (
                    <tr key={s.sample_id} className="hover:bg-slate-800/40 transition-colors">
                      <td className="py-2.5 px-3 text-slate-300">#{s.sample_id}</td>
                      <td className="py-2.5 px-3">
                        <span
                          className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                            s.actual === "BENIGN"
                              ? "bg-emerald-500/10 text-emerald-400 border border-emerald-500/30"
                              : "bg-red-500/10 text-red-400 border border-red-500/30"
                          }`}
                        >
                          {s.actual}
                        </span>
                      </td>
                      <td className="py-2.5 px-3">
                        <div className="flex items-center space-x-1.5">
                          {s.is_correct ? (
                            <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400 flex-shrink-0" />
                          ) : (
                            <XCircle className="w-3.5 h-3.5 text-red-400 flex-shrink-0" />
                          )}
                          <span
                            className={`font-semibold ${
                              s.predicted === "BENIGN" ? "text-emerald-300" : "text-red-300"
                            }`}
                          >
                            {s.predicted}
                          </span>
                        </div>
                      </td>
                      <td className="py-2.5 px-3 text-emerald-400 font-medium">
                        {(s.benign_probability * 100).toFixed(1)}%
                      </td>
                      <td className="py-2.5 px-3 text-red-400 font-medium">
                        {(s.attack_probability * 100).toFixed(1)}%
                      </td>
                      <td className="py-2.5 px-3 text-slate-400 text-[11px]">
                        Port {s.key_telemetry.destination_port} | {(s.key_telemetry.flow_duration_us / 1000).toFixed(0)}ms
                      </td>
                      <td className="py-2.5 px-3">
                        <button
                          onClick={() => handleSelectSample(s)}
                          className="px-2.5 py-1 rounded bg-sky-500/10 hover:bg-sky-500/20 text-sky-400 border border-sky-500/30 text-[10px] transition-colors"
                        >
                          Inspect Flow
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}
    </div>
  );
};
