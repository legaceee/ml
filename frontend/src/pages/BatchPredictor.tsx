import React, { useState, useEffect } from "react";
import {
  UploadCloud,
  FileText,
  CheckCircle2,
  AlertTriangle,
  Download,
  Filter,
  Layers,
  Zap
} from "lucide-react";
import { fetchModels, predictBatchCSV } from "../services/api";
import { BatchPredictionSummary, ModelInfo } from "../types";

export const BatchPredictor: React.FC = () => {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [selectedModel, setSelectedModel] = useState<string>("optimized_xgboost");
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [summary, setSummary] = useState<BatchPredictionSummary | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [filterType, setFilterType] = useState<"ALL" | "ATTACK" | "BENIGN">("ALL");

  useEffect(() => {
    async function loadModelsList() {
      try {
        const m = await fetchModels();
        setModels(m);
        if (m.length > 0) setSelectedModel(m[0].model_id);
      } catch (err) {
        console.error("Batch models load error:", err);
      }
    }
    loadModelsList();
  }, []);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
      setSummary(null);
      setError(null);
    }
  };

  const handleUploadAndScore = async () => {
    if (!file) {
      setError("Please choose a CSV dataset file first.");
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const res = await predictBatchCSV(file, selectedModel);
      setSummary(res);
    } catch (err: any) {
      setError(err.message || "Failed to process batch CSV");
    } finally {
      setLoading(false);
    }
  };

  const downloadResultsCSV = () => {
    if (!summary) return;
    const headers = "Row,Destination Port,Flow Duration,Flow Bytes/s,Prediction,Confidence,Probability Attack\n";
    const rows = summary.predictions_sample
      .map(
        (r) =>
          `${r.row_index},${r.dest_port},${r.flow_duration},${r.flow_bytes_s},${r.prediction},${r.confidence},${r.prob_attack}`
      )
      .join("\n");
    const blob = new Blob([headers + rows], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.setAttribute("download", `scored_threat_results_${summary.model_used}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const filteredRows = summary
    ? summary.predictions_sample.filter((r) => {
        if (filterType === "ALL") return true;
        return r.prediction === filterType;
      })
    : [];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="cyber-card p-6 flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
        <div>
          <h2 className="text-xl font-bold text-white flex items-center space-x-2">
            <UploadCloud className="w-5 h-5 text-indigo-400" />
            <span>Batch Network Telemetry Scoring</span>
          </h2>
          <p className="text-xs text-slate-400 mt-1">
            Upload CSV captures of network flows to perform bulk threat classification, anomaly filtering, and forensic log export.
          </p>
        </div>

        <div className="flex items-center space-x-3 w-full md:w-auto">
          <label className="text-xs text-slate-400 font-mono">Classifier:</label>
          <select
            value={selectedModel}
            onChange={(e) => setSelectedModel(e.target.value)}
            className="bg-slate-900 border border-slate-700 text-slate-200 text-xs rounded-lg px-3 py-2 focus:ring-1 focus:ring-sky-500 focus:outline-none"
          >
            {models.map((m) => (
              <option key={m.model_id} value={m.model_id}>
                {m.model_name}
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Upload Zone */}
      <div className="cyber-card p-8 border-dashed border-2 border-slate-700/80 hover:border-indigo-500/60 transition-all flex flex-col items-center justify-center text-center space-y-4">
        <div className="p-4 rounded-full bg-indigo-500/10 text-indigo-400 border border-indigo-500/20">
          <FileText className="w-8 h-8" />
        </div>

        <div>
          <label className="cursor-pointer">
            <span className="px-4 py-2 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-semibold shadow-lg shadow-indigo-600/20 transition-all inline-block">
              Choose CSV File
            </span>
            <input type="file" accept=".csv" onChange={handleFileChange} className="hidden" />
          </label>
          {file ? (
            <p className="text-xs font-mono text-indigo-300 mt-2">
              Selected: <span className="font-semibold">{file.name}</span> ({(file.size / 1024).toFixed(1)} KB)
            </p>
          ) : (
            <p className="text-xs text-slate-400 mt-2">Supports CSV files matching CIC-IDS2017 flow schema</p>
          )}
        </div>

        {file && (
          <button
            onClick={handleUploadAndScore}
            disabled={loading}
            className="px-6 py-2.5 rounded-xl bg-gradient-to-r from-emerald-500 to-teal-600 hover:from-emerald-400 hover:to-teal-500 text-white font-medium text-xs flex items-center space-x-2 shadow-lg shadow-emerald-500/20 transition-all disabled:opacity-50"
          >
            {loading ? (
              <>
                <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                <span>Scoring Telemetry Stream...</span>
              </>
            ) : (
              <>
                <Zap className="w-4 h-4" />
                <span>Start Batch Analysis</span>
              </>
            )}
          </button>
        )}
      </div>

      {error && (
        <div className="p-4 rounded-xl bg-red-950/40 border border-red-800/60 text-red-300 text-xs flex items-start space-x-2">
          <AlertTriangle className="w-4 h-4 text-red-400 mt-0.5 flex-shrink-0" />
          <span>{error}</span>
        </div>
      )}

      {/* Batch Summary Report */}
      {summary && (
        <div className="space-y-6">
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            <div className="cyber-card p-5">
              <p className="text-xs font-mono text-slate-400">TOTAL FLOWS PROCESSED</p>
              <h3 className="text-2xl font-bold text-white mt-1">{summary.total_records.toLocaleString()}</h3>
              <p className="text-[11px] text-slate-400 mt-1">Processing time: {summary.processing_time_sec.toFixed(3)}s</p>
            </div>

            <div className="cyber-card p-5">
              <p className="text-xs font-mono text-slate-400">DETECTED ATTACKS</p>
              <h3 className="text-2xl font-bold text-red-400 mt-1">{summary.attacks_detected.toLocaleString()}</h3>
              <p className="text-[11px] text-red-400/80 mt-1">{summary.attack_percentage}% of total stream</p>
            </div>

            <div className="cyber-card p-5">
              <p className="text-xs font-mono text-slate-400">BENIGN TRAFFIC</p>
              <h3 className="text-2xl font-bold text-emerald-400 mt-1">{summary.benign_count.toLocaleString()}</h3>
              <p className="text-[11px] text-emerald-400/80 mt-1">Normal authenticated sessions</p>
            </div>

            <div className="cyber-card p-5 flex flex-col justify-between">
              <div>
                <p className="text-xs font-mono text-slate-400">EXPORT SCORED LOGS</p>
                <p className="text-[11px] text-slate-400 mt-1">Download enriched CSV results</p>
              </div>
              <button
                onClick={downloadResultsCSV}
                className="w-full py-2 rounded-lg bg-sky-500/15 hover:bg-sky-500/25 border border-sky-500/40 text-sky-300 text-xs font-semibold flex items-center justify-center space-x-1.5 transition-all mt-2"
              >
                <Download className="w-3.5 h-3.5" />
                <span>Export Scored CSV</span>
              </button>
            </div>
          </div>

          {/* Results Table */}
          <div className="cyber-card p-6 space-y-4">
            <div className="flex items-center justify-between">
              <h4 className="font-semibold text-white text-sm flex items-center space-x-2">
                <FileText className="w-4 h-4 text-sky-400" />
                <span>Flow Telemetry Inspection Sample (Top 20 Records)</span>
              </h4>

              {/* Filter Tabs */}
              <div className="flex items-center space-x-1 bg-slate-900 p-1 rounded-lg border border-slate-800 text-xs">
                <button
                  onClick={() => setFilterType("ALL")}
                  className={`px-2.5 py-1 rounded-md font-medium transition-all ${
                    filterType === "ALL" ? "bg-sky-500 text-white" : "text-slate-400 hover:text-slate-200"
                  }`}
                >
                  All ({summary.predictions_sample.length})
                </button>
                <button
                  onClick={() => setFilterType("ATTACK")}
                  className={`px-2.5 py-1 rounded-md font-medium transition-all ${
                    filterType === "ATTACK" ? "bg-red-500 text-white" : "text-slate-400 hover:text-slate-200"
                  }`}
                >
                  Attacks Only
                </button>
                <button
                  onClick={() => setFilterType("BENIGN")}
                  className={`px-2.5 py-1 rounded-md font-medium transition-all ${
                    filterType === "BENIGN" ? "bg-emerald-500 text-white" : "text-slate-400 hover:text-slate-200"
                  }`}
                >
                  Benign Only
                </button>
              </div>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs font-mono">
                <thead className="bg-slate-900/80 text-slate-400 border-b border-slate-800 uppercase tracking-wider text-[11px]">
                  <tr>
                    <th className="py-2.5 px-3">#</th>
                    <th className="py-2.5 px-3">Verdict</th>
                    <th className="py-2.5 px-3">Confidence</th>
                    <th className="py-2.5 px-3">Dest Port</th>
                    <th className="py-2.5 px-3">Duration (&mu;s)</th>
                    <th className="py-2.5 px-3">Throughput (Bytes/s)</th>
                    <th className="py-2.5 px-3">Attack Prob</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/60">
                  {filteredRows.map((r) => (
                    <tr key={r.row_index} className="hover:bg-slate-900/40 transition-all">
                      <td className="py-2.5 px-3 text-slate-400">{r.row_index}</td>
                      <td className="py-2.5 px-3">
                        <span
                          className={`px-2 py-0.5 rounded font-semibold text-[10px] ${
                            r.prediction === "ATTACK"
                              ? "bg-red-950 text-red-400 border border-red-800/60"
                              : "bg-emerald-950 text-emerald-400 border border-emerald-800/60"
                          }`}
                        >
                          {r.prediction}
                        </span>
                      </td>
                      <td className="py-2.5 px-3 text-slate-200">{(r.confidence * 100).toFixed(1)}%</td>
                      <td className="py-2.5 px-3 text-sky-400">{r.dest_port}</td>
                      <td className="py-2.5 px-3 text-slate-300">{r.flow_duration.toLocaleString()}</td>
                      <td className="py-2.5 px-3 text-slate-300">{r.flow_bytes_s.toFixed(1)}</td>
                      <td className="py-2.5 px-3 font-semibold text-slate-300">{(r.prob_attack * 100).toFixed(1)}%</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
