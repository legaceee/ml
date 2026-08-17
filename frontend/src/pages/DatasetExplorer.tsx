import React, { useEffect, useState } from "react";
import {
  Database,
  FileCheck,
  Filter,
  CheckCircle2,
  AlertTriangle,
  Info,
  Shield,
  Layers
} from "lucide-react";
import { fetchDatasetSummary } from "../services/api";
import { DatasetSummary } from "../types";

export const DatasetExplorer: React.FC = () => {
  const [summary, setSummary] = useState<DatasetSummary | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function load() {
      try {
        const s = await fetchDatasetSummary();
        setSummary(s);
      } catch (err) {
        console.error("Dataset summary error:", err);
      } finally {
        setLoading(false);
      }
    }
    load();
  }, []);

  if (loading || !summary) {
    return (
      <div className="flex items-center justify-center min-h-[50vh]">
        <div className="w-8 h-8 border-4 border-sky-500 border-t-transparent rounded-full animate-spin"></div>
      </div>
    );
  }

  const { dataset_info, cleaning_audit, split_info, features } = summary;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="cyber-card p-6">
        <h2 className="text-xl font-bold text-white flex items-center space-x-2">
          <Database className="w-5 h-5 text-sky-400" />
          <span>Dataset Schema & Preprocessing Pipeline Audit</span>
        </h2>
        <p className="text-xs text-slate-400 mt-1">
          Detailed inspection of network flow attributes, class balance, cleaning logs, and strict train-only transformation parameters.
        </p>
      </div>

      {/* Dataset Overview Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="cyber-card p-5">
          <p className="text-xs font-mono text-slate-400">DATASET SOURCE</p>
          <h3 className="text-lg font-bold text-white mt-1 truncate" title={dataset_info.source_file}>
            {dataset_info.source_file}
          </h3>
          <p className="text-[11px] text-sky-400 mt-1 font-mono">
            {dataset_info.is_synthetic ? "Synthetic Development Fixture" : "Official UNB CIC-IDS2017"}
          </p>
        </div>

        <div className="cyber-card p-5">
          <p className="text-xs font-mono text-slate-400">TOTAL SAMPLES</p>
          <h3 className="text-2xl font-bold text-white mt-1">{dataset_info.total_rows_loaded.toLocaleString()}</h3>
          <p className="text-[11px] text-slate-400 mt-1 font-mono">{dataset_info.memory_usage_mb.toFixed(1)} MB in-memory</p>
        </div>

        <div className="cyber-card p-5">
          <p className="text-xs font-mono text-slate-400">INPUT FEATURES</p>
          <h3 className="text-2xl font-bold text-indigo-400 mt-1">{summary.num_features}</h3>
          <p className="text-[11px] text-slate-400 mt-1 font-mono">Cleaned & Non-constant</p>
        </div>

        <div className="cyber-card p-5">
          <p className="text-xs font-mono text-slate-400">LEAKAGE PREVENTION</p>
          <h3 className="text-2xl font-bold text-emerald-400 mt-1">100% Isolated</h3>
          <p className="text-[11px] text-emerald-400/80 mt-1 font-mono">Train-only fit verified</p>
        </div>
      </div>

      {/* Cleaning Audit Report */}
      <div className="cyber-card p-6 space-y-4">
        <h3 className="font-semibold text-white text-sm flex items-center space-x-2">
          <FileCheck className="w-4 h-4 text-emerald-400" />
          <span>Automated Data Cleaning & Quality Audit</span>
        </h3>

        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 text-xs font-mono">
          <div className="p-4 rounded-lg bg-slate-900 border border-slate-800">
            <span className="text-slate-400">Infinite / NaN Cells Handled:</span>
            <p className="text-lg font-bold text-white mt-1">{cleaning_audit.nan_inf_cells_found}</p>
            <p className="text-[10px] text-slate-400 mt-0.5">Replaced via median imputation</p>
          </div>

          <div className="p-4 rounded-lg bg-slate-900 border border-slate-800">
            <span className="text-slate-400">Deduplicated Records:</span>
            <p className="text-lg font-bold text-white mt-1">{cleaning_audit.duplicates_removed}</p>
            <p className="text-[10px] text-slate-400 mt-0.5">Exact duplicate rows dropped</p>
          </div>

          <div className="p-4 rounded-lg bg-slate-900 border border-slate-800">
            <span className="text-slate-400">Zero-Variance Columns Dropped:</span>
            <p className="text-lg font-bold text-amber-400 mt-1">{cleaning_audit.constant_columns_dropped.length}</p>
            <p className="text-[10px] text-slate-400 mt-0.5">Constant protocol fields purged</p>
          </div>
        </div>
      </div>

      {/* Train / Val / Test Partition Distribution */}
      <div className="cyber-card p-6 space-y-4">
        <h3 className="font-semibold text-white text-sm flex items-center space-x-2">
          <Layers className="w-4 h-4 text-sky-400" />
          <span>Stratified Data Partitioning (70% Train / 15% Val / 15% Test)</span>
        </h3>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 font-mono text-xs">
          <div className="p-4 rounded-lg bg-sky-950/20 border border-sky-800/40">
            <div className="flex justify-between items-center">
              <span className="text-sky-300 font-bold">TRAINING SET (70%)</span>
              <span className="text-slate-400">{split_info.train_count} rows</span>
            </div>
            <p className="text-slate-300 text-[11px] mt-2">
              Attack Proportion: <strong className="text-white">{(split_info.train_attack_ratio * 100).toFixed(1)}%</strong>
            </p>
            <p className="text-[10px] text-slate-400 mt-1 font-sans">Used for model fitting, feature ranking, and CV tuning.</p>
          </div>

          <div className="p-4 rounded-lg bg-purple-950/20 border border-purple-800/40">
            <div className="flex justify-between items-center">
              <span className="text-purple-300 font-bold">VALIDATION SET (15%)</span>
              <span className="text-slate-400">{split_info.val_count} rows</span>
            </div>
            <p className="text-slate-300 text-[11px] mt-2">
              Attack Proportion: <strong className="text-white">{(split_info.val_attack_ratio * 100).toFixed(1)}%</strong>
            </p>
            <p className="text-[10px] text-slate-400 mt-1 font-sans">Used for ensemble probability weight calibration.</p>
          </div>

          <div className="p-4 rounded-lg bg-emerald-950/20 border border-emerald-800/40">
            <div className="flex justify-between items-center">
              <span className="text-emerald-300 font-bold">UNTOUCHED TEST SET (15%)</span>
              <span className="text-slate-400">{split_info.test_count} rows</span>
            </div>
            <p className="text-slate-300 text-[11px] mt-2">
              Attack Proportion: <strong className="text-white">{(split_info.test_attack_ratio * 100).toFixed(1)}%</strong>
            </p>
            <p className="text-[10px] text-slate-400 mt-1 font-sans">Used strictly for final benchmark evaluation.</p>
          </div>
        </div>
      </div>

      {/* Feature Catalog Grid */}
      <div className="cyber-card p-6 space-y-4">
        <div className="flex items-center justify-between">
          <h3 className="font-semibold text-white text-sm">Active Feature Schema Catalog ({features.length} features)</h3>
          <span className="text-xs text-slate-400 font-mono">CICFlowMeter Derived</span>
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-2 text-xs font-mono">
          {features.map((f, i) => (
            <div key={i} className="p-2 rounded bg-slate-900/60 border border-slate-800 text-slate-300 truncate" title={f}>
              <span className="text-slate-500 mr-1.5">{i + 1}.</span>
              <span>{f}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};
