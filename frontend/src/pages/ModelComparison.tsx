import React, { useState, useEffect } from "react";
import {
  BarChart3,
  TrendingUp,
  ShieldCheck,
  Zap,
  ArrowUpDown,
  CheckCircle2,
  AlertCircle
} from "lucide-react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend
} from "recharts";
import { fetchMasterMetrics } from "../services/api";
import { MasterComparison, ModelEvaluationDetail } from "../types";

export const ModelComparison: React.FC = () => {
  const [data, setData] = useState<MasterComparison | null>(null);
  const [selectedModelKey, setSelectedModelKey] = useState<string>("optimized_xgboost");
  const [sortField, setSortField] = useState<keyof ModelEvaluationDetail>("f1");
  const [sortAsc, setSortAsc] = useState(false);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function load() {
      try {
        const m = await fetchMasterMetrics();
        setData(m);
        if (m.models && Object.keys(m.models).length > 0) {
          if (m.models["stacking"]) setSelectedModelKey("stacking");
          else setSelectedModelKey(Object.keys(m.models)[0]);
        }
      } catch (err) {
        console.error("Comparison load error:", err);
      } finally {
        setLoading(false);
      }
    }
    load();
  }, []);

  if (loading || !data) {
    return (
      <div className="flex items-center justify-center min-h-[50vh]">
        <div className="w-8 h-8 border-4 border-sky-500 border-t-transparent rounded-full animate-spin"></div>
      </div>
    );
  }

  const modelsList = Object.entries(data.models).map(([k, v]) => ({
    key: k,
    ...v
  }));

  const sortedModels = [...modelsList].sort((a, b) => {
    const valA = a[sortField] as number;
    const valB = b[sortField] as number;
    return sortAsc ? valA - valB : valB - valA;
  });

  const handleSort = (field: keyof ModelEvaluationDetail) => {
    if (sortField === field) {
      setSortAsc(!sortAsc);
    } else {
      setSortField(field);
      setSortAsc(false);
    }
  };

  const selectedModel = data.models[selectedModelKey] || modelsList[0];
  const cm = selectedModel.confusion_matrix;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="cyber-card p-6">
        <h2 className="text-xl font-bold text-white flex items-center space-x-2">
          <BarChart3 className="w-5 h-5 text-sky-400" />
          <span>Model Benchmark Leaderboard & Comparative Evaluation</span>
        </h2>
        <p className="text-xs text-slate-400 mt-1">
          Rigorous performance evaluation across Baselines, Bayesian Optuna Tuned Models, and Ensembles on the untouched test partition.
        </p>
      </div>

      {/* Master Comparison Table */}
      <div className="cyber-card p-6 space-y-4">
        <div className="flex items-center justify-between">
          <h3 className="font-semibold text-white text-sm">Empirical Benchmark Matrix (Sortable)</h3>
          <span className="text-xs text-slate-400 font-mono">Total Models: {modelsList.length}</span>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs font-mono">
            <thead className="bg-slate-900/80 text-slate-400 border-b border-slate-800 uppercase tracking-wider text-[11px]">
              <tr>
                <th className="py-3 px-3">Algorithm</th>
                <th className="py-3 px-3 cursor-pointer hover:text-sky-300" onClick={() => handleSort("f1")}>
                  <span className="flex items-center">F1-Score <ArrowUpDown className="w-3 h-3 ml-1" /></span>
                </th>
                <th className="py-3 px-3 cursor-pointer hover:text-sky-300" onClick={() => handleSort("recall")}>
                  <span className="flex items-center">Attack Recall <ArrowUpDown className="w-3 h-3 ml-1" /></span>
                </th>
                <th className="py-3 px-3 cursor-pointer hover:text-sky-300" onClick={() => handleSort("precision")}>
                  <span className="flex items-center">Precision <ArrowUpDown className="w-3 h-3 ml-1" /></span>
                </th>
                <th className="py-3 px-3 cursor-pointer hover:text-sky-300" onClick={() => handleSort("accuracy")}>
                  <span className="flex items-center">Accuracy <ArrowUpDown className="w-3 h-3 ml-1" /></span>
                </th>
                <th className="py-3 px-3 cursor-pointer hover:text-sky-300" onClick={() => handleSort("roc_auc")}>
                  <span className="flex items-center">ROC-AUC <ArrowUpDown className="w-3 h-3 ml-1" /></span>
                </th>
                <th className="py-3 px-3 cursor-pointer hover:text-sky-300" onClick={() => handleSort("false_positive_rate")}>
                  <span className="flex items-center">FPR <ArrowUpDown className="w-3 h-3 ml-1" /></span>
                </th>
                <th className="py-3 px-3 cursor-pointer hover:text-sky-300" onClick={() => handleSort("training_time_sec")}>
                  <span className="flex items-center">Train (s) <ArrowUpDown className="w-3 h-3 ml-1" /></span>
                </th>
                <th className="py-3 px-3">Inspect</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60">
              {sortedModels.map((m) => {
                const isSelected = m.key === selectedModelKey;
                const isEnsemble = ["voting", "bagging", "adaboost", "gradient_boosting", "stacking", "weighted"].some((k) => m.key.includes(k));
                const isOptimized = m.key.includes("optimized");

                return (
                  <tr
                    key={m.key}
                    className={`transition-all cursor-pointer ${
                      isSelected
                        ? "bg-sky-500/10 border-l-4 border-sky-400"
                        : "hover:bg-slate-900/40"
                    }`}
                    onClick={() => setSelectedModelKey(m.key)}
                  >
                    <td className="py-3 px-3">
                      <div className="flex items-center space-x-2">
                        <span className="font-semibold text-slate-100 font-sans">{m.key.replace(/_/g, " ").toUpperCase()}</span>
                        {isEnsemble && (
                          <span className="px-1.5 py-0.2 rounded text-[9px] bg-purple-950 text-purple-300 border border-purple-800">
                            Ensemble
                          </span>
                        )}
                        {isOptimized && (
                          <span className="px-1.5 py-0.2 rounded text-[9px] bg-sky-950 text-sky-300 border border-sky-800">
                            Optuna
                          </span>
                        )}
                      </div>
                    </td>
                    <td className="py-3 px-3 font-semibold text-sky-400">{(m.f1 * 100).toFixed(2)}%</td>
                    <td className="py-3 px-3 font-semibold text-indigo-300">{(m.recall * 100).toFixed(2)}%</td>
                    <td className="py-3 px-3 text-slate-300">{(m.precision * 100).toFixed(2)}%</td>
                    <td className="py-3 px-3 text-slate-300">{(m.accuracy * 100).toFixed(2)}%</td>
                    <td className="py-3 px-3 text-emerald-400">{(m.roc_auc * 100).toFixed(2)}%</td>
                    <td className="py-3 px-3 text-red-400 font-semibold">{(m.false_positive_rate * 100).toFixed(2)}%</td>
                    <td className="py-3 px-3 text-slate-400">{m.training_time_sec.toFixed(2)}s</td>
                    <td className="py-3 px-3">
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          setSelectedModelKey(m.key);
                        }}
                        className="px-2 py-1 rounded bg-slate-800 hover:bg-slate-700 text-sky-300 text-[10px] font-sans"
                      >
                        Inspect
                      </button>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Selected Model Deep Dive: Confusion Matrix & ROC/PR Curves */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Confusion Matrix Interactive Box */}
        <div className="lg:col-span-5 cyber-card p-6 space-y-4">
          <div>
            <h3 className="font-semibold text-white text-sm flex items-center space-x-2">
              <ShieldCheck className="w-4 h-4 text-emerald-400" />
              <span>Confusion Matrix: {selectedModelKey.replace(/_/g, " ").toUpperCase()}</span>
            </h3>
            <p className="text-xs text-slate-400 mt-1">Ground Truth vs Model Prediction (Test Split: {selectedModel.num_test_samples} flows)</p>
          </div>

          <div className="grid grid-cols-2 gap-3 pt-2">
            {/* True Positive */}
            <div className="p-4 rounded-xl bg-emerald-950/30 border border-emerald-800/40 text-center">
              <p className="text-[11px] font-mono text-emerald-300 uppercase">True Positive (TP)</p>
              <h4 className="text-2xl font-black text-emerald-400 mt-1">{cm.tp}</h4>
              <p className="text-[10px] text-slate-400 mt-0.5">Attacks Correctly Detected</p>
            </div>

            {/* False Negative */}
            <div className="p-4 rounded-xl bg-red-950/30 border border-red-800/40 text-center">
              <p className="text-[11px] font-mono text-red-300 uppercase">False Negative (FN)</p>
              <h4 className="text-2xl font-black text-red-400 mt-1">{cm.fn}</h4>
              <p className="text-[10px] text-slate-400 mt-0.5">Missed Threats (Critical)</p>
            </div>

            {/* False Positive */}
            <div className="p-4 rounded-xl bg-amber-950/30 border border-amber-800/40 text-center">
              <p className="text-[11px] font-mono text-amber-300 uppercase">False Positive (FP)</p>
              <h4 className="text-2xl font-black text-amber-400 mt-1">{cm.fp}</h4>
              <p className="text-[10px] text-slate-400 mt-0.5">Benign Flagged as Attack</p>
            </div>

            {/* True Negative */}
            <div className="p-4 rounded-xl bg-slate-900 border border-slate-800 text-center">
              <p className="text-[11px] font-mono text-slate-300 uppercase">True Negative (TN)</p>
              <h4 className="text-2xl font-black text-white mt-1">{cm.tn}</h4>
              <p className="text-[10px] text-slate-400 mt-0.5">Benign Correctly Verified</p>
            </div>
          </div>

          <div className="p-3 rounded-lg bg-slate-900/80 border border-slate-800 text-xs space-y-1.5">
            <div className="flex justify-between">
              <span className="text-slate-400">Specificity (TNR):</span>
              <span className="font-mono font-semibold text-white">{(selectedModel.specificity * 100).toFixed(2)}%</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-400">False Positive Rate (FPR):</span>
              <span className="font-mono font-semibold text-amber-400">{(selectedModel.false_positive_rate * 100).toFixed(2)}%</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-400">False Negative Rate (FNR):</span>
              <span className="font-mono font-semibold text-red-400">{(selectedModel.false_negative_rate * 100).toFixed(2)}%</span>
            </div>
          </div>
        </div>

        {/* ROC & PR Curves Column */}
        <div className="lg:col-span-7 cyber-card p-6 flex flex-col justify-between">
          <div>
            <h3 className="font-semibold text-white text-sm flex items-center space-x-2">
              <TrendingUp className="w-4 h-4 text-sky-400" />
              <span>Receiver Operating Characteristic (ROC Curve)</span>
            </h3>
            <p className="text-xs text-slate-400 mt-1">
              True Positive Rate vs False Positive Rate (AUC: {(selectedModel.roc_auc * 100).toFixed(2)}%)
            </p>
          </div>

          <div className="h-64 my-2">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={selectedModel.roc_curve}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1E293B" />
                <XAxis dataKey="fpr" type="number" domain={[0, 1]} stroke="#64748B" fontSize={11} name="False Positive Rate" />
                <YAxis domain={[0, 1]} stroke="#64748B" fontSize={11} name="True Positive Rate" />
                <Tooltip
                  contentStyle={{ backgroundColor: "#0F172A", borderColor: "#334155", borderRadius: "8px", fontSize: "12px" }}
                />
                <Line type="monotone" dataKey="tpr" stroke="#38BDF8" strokeWidth={3} dot={false} name="Model ROC" />
              </LineChart>
            </ResponsiveContainer>
          </div>

          <div className="text-xs text-slate-400 font-mono text-right">
            <span>Evaluated Model: {selectedModelKey.replace(/_/g, " ").toUpperCase()}</span>
          </div>
        </div>
      </div>
    </div>
  );
};
