import React, { useState, useEffect } from "react";
import {
  Layers,
  ShieldCheck,
  TrendingUp,
  Zap,
  Activity,
  Award,
  GitBranch,
  Network
} from "lucide-react";
import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  BarChart,
  Bar,
  Cell
} from "recharts";
import { fetchMasterMetrics } from "../services/api";
import { MasterComparison } from "../types";

export const EnsembleAnalysis: React.FC = () => {
  const [metrics, setMetrics] = useState<MasterComparison | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function load() {
      try {
        const m = await fetchMasterMetrics();
        setMetrics(m);
      } catch (err) {
        console.error("Ensemble analysis error:", err);
      } finally {
        setLoading(false);
      }
    }
    load();
  }, []);

  if (loading || !metrics) {
    return (
      <div className="flex items-center justify-center min-h-[50vh]">
        <div className="w-8 h-8 border-4 border-sky-500 border-t-transparent rounded-full animate-spin"></div>
      </div>
    );
  }

  const ensembles = [
    { key: "voting_hard", name: "Hard Voting", type: "Voting", ...metrics.models["voting_hard"] },
    { key: "voting_soft", name: "Soft Voting", type: "Voting", ...metrics.models["voting_soft"] },
    { key: "bagging_extra_trees", name: "Bagging (ExtraTrees)", type: "Bagging", ...metrics.models["bagging_extra_trees"] },
    { key: "adaboost", name: "AdaBoost", type: "Boosting", ...metrics.models["adaboost"] },
    { key: "gradient_boosting", name: "Gradient Boosting", type: "Boosting", ...metrics.models["gradient_boosting"] },
    { key: "stacking", name: "Stacking Classifier", type: "Stacking", ...metrics.models["stacking"] },
    { key: "weighted_ensemble", name: "Weighted Ensemble", type: "Weighted", ...metrics.models["weighted_ensemble"] }
  ];

  // Best individual model for reference
  const bestIndividual = metrics.models["optimized_xgboost"] || metrics.models["random_forest"];

  const tradeOffData = ensembles.map((e) => ({
    name: e.name,
    f1: Number((e.f1 * 100).toFixed(2)),
    recall: Number((e.recall * 100).toFixed(2)),
    latency: Number(e.inference_time_ms_per_1k.toFixed(2))
  }));

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="cyber-card p-6">
        <h2 className="text-xl font-bold text-white flex items-center space-x-2">
          <Layers className="w-5 h-5 text-purple-400" />
          <span>Ensemble Learning Architectures & Comparative Evaluation (RQ4 & RQ5)</span>
        </h2>
        <p className="text-xs text-slate-400 mt-1">
          Direct experimental investigation into whether combining diverse models (Voting, Bagging, Boosting, Stacking) outperforms individual optimized classifiers.
        </p>
      </div>

      {/* Stacking Architecture Diagram */}
      <div className="cyber-card p-6 space-y-4">
        <div className="flex items-center justify-between border-b border-slate-800 pb-3">
          <h3 className="font-semibold text-white text-sm flex items-center space-x-2">
            <Network className="w-4 h-4 text-sky-400" />
            <span>Stacking Meta-Learner Architecture (Zero-Leakage 5-Fold OOF Scheme)</span>
          </h3>
          <span className="text-xs font-mono text-emerald-400">Champion Generalizer</span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-4 gap-3 text-center">
          <div className="p-3 rounded-lg bg-slate-900 border border-slate-800">
            <p className="text-[10px] font-mono text-slate-400 uppercase">Base Learner 1</p>
            <p className="text-sm font-semibold text-sky-300 mt-1">Random Forest</p>
            <p className="text-[10px] text-slate-400">Bagged Tree Diversity</p>
          </div>
          <div className="p-3 rounded-lg bg-slate-900 border border-slate-800">
            <p className="text-[10px] font-mono text-slate-400 uppercase">Base Learner 2</p>
            <p className="text-sm font-semibold text-emerald-300 mt-1">XGBoost</p>
            <p className="text-[10px] text-slate-400">Gradient Step Regularization</p>
          </div>
          <div className="p-3 rounded-lg bg-slate-900 border border-slate-800">
            <p className="text-[10px] font-mono text-slate-400 uppercase">Base Learner 3</p>
            <p className="text-sm font-semibold text-indigo-300 mt-1">Support Vector Machine</p>
            <p className="text-[10px] text-slate-400">Max-Margin RBF Boundary</p>
          </div>
          <div className="p-3 rounded-lg bg-slate-900 border border-slate-800">
            <p className="text-[10px] font-mono text-slate-400 uppercase">Base Learner 4</p>
            <p className="text-sm font-semibold text-purple-300 mt-1">Logistic Regression</p>
            <p className="text-[10px] text-slate-400">Linear Probabilistic Baseline</p>
          </div>
        </div>

        <div className="flex flex-col items-center justify-center my-1">
          <div className="w-0.5 h-6 bg-sky-500/50"></div>
          <div className="px-6 py-2 rounded-xl bg-gradient-to-r from-sky-500/20 to-purple-500/20 border border-sky-500/40 text-center">
            <p className="text-[10px] font-mono text-slate-400">5-FOLD STRATIFIED OUT-OF-FOLD FEATURE BLENDING</p>
            <h4 className="text-sm font-bold text-white mt-0.5">Meta Learner: Calibrated Logistic Regression</h4>
          </div>
          <div className="w-0.5 h-6 bg-sky-500/50"></div>
          <div className="px-4 py-1.5 rounded-lg bg-emerald-500/20 border border-emerald-500/40 text-emerald-300 text-xs font-semibold font-mono">
            Final Threat Verdict: ATTACK (1) vs BENIGN (0)
          </div>
        </div>
      </div>

      {/* Ensemble Comparison Table */}
      <div className="cyber-card p-6 space-y-4">
        <h3 className="font-semibold text-white text-sm">Ensemble Paradigms Benchmark</h3>
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs font-mono">
            <thead className="bg-slate-900/80 text-slate-400 border-b border-slate-800 uppercase tracking-wider text-[11px]">
              <tr>
                <th className="py-2.5 px-3">Ensemble Method</th>
                <th className="py-2.5 px-3">Paradigm</th>
                <th className="py-2.5 px-3">F1-Score</th>
                <th className="py-2.5 px-3">Attack Recall</th>
                <th className="py-2.5 px-3">Precision</th>
                <th className="py-2.5 px-3">FPR</th>
                <th className="py-2.5 px-3">Inference (ms/1k)</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60">
              {ensembles.map((e) => (
                <tr key={e.key} className="hover:bg-slate-900/40 transition-all">
                  <td className="py-3 px-3 font-semibold text-white font-sans">{e.name}</td>
                  <td className="py-3 px-3">
                    <span className="px-2 py-0.5 rounded text-[10px] bg-slate-800 text-slate-300 border border-slate-700">
                      {e.type}
                    </span>
                  </td>
                  <td className="py-3 px-3 text-sky-400 font-semibold">{(e.f1 * 100).toFixed(2)}%</td>
                  <td className="py-3 px-3 text-indigo-300 font-semibold">{(e.recall * 100).toFixed(2)}%</td>
                  <td className="py-3 px-3 text-slate-300">{(e.precision * 100).toFixed(2)}%</td>
                  <td className="py-3 px-3 text-red-400">{(e.false_positive_rate * 100).toFixed(2)}%</td>
                  <td className="py-3 px-3 text-purple-400">{e.inference_time_ms_per_1k.toFixed(2)}ms</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* F1 vs Latency Trade-Off Scatter Chart */}
      <div className="cyber-card p-6">
        <h3 className="font-semibold text-white text-sm flex items-center space-x-2 mb-1">
          <Activity className="w-4 h-4 text-emerald-400" />
          <span>RQ5 Optimal Trade-Off Frontier: Attack F1 vs Inference Latency</span>
        </h3>
        <p className="text-xs text-slate-400 mb-4">
          Evaluating which ensemble maximizes accuracy while remaining viable for real-time high-throughput packet filtering.
        </p>

        <div className="h-64">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={tradeOffData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1E293B" />
              <XAxis dataKey="name" stroke="#64748B" fontSize={11} />
              <YAxis stroke="#64748B" fontSize={11} domain={[90, 100]} />
              <Tooltip
                contentStyle={{ backgroundColor: "#0F172A", borderColor: "#334155", borderRadius: "8px", fontSize: "12px" }}
              />
              <Bar dataKey="f1" fill="#8B5CF6" name="F1-Score (%)" radius={[4, 4, 0, 0]} />
              <Bar dataKey="recall" fill="#38BDF8" name="Attack Recall (%)" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
};
