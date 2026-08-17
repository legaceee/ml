import React, { useEffect, useState } from "react";
import {
  ShieldAlert,
  ShieldCheck,
  Zap,
  TrendingUp,
  Cpu,
  Layers,
  ArrowUpRight,
  AlertTriangle,
  Radio,
  CheckCircle2
} from "lucide-react";
import {
  PieChart,
  Pie,
  Cell,
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip
} from "recharts";
import { fetchHealth, fetchMasterMetrics, fetchResearchConclusions } from "../services/api";
import { MasterComparison, ResearchConclusions } from "../types";

export const Dashboard: React.FC<{ onNavigate: (tab: string) => void }> = ({ onNavigate }) => {
  const [metrics, setMetrics] = useState<MasterComparison | null>(null);
  const [conclusions, setConclusions] = useState<ResearchConclusions | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function loadData() {
      try {
        const [m, c] = await Promise.all([
          fetchMasterMetrics(),
          fetchResearchConclusions()
        ]);
        setMetrics(m);
        setConclusions(c);
      } catch (err) {
        console.error("Dashboard data load error:", err);
      } finally {
        setLoading(false);
      }
    }
    loadData();
  }, []);

  if (loading || !metrics) {
    return (
      <div className="flex items-center justify-center min-h-[60vh]">
        <div className="flex flex-col items-center space-y-3">
          <div className="w-10 h-10 border-4 border-sky-500 border-t-transparent rounded-full animate-spin"></div>
          <p className="text-slate-400 font-mono text-sm">Initializing Security Telemetry...</p>
        </div>
      </div>
    );
  }

  // Find best performing models
  const modelsArr = Object.entries(metrics.models).map(([k, v]) => ({ key: k, ...v }));
  const bestF1Model = [...modelsArr].sort((a, b) => b.f1 - a.f1)[0];
  const bestRecallModel = [...modelsArr].sort((a, b) => b.recall - a.recall)[0];
  const lowestLatencyModel = [...modelsArr].sort((a, b) => a.inference_time_ms_per_1k - b.inference_time_ms_per_1k)[0];

  const pieData = [
    { name: "Benign Traffic", value: 55, color: "#10B981" },
    { name: "DoS / DDoS", value: 23, color: "#EF4444" },
    { name: "PortScan Probing", value: 12, color: "#F59E0B" },
    { name: "Patator Brute Force", value: 7, color: "#8B5CF6" },
    { name: "Botnet / Web", value: 3, color: "#EC4899" }
  ];

  const barComparisonData = modelsArr.slice(0, 7).map((m) => ({
    name: m.key.replace("_", " ").slice(0, 10),
    F1: (m.f1 * 100).toFixed(1),
    Recall: (m.recall * 100).toFixed(1),
    Accuracy: (m.accuracy * 100).toFixed(1)
  }));

  return (
    <div className="space-y-6">
      {/* Top Banner Notice */}
      {metrics.is_synthetic && (
        <div className="p-4 rounded-xl bg-amber-950/30 border border-amber-800/40 flex items-start space-x-3">
          <AlertTriangle className="w-5 h-5 text-amber-400 mt-0.5 flex-shrink-0" />
          <div className="text-sm">
            <span className="font-semibold text-amber-300">Development Synthetic Fixture Active:</span>
            <span className="text-slate-300 ml-1.5">
              The application is currently running against a strictly formatted development sample matching the 78-feature CIC-IDS2017 schema. To evaluate on the full multi-gigabyte UNB dataset, drop your CSV files into <code className="px-1.5 py-0.5 rounded bg-slate-900 text-amber-200 text-xs">ml/data/raw/</code> and execute <code className="px-1.5 py-0.5 rounded bg-slate-900 text-amber-200 text-xs">python ml/training/run_all_experiments.py</code>.
            </span>
          </div>
        </div>
      )}

      {/* KPI Cards Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {/* Total Models Evaluated */}
        <div className="cyber-card p-5 relative overflow-hidden group hover:border-sky-500/50 transition-all">
          <div className="flex justify-between items-start">
            <div>
              <p className="text-xs font-mono uppercase tracking-wider text-slate-400">Total ML Models</p>
              <h3 className="text-3xl font-bold text-white mt-1">{modelsArr.length}</h3>
              <p className="text-xs text-sky-400 mt-1 flex items-center">
                <CheckCircle2 className="w-3.5 h-3.5 mr-1" />
                7 Baselines + 5 Ensembles
              </p>
            </div>
            <div className="p-3 rounded-xl bg-sky-500/10 text-sky-400 border border-sky-500/20">
              <Layers className="w-6 h-6" />
            </div>
          </div>
        </div>

        {/* Highest F1 Score */}
        <div className="cyber-card p-5 relative overflow-hidden group hover:border-emerald-500/50 transition-all">
          <div className="flex justify-between items-start">
            <div>
              <p className="text-xs font-mono uppercase tracking-wider text-slate-400">Champion F1-Score</p>
              <h3 className="text-3xl font-bold text-emerald-400 mt-1">
                {(bestF1Model.f1 * 100).toFixed(2)}%
              </h3>
              <p className="text-xs text-slate-400 mt-1 truncate max-w-[140px]" title={bestF1Model.key}>
                {bestF1Model.key.replace("_", " ").toUpperCase()}
              </p>
            </div>
            <div className="p-3 rounded-xl bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
              <TrendingUp className="w-6 h-6" />
            </div>
          </div>
        </div>

        {/* Attack Detection Recall */}
        <div className="cyber-card p-5 relative overflow-hidden group hover:border-indigo-500/50 transition-all">
          <div className="flex justify-between items-start">
            <div>
              <p className="text-xs font-mono uppercase tracking-wider text-slate-400">Peak Attack Recall</p>
              <h3 className="text-3xl font-bold text-indigo-400 mt-1">
                {(bestRecallModel.recall * 100).toFixed(2)}%
              </h3>
              <p className="text-xs text-slate-400 mt-1">Zero Missed Attacks Goal</p>
            </div>
            <div className="p-3 rounded-xl bg-indigo-500/10 text-indigo-400 border border-indigo-500/20">
              <ShieldCheck className="w-6 h-6" />
            </div>
          </div>
        </div>

        {/* Lowest Latency Model */}
        <div className="cyber-card p-5 relative overflow-hidden group hover:border-purple-500/50 transition-all">
          <div className="flex justify-between items-start">
            <div>
              <p className="text-xs font-mono uppercase tracking-wider text-slate-400">Fastest Inference</p>
              <h3 className="text-3xl font-bold text-purple-400 mt-1">
                {lowestLatencyModel.inference_time_ms_per_1k.toFixed(2)}ms
              </h3>
              <p className="text-xs text-slate-400 mt-1">Per 1,000 flows analyzed</p>
            </div>
            <div className="p-3 rounded-xl bg-purple-500/10 text-purple-400 border border-purple-500/20">
              <Zap className="w-6 h-6" />
            </div>
          </div>
        </div>
      </div>

      {/* Main Visualizations Row */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Threat Distribution Chart */}
        <div className="cyber-card p-6 flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between">
              <h4 className="font-semibold text-white flex items-center space-x-2">
                <Radio className="w-4 h-4 text-sky-400" />
                <span>Traffic Profile & Threat Distribution</span>
              </h4>
            </div>
            <p className="text-xs text-slate-400 mt-1">Breakdown across Benign vs Attack vectors in CIC-IDS2017</p>
          </div>

          <div className="h-56 my-2">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={pieData}
                  cx="50%"
                  cy="50%"
                  innerRadius={50}
                  outerRadius={80}
                  paddingAngle={4}
                  dataKey="value"
                >
                  {pieData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} stroke="#0B0F19" strokeWidth={2} />
                  ))}
                </Pie>
                <Tooltip
                  contentStyle={{ backgroundColor: "#0F172A", borderColor: "#334155", borderRadius: "8px", fontSize: "12px" }}
                />
              </PieChart>
            </ResponsiveContainer>
          </div>

          <div className="grid grid-cols-2 gap-2 text-xs">
            {pieData.map((d) => (
              <div key={d.name} className="flex items-center space-x-2">
                <span className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: d.color }}></span>
                <span className="text-slate-300 truncate">{d.name}</span>
                <span className="text-slate-500 font-mono">{d.value}%</span>
              </div>
            ))}
          </div>
        </div>

        {/* Model Performance Comparison Bar Chart */}
        <div className="cyber-card p-6 lg:col-span-2 flex flex-col justify-between">
          <div className="flex items-center justify-between">
            <div>
              <h4 className="font-semibold text-white flex items-center space-x-2">
                <Cpu className="w-4 h-4 text-indigo-400" />
                <span>Performance Benchmark (F1 vs Recall vs Accuracy)</span>
              </h4>
              <p className="text-xs text-slate-400 mt-1">Untouched Test Set Evaluation across baseline and ensemble models</p>
            </div>
            <button
              onClick={() => onNavigate("comparison")}
              className="text-xs text-sky-400 hover:text-sky-300 font-medium flex items-center space-x-1"
            >
              <span>View Leaderboard</span>
              <ArrowUpRight className="w-3.5 h-3.5" />
            </button>
          </div>

          <div className="h-64 my-2">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={barComparisonData}>
                <XAxis dataKey="name" stroke="#64748B" fontSize={11} />
                <YAxis domain={[90, 100]} stroke="#64748B" fontSize={11} />
                <Tooltip
                  contentStyle={{ backgroundColor: "#0F172A", borderColor: "#334155", borderRadius: "8px", fontSize: "12px" }}
                />
                <Bar dataKey="F1" fill="#38BDF8" radius={[4, 4, 0, 0]} name="F1-Score (%)" />
                <Bar dataKey="Recall" fill="#818CF8" radius={[4, 4, 0, 0]} name="Attack Recall (%)" />
                <Bar dataKey="Accuracy" fill="#34D399" radius={[4, 4, 0, 0]} name="Accuracy (%)" />
              </BarChart>
            </ResponsiveContainer>
          </div>

          <div className="flex items-center justify-end space-x-4 text-xs font-mono text-slate-400">
            <span className="flex items-center"><span className="w-2.5 h-2.5 bg-sky-400 rounded mr-1.5"></span>F1-Score</span>
            <span className="flex items-center"><span className="w-2.5 h-2.5 bg-indigo-400 rounded mr-1.5"></span>Attack Recall</span>
            <span className="flex items-center"><span className="w-2.5 h-2.5 bg-emerald-400 rounded mr-1.5"></span>Accuracy</span>
          </div>
        </div>
      </div>

      {/* Quick Action Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div
          onClick={() => onNavigate("predictor")}
          className="cyber-card p-5 cursor-pointer hover:border-sky-500/50 hover:bg-slate-900/80 transition-all flex items-center space-x-4"
        >
          <div className="p-3 rounded-xl bg-sky-500/10 text-sky-400 border border-sky-500/20">
            <ShieldAlert className="w-6 h-6" />
          </div>
          <div>
            <h5 className="font-semibold text-white text-sm">Live Flow Inspector</h5>
            <p className="text-xs text-slate-400 mt-0.5">Test individual network flows with real-time SHAP attributions</p>
          </div>
        </div>

        <div
          onClick={() => onNavigate("batch")}
          className="cyber-card p-5 cursor-pointer hover:border-indigo-500/50 hover:bg-slate-900/80 transition-all flex items-center space-x-4"
        >
          <div className="p-3 rounded-xl bg-indigo-500/10 text-indigo-400 border border-indigo-500/20">
            <TrendingUp className="w-6 h-6" />
          </div>
          <div>
            <h5 className="font-semibold text-white text-sm">Batch CSV Predictor</h5>
            <p className="text-xs text-slate-400 mt-0.5">Upload traffic capture CSVs and export scored threat logs</p>
          </div>
        </div>

        <div
          onClick={() => onNavigate("research")}
          className="cyber-card p-5 cursor-pointer hover:border-emerald-500/50 hover:bg-slate-900/80 transition-all flex items-center space-x-4"
        >
          <div className="p-3 rounded-xl bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
            <CheckCircle2 className="w-6 h-6" />
          </div>
          <div>
            <h5 className="font-semibold text-white text-sm">Research Synthesis (RQ1-6)</h5>
            <p className="text-xs text-slate-400 mt-0.5">Read auto-generated empirical answers for academic viva defense</p>
          </div>
        </div>
      </div>
    </div>
  );
};
