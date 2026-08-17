import React, { useEffect, useState } from "react";
import {
  Search,
  Sliders,
  CheckCircle2,
  Info,
  ShieldAlert,
  ArrowRight
} from "lucide-react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer
} from "recharts";
import { fetchGlobalExplain, fetchModels } from "../services/api";
import { ModelInfo } from "../types";

export const Explainability: React.FC = () => {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [selectedModel, setSelectedModel] = useState<string>("optimized_xgboost");
  const [globalShap, setGlobalShap] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function load() {
      try {
        const mList = await fetchModels();
        setModels(mList);
        const opt = mList.find((m) => m.model_id === "optimized_xgboost") || mList[0];
        const mId = opt ? opt.model_id : "optimized_xgboost";
        setSelectedModel(mId);
        const gData = await fetchGlobalExplain(mId);
        setGlobalShap(gData.global_importance || []);
      } catch (err) {
        console.error("SHAP load error:", err);
      } finally {
        setLoading(false);
      }
    }
    load();
  }, []);

  const handleModelChange = async (mId: string) => {
    setSelectedModel(mId);
    setLoading(true);
    try {
      const gData = await fetchGlobalExplain(mId);
      setGlobalShap(gData.global_importance || []);
    } catch (err) {
      console.error("Model SHAP fetch error:", err);
    } finally {
      setLoading(false);
    }
  };

  const chartData = globalShap.slice(0, 15).map((f) => ({
    name: f.feature.length > 22 ? `${f.feature.slice(0, 22)}...` : f.feature,
    full_name: f.feature,
    importance: f.importance
  }));

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="cyber-card p-6 flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
        <div>
          <h2 className="text-xl font-bold text-white flex items-center space-x-2">
            <Search className="w-5 h-5 text-sky-400" />
            <span>Explainable AI (SHAP Global & Local Attributions)</span>
          </h2>
          <p className="text-xs text-slate-400 mt-1">
            Answering the vital cybersecurity question: <em>"Why did the machine learning model classify this specific traffic flow as malicious?"</em>
          </p>
        </div>

        <div className="flex items-center space-x-3 w-full md:w-auto">
          <label className="text-xs text-slate-400 font-mono">Model:</label>
          <select
            value={selectedModel}
            onChange={(e) => handleModelChange(e.target.value)}
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

      {/* Global SHAP Ranking Bar Chart */}
      <div className="cyber-card p-6 space-y-4">
        <div>
          <h3 className="font-semibold text-white text-sm">Global Feature Importance (Mean |SHAP| Attribution)</h3>
          <p className="text-xs text-slate-400 mt-0.5">
            Measured across background test flows to identify the protocol attributes that dominate decision boundaries.
          </p>
        </div>

        {loading ? (
          <div className="h-72 flex items-center justify-center">
            <div className="w-6 h-6 border-2 border-sky-500 border-t-transparent rounded-full animate-spin"></div>
          </div>
        ) : (
          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={chartData} layout="vertical" margin={{ left: 50 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1E293B" />
                <XAxis type="number" stroke="#64748B" fontSize={10} name="Mean |SHAP Value|" />
                <YAxis type="category" dataKey="name" stroke="#94A3B8" fontSize={11} width={130} />
                <Tooltip
                  contentStyle={{ backgroundColor: "#0F172A", borderColor: "#334155", borderRadius: "8px", fontSize: "12px" }}
                  formatter={(val: any) => [val, "SHAP Attribution"]}
                />
                <Bar dataKey="importance" fill="#38BDF8" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}
      </div>

      {/* Top Features Breakdown Grid */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {globalShap.slice(0, 3).map((f, i) => (
          <div key={i} className="cyber-card p-5 space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-sky-950 text-sky-400 border border-sky-800">
                RANK #{f.rank}
              </span>
              <span className="text-xs font-mono font-bold text-white">|SHAP| {f.importance}</span>
            </div>
            <h4 className="font-semibold text-white text-sm">{f.feature}</h4>
            <p className="text-xs text-slate-400">
              High discriminative variance for detecting anomalous burst volumetric spikes and protocol flag mismatches.
            </p>
          </div>
        ))}
      </div>
    </div>
  );
};
