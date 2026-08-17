import React, { useState, useEffect } from "react";
import { Navbar } from "./components/Navbar";
import { Dashboard } from "./pages/Dashboard";
import { Predictor } from "./pages/Predictor";
import { BatchPredictor } from "./pages/BatchPredictor";
import { ModelComparison } from "./pages/ModelComparison";
import { OptimizationAnalysis } from "./pages/OptimizationAnalysis";
import { EnsembleAnalysis } from "./pages/EnsembleAnalysis";
import { Explainability } from "./pages/Explainability";
import { ResearchFindings } from "./pages/ResearchFindings";
import { DatasetExplorer } from "./pages/DatasetExplorer";
import { fetchHealth } from "./services/api";

export const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<string>("dashboard");
  const [isSynthetic, setIsSynthetic] = useState<boolean>(true);

  useEffect(() => {
    async function checkStatus() {
      try {
        const h = await fetchHealth();
        setIsSynthetic(h.is_dataset_synthetic);
      } catch (err) {
        console.error("Health check error:", err);
      }
    }
    checkStatus();
  }, []);

  return (
    <div className="min-h-screen flex flex-col bg-[#080C14] text-slate-100">
      {/* Top Navbar */}
      <Navbar activeTab={activeTab} setActiveTab={setActiveTab} isSynthetic={isSynthetic} />

      {/* Main Content Viewport */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-6">
        {activeTab === "dashboard" && <Dashboard onNavigate={(t) => setActiveTab(t)} />}
        {activeTab === "predictor" && <Predictor />}
        {activeTab === "batch" && <BatchPredictor />}
        {activeTab === "comparison" && <ModelComparison />}
        {activeTab === "optimization" && <OptimizationAnalysis />}
        {activeTab === "ensemble" && <EnsembleAnalysis />}
        {activeTab === "explainability" && <Explainability />}
        {activeTab === "research" && <ResearchFindings />}
        {activeTab === "dataset" && <DatasetExplorer />}
      </main>

      {/* Footer */}
      <footer className="bg-[#06090F] border-t border-slate-800/80 py-6">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-col sm:flex-row items-center justify-between text-xs text-slate-400 gap-3">
          <div className="flex items-center space-x-2">
            <span className="font-semibold text-slate-300">CyberGuard IDS</span>
            <span>&bull;</span>
            <span>ML Capstone Research System</span>
            <span>&bull;</span>
            <span className="font-mono text-sky-400">v1.0.0</span>
          </div>

          <div className="flex items-center space-x-4">
            <span className="font-mono">Leakage-Free Train/Val/Test Split Protocol</span>
            <span>&bull;</span>
            <button
              onClick={() => setActiveTab("research")}
              className="text-sky-400 hover:text-sky-300 underline font-medium"
            >
              Academic Viva Defense Summary
            </button>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default App;
