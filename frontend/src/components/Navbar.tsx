import React from "react";
import {
  ShieldAlert,
  Activity,
  Layers,
  Sparkles,
  Search,
  BookOpen,
  Database,
  UploadCloud,
  BarChart3, FlaskConical } from "lucide-react";

interface NavbarProps {
  activeTab: string;
  setActiveTab: (tab: string) => void;
  isSynthetic: boolean;
}

export const Navbar: React.FC<NavbarProps> = ({ activeTab, setActiveTab, isSynthetic }) => {
  const navItems = [
    { id: "dashboard", label: "SOC Dashboard", icon: Activity },
    { id: "predictor", label: "Live Flow Inspector", icon: ShieldAlert },
    { id: "batch", label: "Batch CSV Predictor", icon: UploadCloud },
    { id: "comparison", label: "Model Leaderboard", icon: BarChart3 },
    { id: "optimization", label: "Optuna Optimization", icon: Sparkles },
    { id: "ensemble", label: "Ensemble Deep-Dive", icon: Layers },
    { id: "explainability", label: "SHAP Explainability", icon: Search },
    { id: "research", label: "Research Findings (RQ1-8)", icon: BookOpen },
    { id: "courselab", label: "Course Topics Lab", icon: FlaskConical },
    { id: "dataset", label: "Dataset Explorer", icon: Database },
  ];

  return (
    <header className="sticky top-0 z-50 bg-[#080C14]/90 backdrop-blur-md border-b border-slate-800">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Brand Logo & Capstone Title */}
          <div className="flex items-center space-x-3 cursor-pointer" onClick={() => setActiveTab("dashboard")}>
            <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-sky-500 to-indigo-600 flex items-center justify-center shadow-lg shadow-sky-500/20 border border-sky-400/30">
              <ShieldAlert className="w-6 h-6 text-white" />
            </div>
            <div>
              <div className="flex items-center space-x-2">
                <span className="font-bold text-lg tracking-tight bg-gradient-to-r from-sky-400 via-indigo-200 to-white bg-clip-text text-transparent">
                  CYBERGUARD IDS
                </span>
                <span className="text-[10px] px-2 py-0.5 rounded-full font-mono bg-sky-950/80 text-sky-400 border border-sky-800/60">
                  ML CAPSTONE
                </span>
              </div>
              <p className="text-[11px] text-slate-400 font-medium">Defensive Cyber Attack Intrusion Detection</p>
            </div>
          </div>

          {/* Dataset Status Banner */}
          <div className="hidden lg:flex items-center space-x-3">
            {isSynthetic ? (
              <div className="flex items-center space-x-2 px-3 py-1 rounded-full bg-amber-950/40 border border-amber-800/50 text-amber-300 text-xs font-mono">
                <span className="w-2 h-2 rounded-full bg-amber-400 animate-pulse"></span>
                <span>Dev Fixture Mode (CIC-IDS2017 Schema)</span>
              </div>
            ) : (
              <div className="flex items-center space-x-2 px-3 py-1 rounded-full bg-emerald-950/40 border border-emerald-800/50 text-emerald-300 text-xs font-mono">
                <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
                <span>Official CIC-IDS2017 Dataset Active</span>
              </div>
            )}
            <div className="flex items-center space-x-1.5 px-2.5 py-1 rounded-md bg-slate-900 border border-slate-800 text-xs text-slate-400 font-mono">
              <span className="w-1.5 h-1.5 rounded-full bg-sky-400 animate-ping"></span>
              <span>API ONLINE</span>
            </div>
          </div>
        </div>

        {/* Navigation Tabs Bar */}
        <nav className="flex space-x-1 overflow-x-auto py-2 scrollbar-none border-t border-slate-800/60">
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive = activeTab === item.id;
            return (
              <button
                key={item.id}
                onClick={() => setActiveTab(item.id)}
                className={`flex items-center space-x-2 px-3.5 py-2 rounded-lg text-xs font-medium whitespace-nowrap transition-all duration-200 ${
                  isActive
                    ? "bg-sky-500/15 text-sky-300 border border-sky-500/40 shadow-sm shadow-sky-500/10"
                    : "text-slate-400 hover:text-slate-200 hover:bg-slate-800/50 border border-transparent"
                }`}
              >
                <Icon className={`w-4 h-4 ${isActive ? "text-sky-400" : "text-slate-400"}`} />
                <span>{item.label}</span>
              </button>
            );
          })}
        </nav>
      </div>
    </header>
  );
};
