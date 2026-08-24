import React, { useEffect, useState } from "react";
import { FlaskConical, Scale, Repeat, Crosshair, Info } from "lucide-react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
  ErrorBar
} from "recharts";
import { fetchArtifact } from "../services/api";

const pretty = (k: string) =>
  ({
    logistic_regression: "Logistic Regression",
    decision_tree: "Decision Tree",
    random_forest: "Random Forest",
    extra_trees: "Extra Trees",
    support_vector_machine: "SVM (RBF)",
    k_nearest_neighbors: "KNN",
    xgboost: "XGBoost",
    optimized_random_forest: "RF (Optuna)",
    optimized_xgboost: "XGB (Optuna)",
    voting_hard: "Voting (hard)",
    voting_soft: "Voting (soft)",
    bagging_extra_trees: "Bagging (ET)",
    adaboost: "AdaBoost",
    gradient_boosting: "Grad. Boosting",
    stacking: "Stacking",
    weighted_ensemble: "Weighted"
  } as Record<string, string>)[k] || k.replace(/_/g, " ");

const COLORS = ["#0072B2", "#E69F00", "#009E73", "#D55E00"];

const Section: React.FC<{ icon: React.ElementType; title: string; subtitle: string; children: React.ReactNode }> = ({
  icon: Icon,
  title,
  subtitle,
  children
}) => (
  <div className="cyber-card p-6 space-y-4">
    <div>
      <h3 className="text-base font-bold text-white flex items-center space-x-2">
        <Icon className="w-4 h-4 text-sky-400" />
        <span>{title}</span>
      </h3>
      <p className="text-xs text-slate-400 mt-1">{subtitle}</p>
    </div>
    {children}
  </div>
);

export const CourseLab: React.FC = () => {
  const [search, setSearch] = useState<any>(null);
  const [imb, setImb] = useState<any>(null);
  const [cv, setCv] = useState<any>(null);
  const [pcr, setPcr] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function load() {
      try {
        const [s, i, c, p] = await Promise.all([
          fetchArtifact("search-comparison"),
          fetchArtifact("imbalance-study"),
          fetchArtifact("cv-results"),
          fetchArtifact("per-class-recall").catch(() => null)
        ]);
        setSearch(s);
        setImb(i);
        setCv(c);
        setPcr(p);
      } catch (err) {
        console.error("Course lab load error:", err);
      } finally {
        setLoading(false);
      }
    }
    load();
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-[50vh]">
        <div className="w-8 h-8 border-4 border-sky-500 border-t-transparent rounded-full animate-spin"></div>
      </div>
    );
  }

  // ---- Grid vs Random vs Optuna ---------------------------------------------
  const searchRows: any[] = search?.search_comparison || [];
  const methods = Array.from(new Set(searchRows.map((r) => r.method)));
  const algos = Array.from(new Set(searchRows.map((r) => r.algorithm)));
  const searchChart = algos.map((a) => {
    const row: any = { algorithm: pretty(a) };
    methods.forEach((m) => {
      const r = searchRows.find((x) => x.algorithm === a && x.method === m);
      row[m] = r ? +(r.best_cv_f1 * 100).toFixed(2) : 0;
      row[`${m}_time`] = r ? r.wall_time_sec : 0;
    });
    return row;
  });

  // ---- Imbalance -------------------------------------------------------------
  const imbRecords: any[] = imb?.records || [];
  const strategies: string[] = imb?.strategies || [];
  const imbModels = Array.from(new Set(imbRecords.map((r) => r.model)));
  const imbChart = imbModels.map((m) => {
    const row: any = { model: pretty(m) };
    strategies.forEach((s) => {
      const r = imbRecords.find((x) => x.model === m && x.strategy === s);
      row[s] = r ? +(r.recall * 100).toFixed(2) : 0;
    });
    return row;
  });

  // ---- CV ---------------------------------------------------------------------
  const cvModels: any = cv?.models || {};
  const cvChart = Object.keys(cvModels)
    .map((k) => ({
      model: pretty(k),
      cv_f1: +(cvModels[k].f1_mean * 100).toFixed(2),
      err: +(cvModels[k].f1_std * 100).toFixed(2),
      test_f1: +(cvModels[k].test_f1 * 100).toFixed(2),
      gap: cvModels[k].overfit_gap_f1
    }))
    .sort((a, b) => b.cv_f1 - a.cv_f1);

  // ---- Per-class recall -------------------------------------------------------
  const pcrModels: any = pcr?.models || {};
  const pcrShow = ["logistic_regression", "support_vector_machine", "optimized_xgboost", "stacking"].filter((k) => pcrModels[k]);
  const pcrClasses: any[] = pcrShow.length ? pcrModels[pcrShow[0]].filter((r: any) => r.is_attack) : [];

  return (
    <div className="space-y-6">
      <div className="cyber-card p-6">
        <h2 className="text-xl font-bold text-white flex items-center space-x-2">
          <FlaskConical className="w-5 h-5 text-sky-400" />
          <span>Course Topics Lab</span>
        </h2>
        <p className="text-xs text-slate-400 mt-1">
          The syllabus experiments, measured on the real CIC-IDS2017 sample: Grid Search vs Random Search vs Bayesian tuning, imbalanced-data
          strategies, k-fold cross-validation, and per-attack-type recall. Every value comes from <code className="font-mono">ml/artifacts/metrics/</code>.
        </p>
      </div>

      {/* Search comparison */}
      <Section
        icon={Crosshair}
        title="Hyperparameter search: Grid vs Random vs Optuna (TPE)"
        subtitle={search?.protocol || "Same CV splitter and objective for all three; only the search strategy differs."}
      >
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={searchChart} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
                <CartesianGrid stroke="#1e293b" vertical={false} />
                <XAxis dataKey="algorithm" stroke="#94a3b8" fontSize={11} />
                <YAxis domain={["dataMin - 1", 100]} stroke="#94a3b8" fontSize={11} unit="%" />
                <Tooltip contentStyle={{ background: "#0f172a", border: "1px solid #334155", fontSize: 12 }} />
                <Legend wrapperStyle={{ fontSize: 11 }} />
                {methods.map((m, i) => (
                  <Bar key={m} dataKey={m} name={`${m} (best CV F1)`} fill={COLORS[i]} radius={[4, 4, 0, 0]} />
                ))}
              </BarChart>
            </ResponsiveContainer>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-xs font-mono">
              <thead className="text-slate-400 border-b border-slate-800">
                <tr>
                  <th className="text-left py-2">Algorithm</th>
                  <th className="text-left">Method</th>
                  <th className="text-right">Configs</th>
                  <th className="text-right">Time (s)</th>
                  <th className="text-right">CV F1</th>
                  <th className="text-right">Test F1</th>
                </tr>
              </thead>
              <tbody>
                {searchRows.map((r, i) => (
                  <tr key={i} className="border-b border-slate-800/60 text-slate-200">
                    <td className="py-1.5">{pretty(r.algorithm)}</td>
                    <td>{r.method}</td>
                    <td className="text-right">{r.n_configs_evaluated}</td>
                    <td className="text-right">{r.wall_time_sec.toFixed(0)}</td>
                    <td className="text-right">{r.best_cv_f1.toFixed(4)}</td>
                    <td className="text-right font-bold text-white">{r.test_f1.toFixed(4)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </Section>

      {/* Imbalance */}
      <Section
        icon={Scale}
        title="Imbalanced-dataset handling"
        subtitle={`Train split is ${((imb?.train_attack_ratio_before || 0) * 100).toFixed(1)}% attack. Resampling is applied to the training split only; every bar is measured on the same untouched test split.`}
      >
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={imbChart} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
                <CartesianGrid stroke="#1e293b" vertical={false} />
                <XAxis dataKey="model" stroke="#94a3b8" fontSize={11} />
                <YAxis domain={["dataMin - 2", 100]} stroke="#94a3b8" fontSize={11} unit="%" />
                <Tooltip contentStyle={{ background: "#0f172a", border: "1px solid #334155", fontSize: 12 }} />
                <Legend wrapperStyle={{ fontSize: 11 }} />
                {strategies.map((s, i) => (
                  <Bar key={s} dataKey={s} name={`${s.replace(/_/g, " ")} (recall)`} fill={COLORS[i]} radius={[4, 4, 0, 0]} />
                ))}
              </BarChart>
            </ResponsiveContainer>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-xs font-mono">
              <thead className="text-slate-400 border-b border-slate-800">
                <tr>
                  <th className="text-left py-2">Strategy</th>
                  <th className="text-left">Model</th>
                  <th className="text-right">Train rows</th>
                  <th className="text-right">Recall</th>
                  <th className="text-right">FPR</th>
                  <th className="text-right">F1</th>
                </tr>
              </thead>
              <tbody>
                {imbRecords.map((r, i) => (
                  <tr key={i} className="border-b border-slate-800/60 text-slate-200">
                    <td className="py-1.5">{r.strategy}</td>
                    <td>{pretty(r.model)}</td>
                    <td className="text-right">{r.train_rows_after_resampling.toLocaleString()}</td>
                    <td className="text-right">{r.recall.toFixed(4)}</td>
                    <td className="text-right">{r.false_positive_rate.toFixed(4)}</td>
                    <td className="text-right font-bold text-white">{r.f1.toFixed(4)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
        <p className="text-[11px] text-slate-400 flex items-start space-x-2">
          <Info className="w-3.5 h-3.5 mt-0.5 text-sky-400 flex-shrink-0" />
          <span>
            Class weighting and SMOTE raise attack recall for the linear model at the price of more false alarms; tree ensembles are largely
            insensitive because a tree can isolate the minority region regardless of the prior.
          </span>
        </p>
      </Section>

      {/* CV */}
      <Section
        icon={Repeat}
        title={`${cv?.n_splits || 5}-fold stratified cross-validation (train split)`}
        subtitle="Mean ± std of F1 across folds. The diamond in the table is the single test-split F1; a large gap between train-fold and validation-fold F1 means the model memorises."
      >
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={cvChart} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
                <CartesianGrid stroke="#1e293b" vertical={false} />
                <XAxis dataKey="model" stroke="#94a3b8" fontSize={10} interval={0} angle={-15} textAnchor="end" height={50} />
                <YAxis domain={["dataMin - 2", 100]} stroke="#94a3b8" fontSize={11} unit="%" />
                <Tooltip contentStyle={{ background: "#0f172a", border: "1px solid #334155", fontSize: 12 }} />
                <Bar dataKey="cv_f1" name="CV F1 (mean)" fill={COLORS[0]} radius={[4, 4, 0, 0]}>
                  <ErrorBar dataKey="err" width={4} stroke="#f8fafc" />
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-xs font-mono">
              <thead className="text-slate-400 border-b border-slate-800">
                <tr>
                  <th className="text-left py-2">Model</th>
                  <th className="text-right">CV F1</th>
                  <th className="text-right">± std</th>
                  <th className="text-right">Test F1</th>
                  <th className="text-right">Overfit gap</th>
                </tr>
              </thead>
              <tbody>
                {cvChart.map((r, i) => (
                  <tr key={i} className="border-b border-slate-800/60 text-slate-200">
                    <td className="py-1.5">{r.model}</td>
                    <td className="text-right">{(r.cv_f1 / 100).toFixed(4)}</td>
                    <td className="text-right">{(r.err / 100).toFixed(4)}</td>
                    <td className="text-right font-bold text-white">{(r.test_f1 / 100).toFixed(4)}</td>
                    <td className={`text-right ${r.gap > 0.01 ? "text-amber-400" : "text-emerald-400"}`}>{r.gap >= 0 ? "+" : ""}{r.gap.toFixed(4)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </Section>

      {/* Per-class recall */}
      {pcrShow.length > 0 && (
        <Section
          icon={FlaskConical}
          title="Per-attack-type recall (test split)"
          subtitle="A binary F1 above 0.99 can hide a completely missed rare class. Detection rate for each original CIC-IDS2017 category."
        >
          <div className="overflow-x-auto">
            <table className="w-full text-xs font-mono">
              <thead className="text-slate-400 border-b border-slate-800">
                <tr>
                  <th className="text-left py-2">Attack category</th>
                  <th className="text-right">n (test)</th>
                  {pcrShow.map((k) => (
                    <th key={k} className="text-right">{pretty(k)}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {pcrClasses.map((c: any) => (
                  <tr key={c.class} className="border-b border-slate-800/60 text-slate-200">
                    <td className="py-1.5">{c.class}</td>
                    <td className="text-right">{c.n_test}</td>
                    {pcrShow.map((k) => {
                      const r = pcrModels[k].find((x: any) => x.class === c.class);
                      const v = r?.recall ?? 0;
                      return (
                        <td key={k} className={`text-right ${v < 0.9 ? "text-rose-400" : v < 0.99 ? "text-amber-400" : "text-emerald-400"}`}>
                          {v.toFixed(3)}
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Section>
      )}
    </div>
  );
};
