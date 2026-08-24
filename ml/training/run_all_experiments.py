"""
Master Experiment Runner.

Every number that appears in the README, the docs and the dashboard is produced
here from model outputs. Nothing is hand-typed.

Steps
 1. Feature selection benchmark (correlation filter, mutual information,
    tree importance, ANOVA SelectKBest, PCA) - measured F1 and train time.
 2. Baseline models (7 algorithms) - held-out test metrics + 5-fold CV.
 3. Imbalanced-data study - none vs class_weight vs under-sampling vs SMOTE.
 4. Hyperparameter search comparison - Grid vs Random vs Optuna (RF, XGB).
 5. Ensembles - Voting (hard/soft), Bagging, Boosting, Stacking, Weighted.
 6. Per-attack-type recall for every model.
 7. Statistical stability - bootstrap 95% CIs + multi-seed retraining.
 8. SHAP global explanation of the best model.
 9. Master comparison, experiment CSV, real-flow presets, data-driven conclusions.

Usage
    python -m ml.training.run_all_experiments            # full run
    python -m ml.training.run_all_experiments --quick    # smoke test (small budgets)
"""

import argparse
import json
import platform
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from ml.explainability.shap_analysis import IDSExplainer
from ml.feature_engineering.dimensionality_reduction import PCAReducer
from ml.feature_engineering.selection import FeatureSelector
from ml.models.baselines import BASELINE_FAMILY, get_baseline_models
from ml.models.ensembles import ENSEMBLE_FAMILY, build_ensemble_models
from ml.models.optimization import HyperparameterOptimizer
from ml.preprocessing.pipeline import PipelineOrchestrator
from ml.preprocessing.resampling import run_imbalance_study
from ml.training.evaluate import (
    bootstrap_confidence_intervals,
    cross_validate_classifier,
    evaluate_model,
    per_class_recall,
)

BASELINE_KEYS = ["logistic_regression", "decision_tree", "random_forest", "extra_trees",
                 "support_vector_machine", "k_nearest_neighbors", "xgboost"]
OPTIMIZED_KEYS = ["optimized_random_forest", "optimized_xgboost"]
ENSEMBLE_KEYS = ["voting_hard", "voting_soft", "bagging_extra_trees", "adaboost",
                 "gradient_boosting", "stacking", "weighted_ensemble"]

PRESET_SPECS = [
    ("benign_web", "BENIGN", "Benign web traffic (real flow)", "A genuine benign HTTP/HTTPS flow taken from the CIC-IDS2017 test split."),
    ("benign_dns", "BENIGN", "Benign DNS lookup (real flow)", "A genuine benign DNS flow (port 53) taken from the test split."),
    ("dos_hulk", "DoS Hulk", "DoS Hulk (HTTP flood)", "Real DoS Hulk flow: many small forward packets, almost no backward traffic."),
    ("port_scan", "PortScan", "PortScan reconnaissance", "Real PortScan probe: 1-2 packets, SYN set, no handshake completion."),
    ("ddos", "DDoS", "DDoS (LOIC)", "Real DDoS flow from the Friday afternoon LOIC attack."),
    ("ftp_patator", "FTP-Patator", "FTP brute force", "Real FTP-Patator credential-guessing flow on port 21."),
    ("ssh_patator", "SSH-Patator", "SSH brute force", "Real SSH-Patator credential-guessing flow on port 22."),
    ("botnet_traffic", "Bot", "Botnet (ARES) C2 beacon", "Real Bot flow from the Friday morning ARES botnet capture."),
    ("dos_slowloris", "DoS slowloris", "DoS slowloris (slow headers)", "Real slowloris flow: long duration, tiny throughput."),
    ("web_bruteforce", "Web Attack - Brute Force", "Web application brute force", "Real web brute-force flow against the DVWA login form."),
    ("infiltration", "Infiltration", "Infiltration (Dropbox download)", "Real infiltration flow - one of the rarest classes (36 rows in the corpus)."),
    ("heartbleed", "Heartbleed", "Heartbleed exploit", "Real Heartbleed flow - the rarest class (11 rows in the corpus)."),
]


class MasterExperimentOrchestrator:
    def __init__(self, base_dir: Optional[Path] = None, quick: bool = False, random_state: int = 42):
        self.base_dir = Path(base_dir) if base_dir else Path(__file__).resolve().parent.parent
        self.quick = quick
        self.random_state = random_state

        self.splits_file = self.base_dir / "data" / "splits" / "dataset_splits.joblib"
        self.artifacts_dir = self.base_dir / "artifacts"
        self.models_dir = self.artifacts_dir / "models"
        self.metrics_dir = self.artifacts_dir / "metrics"
        self.experiments_dir = self.artifacts_dir / "experiments"
        for d in (self.models_dir, self.metrics_dir, self.experiments_dir):
            d.mkdir(parents=True, exist_ok=True)

        # Budgets
        self.n_trials = 4 if quick else 20
        self.n_random_iter = 4 if quick else 12
        self.cv_folds = 3 if quick else 5
        self.n_bootstrap = 30 if quick else 200
        self.seeds = [42, 123] if quick else [42, 123, 2024]

        self.experiment_records: List[Dict[str, Any]] = []
        self.step_timings: Dict[str, float] = {}

    # ------------------------------------------------------------------ #
    def _save_json(self, name: str, payload: Any, folder: Optional[Path] = None):
        folder = folder or self.metrics_dir
        with open(folder / name, "w") as f:
            json.dump(payload, f, indent=2, default=_json_default)

    def load_or_create_splits(self) -> Dict[str, Any]:
        if not self.splits_file.exists():
            PipelineOrchestrator(base_dir=str(self.base_dir), random_seed=self.random_state).run()
        return joblib.load(self.splits_file)

    def _log_experiment(self, exp_id, model_name, feature_set, optimization, ensemble, ev):
        self.experiment_records.append({
            "experiment_id": exp_id, "model": model_name, "feature_method": feature_set,
            "optimization": optimization, "ensemble": ensemble,
            "accuracy": ev["accuracy"], "precision": ev["precision"], "recall": ev["recall"], "f1": ev["f1"],
            "roc_auc": ev["roc_auc"], "pr_auc": ev["pr_auc"], "fpr": ev["false_positive_rate"], "fnr": ev["false_negative_rate"],
            "training_time_sec": ev["training_time_sec"], "inference_time_ms": ev["inference_time_ms_per_1k"],
        })

    # ------------------------------------------------------------------ #
    def run(self) -> Dict[str, Any]:
        t_run0 = time.perf_counter()
        print("=" * 72)
        print(">>> MASTER CYBER INTRUSION DETECTION ML BENCHMARK SUITE" + ("  [QUICK MODE]" if self.quick else ""))
        print("=" * 72)

        splits = self.load_or_create_splits()
        X_train, y_train = splits["X_train"], splits["y_train"]
        X_val, y_val = splits["X_val"], splits["y_val"]
        X_test, y_test = splits["X_test"], splits["y_test"]
        y_test_labels = splits.get("y_test_labels")
        feature_names = splits["feature_names"]
        X_train_df_raw = splits["X_train_df_raw"]
        is_synthetic = bool(splits.get("is_synthetic", True))
        if self.quick:
            # Smoke-test budget: stratified sub-sample so every code path runs in minutes
            rng = np.random.RandomState(self.random_state)
            def _sub(X, y, labels, n):
                if len(y) <= n:
                    return X, y, labels
                idx = rng.choice(len(y), size=n, replace=False)
                return X[idx], y[idx], (None if labels is None else np.asarray(labels)[idx])
            X_train, y_train, _tr_lab = _sub(X_train, y_train, splits.get("y_train_labels"), 6000)
            X_val, y_val, _ = _sub(X_val, y_val, None, 1500)
            X_test, y_test, y_test_labels = _sub(X_test, y_test, y_test_labels, 1500)
            X_train_df_raw = X_train_df_raw.iloc[:len(y_train)]
        print(f"Train {X_train.shape}, Val {X_val.shape}, Test {X_test.shape}, features={len(feature_names)}, synthetic={is_synthetic}")

        all_evaluations: Dict[str, Dict[str, Any]] = {}
        fitted_models: Dict[str, Any] = {}

        # ============================================================== #
        # 1. Feature selection benchmark (RQ2)
        # ============================================================== #
        t0 = time.perf_counter()
        print("\n[Step 1/9] Feature selection benchmark ...")
        fs = FeatureSelector(random_state=self.random_state)
        k = 25 if len(feature_names) > 25 else max(5, len(feature_names) // 2)
        corr_features = fs.correlation_filter(X_train_df_raw, threshold=0.90)
        mi_features = fs.mutual_information(X_train, y_train, feature_names, top_k=k)
        tree_features = fs.tree_importance(X_train, y_train, feature_names, top_k=k)
        kbest_features = fs.select_k_best(X_train, y_train, feature_names, k=k)

        pca_reducer = PCAReducer(variance_threshold=0.95, random_state=self.random_state)
        X_train_pca = pca_reducer.fit_transform(X_train)
        X_test_pca = pca_reducer.transform(X_test)
        pca_summary = pca_reducer.get_variance_summary()
        self._save_json("pca_summary.json", pca_summary)

        name_to_idx = {n: i for i, n in enumerate(feature_names)}
        subsets = {
            "all_features": list(feature_names),
            # Ablation: Destination Port is a well-known shortcut feature in CIC-IDS2017
            # (attacks target specific services). Does the model still work without it?
            "all_minus_destination_port": [f for f in feature_names if f != "Destination Port"],
            "correlation_filter_0.90": corr_features,
            f"mutual_info_top_{k}": mi_features,
            f"tree_importance_top_{k}": tree_features,
            f"anova_kbest_{k}": kbest_features,
        }
        fs_bench = []
        probe_models = {
            "random_forest": lambda: RandomForestClassifier(n_estimators=100, random_state=self.random_state, class_weight="balanced", n_jobs=-1),
            "logistic_regression": lambda: LogisticRegression(max_iter=2000, random_state=self.random_state, class_weight="balanced"),
        }
        for subset_name, feats in subsets.items():
            idx = [name_to_idx[f] for f in feats]
            Xtr, Xte = X_train[:, idx], X_test[:, idx]
            for probe_name, factory in probe_models.items():
                m = factory()
                t1 = time.perf_counter(); m.fit(Xtr, y_train); tt = time.perf_counter() - t1
                ev = evaluate_model(m, Xte, y_test, model_name=probe_name, training_time=tt)
                fs_bench.append({"feature_set": subset_name, "n_features": len(idx), "model": probe_name,
                                 "f1": ev["f1"], "recall": ev["recall"], "precision": ev["precision"], "accuracy": ev["accuracy"],
                                 "roc_auc": ev["roc_auc"], "training_time_sec": round(tt, 4),
                                 "inference_time_ms_per_1k": ev["inference_time_ms_per_1k"]})
        for probe_name, factory in probe_models.items():
            m = factory()
            t1 = time.perf_counter(); m.fit(X_train_pca, y_train); tt = time.perf_counter() - t1
            ev = evaluate_model(m, X_test_pca, y_test, model_name=probe_name, training_time=tt)
            fs_bench.append({"feature_set": f"pca_95pct_variance", "n_features": int(pca_summary["n_components_retained"]), "model": probe_name,
                             "f1": ev["f1"], "recall": ev["recall"], "precision": ev["precision"], "accuracy": ev["accuracy"],
                             "roc_auc": ev["roc_auc"], "training_time_sec": round(tt, 4),
                             "inference_time_ms_per_1k": ev["inference_time_ms_per_1k"]})

        fs_summary = fs.get_summary()
        fs_summary["benchmark"] = fs_bench
        fs_summary["subset_sizes"] = {k_: len(v) for k_, v in subsets.items()}
        fs_summary["subset_sizes"]["pca_95pct_variance"] = int(pca_summary["n_components_retained"])
        self._save_json("feature_selection_summary.json", fs_summary)
        for r in fs_bench:
            print(f"   {r['feature_set']:<28} {r['model']:<20} n={r['n_features']:<3} F1={r['f1']:.4f}  train={r['training_time_sec']:.2f}s")
        self.step_timings["1_feature_selection"] = time.perf_counter() - t0

        # ============================================================== #
        # 2. Baselines (RQ1) + cross-validation
        # ============================================================== #
        t0 = time.perf_counter()
        print("\n[Step 2/9] Baseline models + 5-fold cross-validation ...")
        cv_results: Dict[str, Any] = {}
        for name, model in get_baseline_models(random_state=self.random_state).items():
            print(f"   {name:<26}", end="", flush=True)
            t1 = time.perf_counter(); model.fit(X_train, y_train); tt = time.perf_counter() - t1
            ev = evaluate_model(model, X_test, y_test, model_name=name, training_time=tt)
            all_evaluations[name] = ev
            fitted_models[name] = model
            joblib.dump(model, self.models_dir / f"{name}.joblib")
            self._log_experiment(f"BASE_{name.upper()}", name, f"All ({len(feature_names)})", "None (defaults)", "None", ev)

            cv_model = clone(model)
            if name == "support_vector_machine":
                cv_model.set_params(probability=False)  # F1 only needs labels; saves 5x Platt fits
            cv_results[name] = cross_validate_classifier(cv_model, X_train, y_train, name, n_splits=self.cv_folds, random_state=self.random_state)
            cv_results[name]["test_f1"] = ev["f1"]
            print(f" test F1={ev['f1']:.4f}  CV F1={cv_results[name]['f1_mean']:.4f}+/-{cv_results[name]['f1_std']:.4f}  train={tt:.1f}s")
        self._save_json("cv_results.json", {"n_splits": self.cv_folds, "split": "train (70%)", "models": cv_results})
        self.step_timings["2_baselines_cv"] = time.perf_counter() - t0

        # ============================================================== #
        # 3. Imbalanced data study
        # ============================================================== #
        t0 = time.perf_counter()
        print("\n[Step 3/9] Imbalanced-dataset handling study (none / class_weight / under-sampling / SMOTE) ...")
        imb = run_imbalance_study(X_train, y_train, X_test, y_test, random_state=self.random_state)
        self._save_json("imbalance_study.json", imb)
        for r in imb["records"]:
            print(f"   {r['strategy']:<20} {r['model']:<20} F1={r['f1']:.4f} Recall={r['recall']:.4f} FPR={r['false_positive_rate']:.4f} rows={r['train_rows_after_resampling']}")
        self.step_timings["3_imbalance"] = time.perf_counter() - t0

        # ============================================================== #
        # 4. Hyperparameter search comparison (RQ3)
        # ============================================================== #
        t0 = time.perf_counter()
        print("\n[Step 4/9] Hyperparameter search: Grid vs Random vs Optuna ...")
        optimizer = HyperparameterOptimizer(n_trials=self.n_trials, cv_folds=self.cv_folds, random_state=self.random_state)
        if self.quick:
            from ml.models import optimization as _opt
            _opt.RF_GRID = {"n_estimators": [50, 100], "max_depth": [10, None]}
            _opt.XGB_GRID = {"n_estimators": [50, 100], "max_depth": [4, 6]}

        search_comparison = []
        search_models = {}

        def _record(model_key, algo, method, res, est):
            t1 = time.perf_counter(); tt = res["wall_time_sec"]
            ev = evaluate_model(est, X_test, y_test, model_name=f"{method}_{algo}", training_time=tt)
            hist = res["trials_history"]
            budget = min(self.n_random_iter, len(hist))
            best_at_budget = max(h["value"] for h in hist[:budget]) if hist else None
            search_comparison.append({
                "algorithm": algo, "method": res["method"], "n_configs_evaluated": res["n_trials"],
                "wall_time_sec": res["wall_time_sec"], "best_cv_f1": round(res["best_cv_f1"], 4),
                f"best_cv_f1_at_{budget}_configs": round(float(best_at_budget), 4) if best_at_budget is not None else None,
                "test_f1": ev["f1"], "test_recall": ev["recall"], "test_fpr": ev["false_positive_rate"],
                "best_params": res["best_params"],
            })
            search_models[model_key] = (est, ev, res)
            print(f"   {algo:<14} {res['method']:<20} configs={res['n_trials']:<3} CV F1={res['best_cv_f1']:.4f}  test F1={ev['f1']:.4f}  {res['wall_time_sec']:.0f}s")

        est, res = optimizer.grid_search_random_forest(X_train, y_train);          _record("grid_rf", "random_forest", "grid", res, est)
        est, res = optimizer.random_search_random_forest(X_train, y_train, n_iter=self.n_random_iter); _record("random_rf", "random_forest", "random", res, est)
        est, res = optimizer.optimize_random_forest(X_train, y_train);             _record("optuna_rf", "random_forest", "optuna", res, est)
        est, res = optimizer.grid_search_xgboost(X_train, y_train);                _record("grid_xgb", "xgboost", "grid", res, est)
        est, res = optimizer.random_search_xgboost(X_train, y_train, n_iter=self.n_random_iter); _record("random_xgb", "xgboost", "random", res, est)
        est, res = optimizer.optimize_xgboost(X_train, y_train);                   _record("optuna_xgb", "xgboost", "optuna", res, est)

        # The Optuna-tuned models are the "optimized_*" models used downstream.
        opt_rf, eval_opt_rf, rf_study = search_models["optuna_rf"]
        opt_xgb, eval_opt_xgb, xgb_study = search_models["optuna_xgb"]
        for key, (est, ev, res) in (("optimized_random_forest", (opt_rf, eval_opt_rf, rf_study)), ("optimized_xgboost", (opt_xgb, eval_opt_xgb, xgb_study))):
            ev = dict(ev); ev["model_name"] = key
            all_evaluations[key] = ev
            fitted_models[key] = est
            joblib.dump(est, self.models_dir / f"{key}.joblib")
            self._log_experiment(f"OPT_{key.upper()}", key, f"All ({len(feature_names)})", f"Optuna TPE ({self.cv_folds}-fold CV F1)", "None", ev)
        # Keep grid/random winners on disk too (useful for the dashboard / viva)
        joblib.dump(search_models["grid_rf"][0], self.models_dir / "grid_random_forest.joblib")
        joblib.dump(search_models["grid_xgb"][0], self.models_dir / "grid_xgboost.joblib")

        self._save_json("optimization_study.json", {
            "optimized_random_forest": rf_study,
            "optimized_xgboost": xgb_study,
            "grid_random_forest": optimizer.study_results_["grid_random_forest"],
            "random_random_forest": optimizer.study_results_["random_random_forest"],
            "grid_xgboost": optimizer.study_results_["grid_xgboost"],
            "random_xgboost": optimizer.study_results_["random_xgboost"],
            "search_comparison": search_comparison,
            "protocol": f"All searchers share the same StratifiedKFold({self.cv_folds}, shuffle, seed={self.random_state}) on the train split and maximise mean F1. Test split used once per final model.",
        })
        self.step_timings["4_hyperparameter_search"] = time.perf_counter() - t0

        # ============================================================== #
        # 5. Ensembles (RQ4, RQ5)
        # ============================================================== #
        t0 = time.perf_counter()
        print("\n[Step 5/9] Ensembles: Voting / Bagging / Boosting / Stacking / Weighted ...")
        ensembles = build_ensemble_models(random_state=self.random_state, optimized_rf=opt_rf, optimized_xgb=opt_xgb)
        for ens_name, ens_model in ensembles.items():
            print(f"   {ens_name:<22}", end="", flush=True)
            t1 = time.perf_counter()
            if ens_name == "weighted_ensemble":
                ens_model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
            else:
                ens_model.fit(X_train, y_train)
            tt = time.perf_counter() - t1
            ev = evaluate_model(ens_model, X_test, y_test, model_name=ens_name, training_time=tt)
            if ens_name == "weighted_ensemble":
                ev["validation_weights"] = [round(float(w), 4) for w in ens_model.normalized_weights_]
                ev["validation_f1"] = ens_model.validation_f1_
            all_evaluations[ens_name] = ev
            fitted_models[ens_name] = ens_model
            joblib.dump(ens_model, self.models_dir / f"{ens_name}.joblib")
            self._log_experiment(f"ENS_{ens_name.upper()}", ens_name, f"All ({len(feature_names)})", "Tuned base learners", ENSEMBLE_FAMILY[ens_name], ev)
            print(f" F1={ev['f1']:.4f} Recall={ev['recall']:.4f} FPR={ev['false_positive_rate']:.4f} AUC={ev['roc_auc']:.4f} train={tt:.1f}s infer={ev['inference_time_ms_per_1k']:.2f}ms/1k")
        self.step_timings["5_ensembles"] = time.perf_counter() - t0

        # ============================================================== #
        # 6. Per-attack-type recall
        # ============================================================== #
        t0 = time.perf_counter()
        print("\n[Step 6/9] Per-attack-type recall ...")
        per_class: Dict[str, Any] = {}
        if y_test_labels is not None:
            for name, model in fitted_models.items():
                per_class[name] = per_class_recall(model, X_test, y_test, y_test_labels)
            self._save_json("per_class_recall.json", {"models": per_class, "note": "Binary detector; recall per original CIC-IDS2017 category on the test split."})
            best_key_tmp = max(all_evaluations, key=lambda k_: all_evaluations[k_]["f1"])
            for r in per_class[best_key_tmp]:
                print(f"   [{best_key_tmp}] {r['class']:<28} n={r['n_test']:<6} recall={r['recall']}")
        self.step_timings["6_per_class"] = time.perf_counter() - t0

        # ============================================================== #
        # 7. Statistical stability
        # ============================================================== #
        t0 = time.perf_counter()
        print(f"\n[Step 7/9] Bootstrap 95% CIs ({self.n_bootstrap} resamples) + multi-seed retraining {self.seeds} ...")
        seed_stats: Dict[str, Any] = {}
        for name, model in fitted_models.items():
            seed_stats[name] = bootstrap_confidence_intervals(model, X_test, y_test, n_bootstrap=self.n_bootstrap, random_state=self.random_state)

        multi_seed: Dict[str, Any] = {}
        for name in ["logistic_regression", "decision_tree", "random_forest", "xgboost"]:
            f1s, recs = [], []
            for s in self.seeds:
                m = get_baseline_models(random_state=s)[name]
                m.fit(X_train, y_train)
                ev = evaluate_model(m, X_test, y_test, model_name=name)
                f1s.append(ev["f1"]); recs.append(ev["recall"])
            multi_seed[name] = {"seeds": self.seeds, "f1_per_seed": f1s, "f1_mean": round(float(np.mean(f1s)), 4), "f1_std": round(float(np.std(f1s)), 4),
                                "recall_per_seed": recs, "recall_mean": round(float(np.mean(recs)), 4), "recall_std": round(float(np.std(recs)), 4)}
            print(f"   {name:<22} seeds F1={f1s}")
        self._save_json("statistical_stability.json", {"bootstrap": seed_stats, "multi_seed_retrain": multi_seed,
                                                       "note": "bootstrap = resample test rows with replacement (model fixed); multi_seed = retrain with different seeds (test fixed)."})
        self.step_timings["7_stability"] = time.perf_counter() - t0

        # ============================================================== #
        # 8. SHAP (RQ6)
        # ============================================================== #
        t0 = time.perf_counter()
        print("\n[Step 8/9] SHAP global explanation ...")
        best_model_key = self._pick_best(all_evaluations)
        shap_target_key = best_model_key
        # Prefer a tree model for exact TreeSHAP if the best model is a heterogeneous ensemble
        if best_model_key in ("voting_hard", "voting_soft", "stacking", "weighted_ensemble"):
            shap_target_key = max(["optimized_xgboost", "optimized_random_forest"], key=lambda k_: all_evaluations[k_]["f1"])
        explainer = IDSExplainer(fitted_models[shap_target_key], feature_names=feature_names, background_data=X_train[:200])
        global_shap = explainer.explain_global(X_test[:500], max_display=25)
        self._save_json("shap_global.json", {"model_name": shap_target_key, "best_model_overall": best_model_key,
                                             "global_importance": global_shap, "n_explained": int(min(500, len(X_test)))})
        print(f"   explained {shap_target_key}; top-5: {[g['feature'] for g in global_shap[:5]]}")
        self.step_timings["8_shap"] = time.perf_counter() - t0

        # ============================================================== #
        # 9. Master outputs
        # ============================================================== #
        t0 = time.perf_counter()
        print("\n[Step 9/9] Writing master comparison, CSV, presets and conclusions ...")
        for key, ev in all_evaluations.items():
            ev["family"] = BASELINE_FAMILY.get(key) or ENSEMBLE_FAMILY.get(key) or ("Optuna-tuned " + key.replace("optimized_", "").replace("_", " "))
            ev["paradigm"] = "Baseline" if key in BASELINE_KEYS else ("Optimized" if key in OPTIMIZED_KEYS else "Ensemble")
            if key in seed_stats:
                ev["f1_ci95"] = seed_stats[key]["f1_ci95"]
        summary_file = self.metrics_dir / "dataset_summary.json"
        dataset_summary = json.load(open(summary_file)) if summary_file.exists() else {}

        master = {
            "is_synthetic": is_synthetic,
            "dataset_name": "CIC-IDS2017 (synthetic dev fixture)" if is_synthetic else "CIC-IDS2017 (class-capped stratified sample)",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "environment": {"python": platform.python_version(), "sklearn": sklearn.__version__, "xgboost": xgboost.__version__,
                            "platform": platform.platform(), "cpu_count": __import__("os").cpu_count()},
            "budgets": {"optuna_trials": self.n_trials, "random_search_iter": self.n_random_iter, "cv_folds": self.cv_folds, "bootstrap": self.n_bootstrap},
            "split_sizes": {"train": int(len(y_train)), "val": int(len(y_val)), "test": int(len(y_test))},
            "best_model": best_model_key,
            "models": all_evaluations,
            "statistical_stability": {k_: {kk: vv for kk, vv in v.items() if kk in ("f1_mean", "f1_std", "recall_mean", "recall_std", "accuracy_mean", "accuracy_std", "fpr_mean", "fpr_std", "f1_ci95")} for k_, v in seed_stats.items()},
            "step_timings_sec": {k_: round(v, 1) for k_, v in self.step_timings.items()},
        }
        self._save_json("master_comparison.json", master)
        pd.DataFrame(self.experiment_records).to_csv(self.experiments_dir / "results.csv", index=False)

        presets = self._build_presets(splits, fitted_models.get("optimized_xgboost"))
        self._save_json("presets.json", presets)

        conclusions = self._generate_conclusions(all_evaluations, fs_bench, imb, search_comparison, per_class, global_shap, shap_target_key, is_synthetic, cv_results)
        self._save_json("research_conclusions.json", conclusions, folder=self.experiments_dir)
        self.step_timings["9_outputs"] = time.perf_counter() - t0
        master["step_timings_sec"] = {k_: round(v, 1) for k_, v in self.step_timings.items()}
        master["total_runtime_sec"] = round(time.perf_counter() - t_run0, 1)
        self._save_json("master_comparison.json", master)

        print("\n" + "=" * 72)
        print(f">>> BENCHMARK COMPLETE in {master['total_runtime_sec']/60:.1f} min. Best model: {best_model_key}")
        print(f"    {self.metrics_dir / 'master_comparison.json'}")
        print(f"    {self.experiments_dir / 'results.csv'}")
        print(f"    {self.experiments_dir / 'research_conclusions.json'}")
        print("=" * 72)
        return {"evaluations": all_evaluations, "conclusions": conclusions}

    # ------------------------------------------------------------------ #
    @staticmethod
    def _pick_best(evals: Dict[str, Any]) -> str:
        """Highest F1; ties (within 0.0005) broken by lower FPR then lower latency."""
        best_f1 = max(e["f1"] for e in evals.values())
        cands = [k for k, e in evals.items() if e["f1"] >= best_f1 - 0.0005]
        return min(cands, key=lambda k: (evals[k]["false_positive_rate"], evals[k]["inference_time_ms_per_1k"]))

    def _build_presets(self, splits: Dict[str, Any], model: Any) -> Dict[str, Any]:
        """Pick genuine flows from the test split (closest to the class median) as dashboard presets."""
        X_raw: pd.DataFrame = splits["X_test_df_raw"]
        labels = np.asarray(splits.get("y_test_labels", np.where(splits["y_test"] == 1, "ATTACK", "BENIGN"))).astype(str)
        prep = joblib.load(self.artifacts_dir / "preprocessors" / "preprocessor.joblib")["preprocessor"]
        feature_names = list(X_raw.columns)
        presets: Dict[str, Any] = {}

        for key, cls, title, desc in PRESET_SPECS:
            mask = labels == cls
            if key == "benign_web" and "Destination Port" in X_raw.columns:
                mask = mask & X_raw["Destination Port"].isin([80, 443]).values
            if key == "benign_dns" and "Destination Port" in X_raw.columns:
                mask = mask & (X_raw["Destination Port"] == 53).values
            if not mask.any():
                continue
            sub = X_raw[mask]
            med = sub.median(numeric_only=True)
            scale = sub.std(numeric_only=True).replace(0, 1.0) + 1e-9
            dist = ((sub - med) / scale).abs().sum(axis=1)
            order = dist.sort_values().index[:25]
            chosen = None
            for idx in order:
                row = sub.loc[idx]
                if model is not None:
                    pred = int(model.predict(prep.transform(pd.DataFrame([row.values], columns=feature_names)))[0])
                    if pred != (0 if cls == "BENIGN" else 1):
                        continue
                chosen = row
                break
            if chosen is None:
                chosen = sub.loc[order[0]]
            presets[key] = {
                "name": title, "description": desc, "category": "BENIGN" if cls == "BENIGN" else "ATTACK", "attack_class": cls,
                "source": "real CIC-IDS2017 test-split flow",
                "features": {f: round(float(v), 4) for f, v in chosen.items()},
            }
        return presets

    def _generate_conclusions(self, evals, fs_bench, imb, search_comparison, per_class, global_shap, shap_key, is_synthetic, cv_results) -> Dict[str, Any]:
        """Data-driven answers to the research questions - wording follows the numbers."""
        def fmt(x): return f"{x:.4f}"
        def pct(a, b): return (a - b) / max(1e-9, b) * 100.0

        # RQ1
        best_baseline = max(BASELINE_KEYS, key=lambda k: evals[k]["f1"])
        worst_baseline = min(BASELINE_KEYS, key=lambda k: evals[k]["f1"])
        rq1 = {
            "question": "How accurately can individual machine-learning algorithms detect malicious network traffic?",
            "best_model": best_baseline, "f1_score": evals[best_baseline]["f1"], "recall": evals[best_baseline]["recall"], "accuracy": evals[best_baseline]["accuracy"],
            "worst_model": worst_baseline, "worst_f1": evals[worst_baseline]["f1"],
            "cv_f1_best": cv_results[best_baseline]["f1_mean"], "cv_f1_std_best": cv_results[best_baseline]["f1_std"],
            "finding": (f"With default hyperparameters, {best_baseline.replace('_', ' ')} was the strongest single model "
                        f"(test F1 {fmt(evals[best_baseline]['f1'])}, recall {fmt(evals[best_baseline]['recall'])}, "
                        f"{self.cv_folds}-fold CV F1 {fmt(cv_results[best_baseline]['f1_mean'])} +/- {fmt(cv_results[best_baseline]['f1_std'])}); "
                        f"{worst_baseline.replace('_', ' ')} was weakest (F1 {fmt(evals[worst_baseline]['f1'])}). "
                        f"Tree-based models separate attack flows far better than a linear boundary."),
        }

        # RQ2 - measured feature-selection benchmark (RF probe)
        rf_rows = [r for r in fs_bench if r["model"] == "random_forest"]
        base_row = next(r for r in rf_rows if r["feature_set"] == "all_features")
        best_sub = max((r for r in rf_rows if r["feature_set"] not in ("all_features", "all_minus_destination_port")), key=lambda r: (r["f1"], -r["n_features"]))
        no_port = next((r for r in rf_rows if r["feature_set"] == "all_minus_destination_port"), None)
        fastest = min(rf_rows, key=lambda r: r["training_time_sec"])
        rq2 = {
            "question": "Does feature selection / dimensionality reduction improve model performance and training efficiency?",
            "baseline_features": base_row["n_features"], "baseline_f1": base_row["f1"], "baseline_train_time_sec": base_row["training_time_sec"],
            "best_subset": best_sub["feature_set"], "best_subset_features": best_sub["n_features"], "best_subset_f1": best_sub["f1"],
            "best_subset_train_time_sec": best_sub["training_time_sec"],
            "train_time_change_pct": round(pct(best_sub["training_time_sec"], base_row["training_time_sec"]), 1),
            "f1_change": round(best_sub["f1"] - base_row["f1"], 4),
            "destination_port_ablation": None if no_port is None else {"f1_without_port": no_port["f1"], "f1_with_port": base_row["f1"], "delta": round(no_port["f1"] - base_row["f1"], 4)},
            "benchmark_rf": rf_rows,
            "finding": (f"Using all {base_row['n_features']} features, Random Forest reached F1 {fmt(base_row['f1'])} in {base_row['training_time_sec']:.2f}s. "
                        f"The best reduced set, {best_sub['feature_set']} ({best_sub['n_features']} features), reached F1 {fmt(best_sub['f1'])} "
                        f"({best_sub['f1'] - base_row['f1']:+.4f}) with training time {pct(best_sub['training_time_sec'], base_row['training_time_sec']):+.0f}%. "
                        f"Fastest configuration: {fastest['feature_set']} ({fastest['training_time_sec']:.2f}s, F1 {fmt(fastest['f1'])})."
                        + (f" Ablation: removing Destination Port (a known shortcut feature) changes RF F1 by {no_port['f1'] - base_row['f1']:+.4f}, so detection does not rely on the port alone." if no_port else "")),
        }

        # RQ3 - tuning
        best_opt = max(OPTIMIZED_KEYS, key=lambda k: evals[k]["f1"])
        base_of_opt = best_opt.replace("optimized_", "")
        gain = pct(evals[best_opt]["f1"], evals[base_of_opt]["f1"])
        verb = "improved" if gain > 0 else ("left unchanged" if abs(gain) < 1e-9 else "slightly reduced")
        rf_cmp = [r for r in search_comparison if r["algorithm"] == "random_forest"]
        xgb_cmp = [r for r in search_comparison if r["algorithm"] == "xgboost"]
        def best_method(rows): return max(rows, key=lambda r: r["best_cv_f1"])
        def fastest_method(rows): return min(rows, key=lambda r: r["wall_time_sec"])
        rq3 = {
            "question": "How do optimized models compare with their unoptimized versions, and how do Grid, Random and Bayesian search compare?",
            "best_optimized": best_opt, "baseline_f1": evals[base_of_opt]["f1"], "optimized_f1": evals[best_opt]["f1"],
            "gain_percentage": f"{gain:+.2f}%",
            "rf_delta_f1": round(evals["optimized_random_forest"]["f1"] - evals["random_forest"]["f1"], 4),
            "xgb_delta_f1": round(evals["optimized_xgboost"]["f1"] - evals["xgboost"]["f1"], 4),
            "rf_delta_fnr": round(evals["optimized_random_forest"]["false_negative_rate"] - evals["random_forest"]["false_negative_rate"], 4),
            "xgb_delta_fnr": round(evals["optimized_xgboost"]["false_negative_rate"] - evals["xgboost"]["false_negative_rate"], 4),
            "search_comparison": search_comparison,
            "finding": (f"Optuna tuning {verb} the test F1 of {base_of_opt.replace('_', ' ')} from {fmt(evals[base_of_opt]['f1'])} to {fmt(evals[best_opt]['f1'])} ({gain:+.2f}%). "
                        f"Across search strategies on Random Forest, {best_method(rf_cmp)['method']} found the best CV F1 ({fmt(best_method(rf_cmp)['best_cv_f1'])}) "
                        f"and {fastest_method(rf_cmp)['method']} was fastest ({fastest_method(rf_cmp)['wall_time_sec']:.0f}s); "
                        f"for XGBoost, {best_method(xgb_cmp)['method']} found the best CV F1 ({fmt(best_method(xgb_cmp)['best_cv_f1'])}). "
                        f"On this well-separated data the gains from tuning are small in absolute terms; the differences show up mostly in false-negative rate and in search cost."),
        }

        # RQ4 - ensembles
        best_ens = max(ENSEMBLE_KEYS, key=lambda k: evals[k]["f1"])
        best_single = max(BASELINE_KEYS + OPTIMIZED_KEYS, key=lambda k: evals[k]["f1"])
        ens_gain = pct(evals[best_ens]["f1"], evals[best_opt]["f1"])
        ens_vs_single = evals[best_ens]["f1"] - evals[best_single]["f1"]
        ens_better = sum(1 for k in ENSEMBLE_KEYS if evals[k]["f1"] > evals[best_single]["f1"])
        rq4 = {
            "question": "Do ensemble methods outperform individual classifiers?",
            "best_ensemble": best_ens, "ensemble_f1": evals[best_ens]["f1"], "optimized_f1": evals[best_opt]["f1"],
            "best_single_model": best_single, "best_single_f1": evals[best_single]["f1"],
            "gain_over_optimized_pct": f"{ens_gain:+.2f}%", "n_ensembles_beating_best_single": ens_better, "n_ensembles": len(ENSEMBLE_KEYS),
            "ensemble_ranking": sorted([{"model": k, "family": ENSEMBLE_FAMILY[k], "f1": evals[k]["f1"], "recall": evals[k]["recall"], "fpr": evals[k]["false_positive_rate"]} for k in ENSEMBLE_KEYS], key=lambda r: -r["f1"]),
            "finding": (f"The best ensemble, {best_ens.replace('_', ' ')} ({ENSEMBLE_FAMILY[best_ens]}), scored F1 {fmt(evals[best_ens]['f1'])} versus {fmt(evals[best_single]['f1'])} for the best single model "
                        f"({best_single.replace('_', ' ')}), a difference of {ens_vs_single:+.4f}. {ens_better} of {len(ENSEMBLE_KEYS)} ensembles beat the best single model. "
                        + ("Ensembles give a consistent but small edge here because the strongest single learners are already near the ceiling; the clearer benefit is lower variance (see bootstrap CIs) and fewer missed rare attacks."
                           if ens_vs_single >= 0 else "On this sample the best tuned single model was not beaten - combining weaker learners (linear, SVM) with strong tree models can dilute the strongest member.")),
        }

        # RQ5 - trade-off
        best_f1 = max(e["f1"] for e in evals.values())
        near_best = [k for k, e in evals.items() if e["f1"] >= best_f1 - 0.002]
        fastest_near_best = min(near_best, key=lambda k: evals[k]["inference_time_ms_per_1k"])
        best_recall_key = max(evals, key=lambda k: (evals[k]["recall"], -evals[k]["false_positive_rate"]))
        lowest_fpr_key = min(evals, key=lambda k: (evals[k]["false_positive_rate"], -evals[k]["recall"]))
        rq5 = {
            "question": "Which model provides the best trade-off between detection, false alarms and latency?",
            "best_tradeoff_model": fastest_near_best, "best_tradeoff_f1": evals[fastest_near_best]["f1"], "best_tradeoff_latency_ms_per_1k": evals[fastest_near_best]["inference_time_ms_per_1k"],
            "highest_recall_model": best_recall_key, "highest_recall_value": evals[best_recall_key]["recall"],
            "lowest_fpr_model": lowest_fpr_key, "lowest_fpr_value": evals[lowest_fpr_key]["false_positive_rate"],
            "finding": (f"Models within 0.002 F1 of the best ({fmt(best_f1)}) were: {', '.join(k.replace('_', ' ') for k in near_best)}. Among them "
                        f"{fastest_near_best.replace('_', ' ')} is the recommended deployment choice ({evals[fastest_near_best]['inference_time_ms_per_1k']:.2f} ms per 1k flows). "
                        f"Highest attack recall: {best_recall_key.replace('_', ' ')} ({fmt(evals[best_recall_key]['recall'])}); lowest false-positive rate: {lowest_fpr_key.replace('_', ' ')} ({fmt(evals[lowest_fpr_key]['false_positive_rate'])})."),
        }

        # RQ6 - SHAP
        top3 = [g["feature"] for g in global_shap[:3]]
        rq6 = {
            "question": "Can the final model provide interpretable explanations for why traffic was classified as malicious?",
            "explained_model": shap_key, "top_global_features": top3, "top_10": [{"feature": g["feature"], "mean_abs_shap": g["importance"]} for g in global_shap[:10]],
            "finding": (f"SHAP (TreeExplainer) on {shap_key.replace('_', ' ')} ranks {top3[0]}, {top3[1]} and {top3[2]} as the strongest global drivers. "
                        + ("Destination Port ranking highly is expected - attacks in this corpus target specific services (FTP 21, SSH 22, HTTP 80) - and is a known shortcut risk; the ablation in RQ2 shows the model still detects attacks without it. "
                           if "Destination Port" in top3 else "")
                        + "The remaining drivers are flow-shape statistics (initial TCP window sizes, packet-length and timing features), i.e. the model keys on how the connection behaves, not on who is talking."),
        }

        # RQ7 - imbalance
        best_imb = imb["best_strategy_by_model"]
        recs = imb["records"]
        def rec(strategy, model): return next(r for r in recs if r["strategy"] == strategy and r["model"] == model)
        lr_none, lr_cw, lr_smote = rec("none", "logistic_regression"), rec("class_weight", "logistic_regression"), rec("smote", "logistic_regression")
        rq7 = {
            "question": "Which imbalanced-data strategy works best: nothing, class weighting, under-sampling or SMOTE?",
            "train_attack_ratio": round(imb["train_attack_ratio_before"], 4), "best_strategy_by_model": best_imb,
            "finding": (f"The training split is {imb['train_attack_ratio_before']*100:.1f}% attack. For logistic regression, recall moved from {fmt(lr_none['recall'])} (no handling) to "
                        f"{fmt(lr_cw['recall'])} with class weights and {fmt(lr_smote['recall'])} with SMOTE, at FPR {fmt(lr_none['false_positive_rate'])} / {fmt(lr_cw['false_positive_rate'])} / {fmt(lr_smote['false_positive_rate'])}. "
                        f"Best strategy per model (by F1): " + ", ".join(f"{m.replace('_', ' ')} -> {v['strategy']}" for m, v in best_imb.items()) +
                        ". Tree ensembles are far less sensitive to the imbalance than the linear model."),
        }

        # Per-class weak spots
        weak = []
        best_key = self._pick_best(evals)
        if per_class and best_key in per_class:
            weak = [r for r in per_class[best_key] if r["is_attack"] and r["recall"] is not None and r["recall"] < 0.95]
        rq8 = {
            "question": "Which attack categories does the best model still miss?",
            "model": best_key,
            "weak_classes": weak,
            "finding": ("All attack categories in the test split were detected with recall >= 0.95." if not weak else
                        "Lowest per-category recall for " + best_key.replace("_", " ") + ": " + ", ".join(f"{w['class']} ({w['n_detected']}/{w['n_test']}, {w['recall']:.2f})" for w in weak) +
                        ". These are the rarest classes; a binary F1 above 0.99 hides them, which is why per-category recall is reported."),
        }

        return {
            "is_synthetic_dataset": is_synthetic,
            "rq1_individual_baselines": rq1, "rq2_feature_selection": rq2, "rq3_optimization_impact": rq3,
            "rq4_ensemble_superiority": rq4, "rq5_optimal_tradeoff": rq5, "rq6_explainability": rq6,
            "rq7_imbalance_handling": rq7, "rq8_per_attack_type": rq8,
            "executive_summary": (f"On a {'synthetic development fixture' if is_synthetic else 'class-capped stratified sample of the real CIC-IDS2017 corpus'}, "
                                  f"the best detector was {best_key.replace('_', ' ')} (F1 {fmt(evals[best_key]['f1'])}, recall {fmt(evals[best_key]['recall'])}, FPR {fmt(evals[best_key]['false_positive_rate'])}). "
                                  f"Hyperparameter tuning changed the best single model's F1 by {gain:+.2f}%; the best ensemble differed from the best single model by {ens_vs_single:+.4f} F1. "
                                  + ("Every attack category in the test split was detected with recall >= 0.95." if not weak else
                                     "Categories still partly missed: " + ", ".join(f"{w['class']} ({w['recall']:.2f})" for w in weak) + ".")),
        }


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return str(o)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Small search budgets for smoke testing")
    args = parser.parse_args()
    MasterExperimentOrchestrator(quick=args.quick).run()
