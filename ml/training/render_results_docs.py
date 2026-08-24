"""
Render docs/results.md (and the README leaderboard block) from the JSON
artifacts, so the numbers in the documentation can never drift from the
numbers the experiments actually produced.

    python -m ml.training.render_results_docs
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List

ML_DIR = Path(__file__).resolve().parent.parent
ROOT = ML_DIR.parent
METRICS = ML_DIR / "artifacts" / "metrics"
EXPERIMENTS = ML_DIR / "artifacts" / "experiments"
DOCS = ROOT / "docs"

PRETTY = {
    "logistic_regression": "Logistic Regression", "decision_tree": "Decision Tree", "random_forest": "Random Forest",
    "extra_trees": "Extra Trees", "support_vector_machine": "SVM (RBF)", "k_nearest_neighbors": "KNN (k=5)", "xgboost": "XGBoost",
    "optimized_random_forest": "Random Forest (Optuna)", "optimized_xgboost": "XGBoost (Optuna)",
    "voting_hard": "Voting (hard)", "voting_soft": "Voting (soft)", "bagging_extra_trees": "Bagging (Extra Trees)",
    "adaboost": "AdaBoost", "gradient_boosting": "Gradient Boosting", "stacking": "Stacking", "weighted_ensemble": "Weighted Voting",
}


def _load(name: str, folder: Path = METRICS) -> Any:
    with open(folder / name, "r", encoding="utf-8") as f:
        return json.load(f)


def p(k: str) -> str:
    return PRETTY.get(k, k.replace("_", " ").title())


def pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def leaderboard_table(master: Dict[str, Any]) -> str:
    models = master["models"]
    keys = sorted(models, key=lambda k: (-models[k]["f1"], models[k]["false_positive_rate"], models[k]["inference_time_ms_per_1k"]))
    best = master["best_model"]
    lines = ["| # | Model | Paradigm | Accuracy | Precision | Recall | F1 | F1 95% CI | ROC-AUC | FPR | FNR | Train (s) | Infer (ms/1k) |",
             "|:-:|:--|:--|--:|--:|--:|--:|:--|--:|--:|--:|--:|--:|"]
    for i, k in enumerate(keys, 1):
        e = models[k]
        ci = e.get("f1_ci95")
        ci_s = f"{ci[0]:.4f}–{ci[1]:.4f}" if ci else "—"
        name = f"**{p(k)}** ★" if k == best else p(k)
        lines.append(f"| {i} | {name} | {e['paradigm']} | {pct(e['accuracy'])} | {pct(e['precision'])} | {pct(e['recall'])} | **{e['f1']:.4f}** | {ci_s} | {e['roc_auc']:.4f} | {e['false_positive_rate']:.4f} | {e['false_negative_rate']:.4f} | {e['training_time_sec']:.2f} | {e['inference_time_ms_per_1k']:.2f} |")
    return "\n".join(lines)


def cv_table(cv: Dict[str, Any]) -> str:
    models = cv["models"]
    keys = sorted(models, key=lambda k: -models[k]["f1_mean"])
    lines = [f"| Model | CV F1 (mean ± std over {cv['n_splits']} folds) | CV Recall | CV Precision | Train-fold F1 | Overfit gap | Test F1 |", "|:--|--:|--:|--:|--:|--:|--:|"]
    for k in keys:
        m = models[k]
        lines.append(f"| {p(k)} | {m['f1_mean']:.4f} ± {m['f1_std']:.4f} | {m['recall_mean']:.4f} ± {m['recall_std']:.4f} | {m['precision_mean']:.4f} ± {m['precision_std']:.4f} | {m['train_f1_mean']:.4f} | {m['overfit_gap_f1']:+.4f} | {m['test_f1']:.4f} |")
    return "\n".join(lines)


def fs_table(fs: Dict[str, Any]) -> str:
    lines = ["| Feature set | # features | Model | F1 | Recall | ROC-AUC | Train (s) | Infer (ms/1k) |", "|:--|--:|:--|--:|--:|--:|--:|--:|"]
    for r in fs["benchmark"]:
        lines.append(f"| {r['feature_set']} | {r['n_features']} | {p(r['model'])} | {r['f1']:.4f} | {r['recall']:.4f} | {r['roc_auc']:.4f} | {r['training_time_sec']:.2f} | {r['inference_time_ms_per_1k']:.2f} |")
    return "\n".join(lines)


def search_table(opt: Dict[str, Any]) -> str:
    lines = ["| Algorithm | Search method | Configs evaluated | Wall time (s) | Best CV F1 | Test F1 | Test Recall | Test FPR | Best params |", "|:--|:--|--:|--:|--:|--:|--:|--:|:--|"]
    for r in opt["search_comparison"]:
        params = ", ".join(f"{k}={v}" for k, v in r["best_params"].items())
        lines.append(f"| {p(r['algorithm'])} | {r['method']} | {r['n_configs_evaluated']} | {r['wall_time_sec']:.0f} | {r['best_cv_f1']:.4f} | {r['test_f1']:.4f} | {r['test_recall']:.4f} | {r['test_fpr']:.4f} | `{params}` |")
    return "\n".join(lines)


def imbalance_table(imb: Dict[str, Any]) -> str:
    lines = ["| Strategy | Model | Train rows | Recall | Precision | F1 | FPR | FNR |", "|:--|:--|--:|--:|--:|--:|--:|--:|"]
    for r in imb["records"]:
        lines.append(f"| {r['strategy']} | {p(r['model'])} | {r['train_rows_after_resampling']:,} | {r['recall']:.4f} | {r['precision']:.4f} | {r['f1']:.4f} | {r['false_positive_rate']:.4f} | {r['false_negative_rate']:.4f} |")
    return "\n".join(lines)


def per_class_table(pcr: Dict[str, Any], master: Dict[str, Any]) -> str:
    models = pcr["models"]
    show = [k for k in ["logistic_regression", "support_vector_machine", "k_nearest_neighbors", "random_forest", "optimized_xgboost", "voting_soft", "stacking", master["best_model"]] if k in models]
    show = list(dict.fromkeys(show))
    classes = [r for r in models[show[0]]]
    lines = ["| Attack category | n (test) | " + " | ".join(p(k) for k in show) + " |", "|:--|--:|" + "--:|" * len(show)]
    for c in classes:
        row = [f"| {c['class']} | {c['n_test']} |"]
        for k in show:
            r = next(x for x in models[k] if x["class"] == c["class"])
            row.append(f" {r['recall']:.3f} |" if r["recall"] is not None else " — |")
        lines.append("".join(row))
    return "\n".join(lines)


def stability_table(stab: Dict[str, Any]) -> str:
    lines = ["| Model | Seeds | F1 per seed | F1 mean ± std | Recall mean ± std |", "|:--|:--|:--|--:|--:|"]
    for k, m in stab["multi_seed_retrain"].items():
        lines.append(f"| {p(k)} | {m['seeds']} | {m['f1_per_seed']} | {m['f1_mean']:.4f} ± {m['f1_std']:.4f} | {m['recall_mean']:.4f} ± {m['recall_std']:.4f} |")
    return "\n".join(lines)


def dataset_block(summary: Dict[str, Any]) -> str:
    info = summary["dataset_info"]; man = info.get("sampling_manifest") or {}; split = summary["split_info"]; clean = summary["cleaning_audit"]
    lines = []
    if man:
        lines.append(f"- **Source corpus**: {man['source']} — {man['full_corpus_rows']:,} flows, {len(man['full_corpus_class_distribution'])} classes.")
        s = man["sampling_strategy"]
        lines.append(f"- **Sample**: class-capped stratified sample — BENIGN capped at {s['benign_rows']:,}, every attack class capped at {s['attack_cap_per_class']:,} (rare classes kept in full), seed {s['random_state']} → {man['sample_rows']:,} rows, {man['sample_attack_ratio']*100:.1f}% attack.")
    lines.append(f"- **Cleaning**: {clean['duplicates_removed']:,} exact-duplicate rows removed, {clean['nan_inf_cells_found']:,} Inf/NaN cells imputed (median, train-fitted), {len(clean['constant_columns_dropped'])} constant columns dropped → {summary['num_features']} features.")
    lines.append(f"- **Splits** (stratified on attack category): train {split['train_count']:,} / val {split['val_count']:,} / test {split['test_count']:,}; attack ratio {split['train_attack_ratio']*100:.1f}% / {split['val_attack_ratio']*100:.1f}% / {split['test_attack_ratio']*100:.1f}%.")
    lines.append("")
    lines.append("| Class | Full corpus | Sample | Train | Val | Test |")
    lines.append("|:--|--:|--:|--:|--:|--:|")
    full = man.get("full_corpus_class_distribution", {}); samp = man.get("sample_class_distribution", info["class_distribution"])
    for cls in sorted(samp, key=lambda c: -samp[c]):
        lines.append(f"| {cls} | {full.get(cls, '—'):,} | {samp[cls]:,} | {split['train_class_counts'].get(cls, 0):,} | {split['val_class_counts'].get(cls, 0):,} | {split['test_class_counts'].get(cls, 0):,} |")
    return "\n".join(lines)


def render_results_md() -> str:
    master = _load("master_comparison.json"); summary = _load("dataset_summary.json"); cv = _load("cv_results.json")
    fs = _load("feature_selection_summary.json"); opt = _load("optimization_study.json"); imb = _load("imbalance_study.json")
    stab = _load("statistical_stability.json"); shap_d = _load("shap_global.json"); pca = _load("pca_summary.json")
    concl = _load("research_conclusions.json", EXPERIMENTS)
    pcr = _load("per_class_recall.json") if (METRICS / "per_class_recall.json").exists() else None
    env = master["environment"]; b = master["budgets"]

    out = [
        "# Results",
        "",
        f"> Auto-generated by `python -m ml.training.render_results_docs` from the JSON artifacts in `ml/artifacts/` on {master['timestamp']}. "
        f"Do not edit by hand — re-run the experiments and re-render.",
        "",
        f"**Dataset**: {master['dataset_name']} · **Best model**: {p(master['best_model'])} · **Total runtime**: {master.get('total_runtime_sec', 0)/60:.1f} min · "
        f"Python {env['python']}, scikit-learn {env['sklearn']}, XGBoost {env['xgboost']}, {env['cpu_count']} CPUs.",
        "",
        "## 1. Dataset and splits", "", dataset_block(summary), "",
        "![class distribution](figures/01_class_distribution.png)", "",
        "## 2. Leaderboard (untouched test split)", "",
        "Sorted by F1; ties broken by lower FPR then lower latency. The 95% CI is a 200-resample bootstrap of the test rows (model fixed).", "",
        leaderboard_table(master), "",
        "![leaderboard](figures/02_model_leaderboard_f1.png)", "",
        "![recall vs fpr](figures/03_recall_vs_fpr.png)", "",
        "![roc pr](figures/04_roc_pr_curves.png)", "",
        "![confusion](figures/05_confusion_matrices.png)", "",
        "## 3. Cross-validation (baselines, train split)", "",
        "Stratified k-fold on the 70% train split only. The overfit gap is train-fold F1 minus validation-fold F1: a large gap means the model memorises.", "",
        cv_table(cv), "",
        "![cv](figures/06_cross_validation_boxplot.png)", "",
        "## 4. Feature selection and PCA (RQ2)", "",
        f"PCA: {pca['n_components_retained']} components explain {pca['total_variance_explained']*100:.1f}% of variance (fitted on train only).", "",
        fs_table(fs), "",
        f"**Finding**: {concl['rq2_feature_selection']['finding']}", "",
        "![fs](figures/07_feature_selection_benchmark.png)", "",
        "![pca](figures/08_pca_cumulative_variance.png)", "",
        "![corr](figures/15_feature_correlation_heatmap.png)", "",
        "## 5. Hyperparameter search: Grid vs Random vs Bayesian (RQ3)", "",
        f"{opt['protocol']}", "",
        search_table(opt), "",
        f"**Finding**: {concl['rq3_optimization_impact']['finding']}", "",
        "![search](figures/09_grid_vs_random_vs_optuna.png)", "",
        "![optuna](figures/10_optuna_convergence.png)", "",
        "## 6. Imbalanced-data handling (RQ7)", "",
        imbalance_table(imb), "",
        f"**Finding**: {concl['rq7_imbalance_handling']['finding']}", "",
        "![imbalance](figures/11_imbalance_strategies.png)", "",
        "## 7. Ensembles (RQ4, RQ5)", "",
        f"**RQ4**: {concl['rq4_ensemble_superiority']['finding']}", "",
        f"**RQ5**: {concl['rq5_optimal_tradeoff']['finding']}", "",
        "![latency](figures/14_inference_latency.png)", "",
    ]
    if pcr:
        out += ["## 8. Per-attack-type recall (RQ8)", "", "A binary F1 above 0.99 can hide a completely missed rare class. This table shows the detection rate for each original CIC-IDS2017 category.", "",
                per_class_table(pcr, master), "", f"**Finding**: {concl['rq8_per_attack_type']['finding']}", "", "![per class](figures/12_per_attack_type_recall.png)", ""]
    out += [
        "## 9. Statistical stability", "",
        "Bootstrap CIs are in the leaderboard. Multi-seed retraining (train split fixed, model seed varied):", "",
        stability_table(stab), "",
        "## 10. Explainability (RQ6)", "",
        f"SHAP TreeExplainer on **{p(shap_d['model_name'])}**, {shap_d['n_explained']} test flows. Top-10 mean |SHAP|:", "",
        "| Rank | Feature | mean abs SHAP |", "|--:|:--|--:|",
        *[f"| {g['rank']} | {g['feature']} | {g['importance']:.4f} |" for g in shap_d["global_importance"][:10]], "",
        f"**Finding**: {concl['rq6_explainability']['finding']}", "",
        "![shap](figures/13_shap_global_importance.png)", "",
        "## 11. Executive summary", "", concl["executive_summary"], "",
        "## 12. Step timings", "",
        "| Step | Seconds |", "|:--|--:|",
        *[f"| {k} | {v:.0f} |" for k, v in master.get("step_timings_sec", {}).items()], "",
        f"Budgets: Optuna {b['optuna_trials']} trials, RandomizedSearch {b['random_search_iter']} iterations, {b['cv_folds']}-fold CV, {b['bootstrap']} bootstrap resamples.",
    ]
    return "\n".join(out) + "\n"


def update_readme_leaderboard(master: Dict[str, Any]):
    readme = ROOT / "README.md"
    if not readme.exists():
        return
    text = readme.read_text(encoding="utf-8")
    start, end = "<!-- LEADERBOARD:START -->", "<!-- LEADERBOARD:END -->"
    if start not in text or end not in text:
        return
    models = master["models"]
    keys = sorted(models, key=lambda k: (-models[k]["f1"], models[k]["false_positive_rate"]))
    lines = ["| Model | Paradigm | Accuracy | Recall | Precision | F1 | ROC-AUC | FPR | Infer (ms/1k) |", "|:--|:--|--:|--:|--:|--:|--:|--:|--:|"]
    for k in keys:
        e = models[k]
        name = f"**{p(k)}** ★" if k == master["best_model"] else p(k)
        lines.append(f"| {name} | {e['paradigm']} | {pct(e['accuracy'])} | {pct(e['recall'])} | {pct(e['precision'])} | **{e['f1']:.4f}** | {e['roc_auc']:.4f} | {e['false_positive_rate']:.4f} | {e['inference_time_ms_per_1k']:.2f} |")
    block = f"{start}\n_Generated {master['timestamp']} from `ml/artifacts/metrics/master_comparison.json` — {master['dataset_name']}, test split n={master['split_sizes']['test']:,}._\n\n" + "\n".join(lines) + f"\n{end}"
    new_text = re.sub(re.escape(start) + r".*?" + re.escape(end), lambda m: block, text, flags=re.S)
    readme.write_text(new_text, encoding="utf-8")
    print(f"  updated leaderboard block in {readme.relative_to(ROOT)}")


def main():
    DOCS.mkdir(exist_ok=True)
    md = render_results_md()
    (DOCS / "results.md").write_text(md, encoding="utf-8")
    print(f"  wrote docs/results.md ({len(md.splitlines())} lines)")
    update_readme_leaderboard(_load("master_comparison.json"))


if __name__ == "__main__":
    main()
