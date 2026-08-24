"""
Figure generator - turns the JSON artifacts written by the experiment runner
into the PNG charts used in the docs and the report.

    python -m ml.training.generate_figures

Every figure is produced from artifacts on disk; nothing is typed in by hand.
Charts follow a few fixed rules: one axis per chart, categorical colours in a
fixed colour-blind-safe order (Okabe-Ito), a single hue for magnitudes, thin
marks, recessive grids and direct labels where they help.
"""

import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ML_DIR = Path(__file__).resolve().parent.parent
METRICS = ML_DIR / "artifacts" / "metrics"
EXPERIMENTS = ML_DIR / "artifacts" / "experiments"
FIG_DIR = ML_DIR.parent / "docs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Okabe-Ito, ordered so adjacent hues stay separable under CVD
CAT = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442", "#999999"]
INK, INK2, GRID = "#1f2933", "#52606d", "#e4e7eb"
PARADIGM_COLOR = {"Baseline": CAT[0], "Optimized": CAT[1], "Ensemble": CAT[2]}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "figure.dpi": 150, "savefig.dpi": 150, "savefig.bbox": "tight", "figure.facecolor": "white",
})

PRETTY = {
    "logistic_regression": "Logistic Regression", "decision_tree": "Decision Tree", "random_forest": "Random Forest",
    "extra_trees": "Extra Trees", "support_vector_machine": "SVM (RBF)", "k_nearest_neighbors": "KNN (k=5)", "xgboost": "XGBoost",
    "optimized_random_forest": "RF (Optuna)", "optimized_xgboost": "XGBoost (Optuna)",
    "voting_hard": "Voting (hard)", "voting_soft": "Voting (soft)", "bagging_extra_trees": "Bagging (Extra Trees)",
    "adaboost": "AdaBoost", "gradient_boosting": "Gradient Boosting", "stacking": "Stacking", "weighted_ensemble": "Weighted Voting",
}


def _load(name: str, folder: Path = METRICS) -> Any:
    with open(folder / name, "r") as f:
        return json.load(f)


def _save(fig, name: str):
    out = FIG_DIR / name
    fig.savefig(out)
    plt.close(fig)
    print(f"  wrote {out.relative_to(ML_DIR.parent)}")


def _pretty(k: str) -> str:
    return PRETTY.get(k, k.replace("_", " ").title())


# --------------------------------------------------------------------------- #
def fig_class_distribution(summary: Dict[str, Any]):
    dist = summary["dataset_info"]["class_distribution"]
    manifest = summary["dataset_info"].get("sampling_manifest") or {}
    full = manifest.get("full_corpus_class_distribution", {})
    labels = sorted(dist, key=lambda k: -dist[k])
    fig, axes = plt.subplots(1, 2 if full else 1, figsize=(12 if full else 7, 5.2))
    axes = np.atleast_1d(axes)
    for ax, data, title in zip(axes, [full, dist] if full else [dist], ["Full CIC-IDS2017 corpus (2.83M flows)", "Class-capped sample used here"]):
        if not data:
            continue
        vals = [data.get(l, 0) for l in labels]
        colors = [CAT[7] if l == "BENIGN" else CAT[3] for l in labels]
        ax.barh(range(len(labels)), vals, color=colors, height=0.7)
        ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=8)
        ax.invert_yaxis(); ax.set_xscale("log"); ax.set_xlabel("flows (log scale)"); ax.set_title(title, fontsize=10, loc="left")
        for i, v in enumerate(vals):
            ax.text(v * 1.15, i, f"{v:,}", va="center", fontsize=7.5, color=INK2)
        ax.grid(axis="y", visible=False)
    _save(fig, "01_class_distribution.png")


def fig_model_leaderboard(master: Dict[str, Any]):
    models = master["models"]
    keys = sorted(models, key=lambda k: -models[k]["f1"])
    f1 = [models[k]["f1"] for k in keys]
    ci = [models[k].get("f1_ci95") for k in keys]
    colors = [PARADIGM_COLOR[models[k]["paradigm"]] for k in keys]
    fig, ax = plt.subplots(figsize=(9, 6.5))
    y = np.arange(len(keys))
    ax.barh(y, f1, color=colors, height=0.66)
    for i, (v, c) in enumerate(zip(f1, ci)):
        if c:
            ax.plot([c[0], c[1]], [i, i], color=INK, linewidth=1.2)
        ax.text(min(v, 0.999) + 0.0005, i, f"{v:.4f}", va="center", fontsize=8, color=INK)
    lo = max(0.0, min(f1) - 0.02)
    ax.set_xlim(lo, 1.002); ax.set_yticks(y); ax.set_yticklabels([_pretty(k) for k in keys]); ax.invert_yaxis()
    ax.set_xlabel("F1-score on untouched test split (bar) with bootstrap 95% CI (line)")
    ax.set_title("Model leaderboard", loc="left", fontweight="bold")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=v, label=k) for k, v in PARADIGM_COLOR.items()], loc="lower right", frameon=False)
    ax.grid(axis="y", visible=False)
    _save(fig, "02_model_leaderboard_f1.png")


def fig_recall_vs_fpr(master: Dict[str, Any]):
    models = master["models"]
    fig, ax = plt.subplots(figsize=(8, 6))
    for k, ev in models.items():
        c = PARADIGM_COLOR[ev["paradigm"]]
        ax.scatter(ev["false_positive_rate"] * 100, ev["recall"] * 100, s=60, color=c, edgecolor="white", linewidth=1.2, zorder=3)
        ax.annotate(_pretty(k), (ev["false_positive_rate"] * 100, ev["recall"] * 100), xytext=(5, 4), textcoords="offset points", fontsize=7.5, color=INK2)
    ax.set_xlabel("False-positive rate (%)  -  lower is better"); ax.set_ylabel("Attack recall (%)  -  higher is better")
    ax.set_title("Detection vs false alarms (test split)", loc="left", fontweight="bold")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=v, label=k) for k, v in PARADIGM_COLOR.items()], loc="lower right", frameon=False)
    _save(fig, "03_recall_vs_fpr.png")


def fig_roc_pr(master: Dict[str, Any]):
    models = master["models"]
    show = ["logistic_regression", "decision_tree", "support_vector_machine", "k_nearest_neighbors", "optimized_xgboost", "stacking"]
    show = [k for k in show if k in models]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for i, k in enumerate(show):
        ev = models[k]; c = CAT[i % len(CAT)]
        roc = ev["roc_curve"]; pr = ev["pr_curve"]
        axes[0].plot([p["fpr"] for p in roc], [p["tpr"] for p in roc], color=c, linewidth=2, label=f"{_pretty(k)} (AUC {ev['roc_auc']:.4f})")
        axes[1].plot([p["recall"] for p in pr], [p["precision"] for p in pr], color=c, linewidth=2, label=f"{_pretty(k)} (AP {ev['pr_auc']:.4f})")
    axes[0].plot([0, 1], [0, 1], color=GRID, linestyle="--", linewidth=1)
    axes[0].set_xlabel("False-positive rate"); axes[0].set_ylabel("True-positive rate (recall)"); axes[0].set_title("ROC curves", loc="left", fontweight="bold")
    axes[0].set_xlim(-0.005, 0.2); axes[0].set_ylim(0.8, 1.005); axes[0].legend(frameon=False, fontsize=8, loc="lower right")
    axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision"); axes[1].set_title("Precision-Recall curves", loc="left", fontweight="bold")
    axes[1].set_xlim(0.8, 1.005); axes[1].set_ylim(0.8, 1.005); axes[1].legend(frameon=False, fontsize=8, loc="lower left")
    fig.text(0.5, -0.02, "Axes are zoomed to the top-left corner: every model is above 0.8, the differences live there.", ha="center", fontsize=8, color=INK2)
    _save(fig, "04_roc_pr_curves.png")


def fig_confusion_matrices(master: Dict[str, Any]):
    models = master["models"]
    show = ["logistic_regression", "support_vector_machine", "random_forest", "optimized_xgboost", "voting_soft", "stacking"]
    show = [k for k in show if k in models]
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.6))
    for ax, k in zip(axes.ravel(), show):
        cm = models[k]["confusion_matrix"]
        mat = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
        ax.imshow(np.log1p(mat), cmap="Blues", vmin=0)
        for (i, j), v in np.ndenumerate(mat):
            ax.text(j, i, f"{v:,}", ha="center", va="center", fontsize=10, color="white" if np.log1p(v) > 0.6 * np.log1p(mat.max()) else INK)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["pred BENIGN", "pred ATTACK"], fontsize=8)
        ax.set_yticks([0, 1]); ax.set_yticklabels(["BENIGN", "ATTACK"], fontsize=8)
        ax.set_title(f"{_pretty(k)}  (FN={cm['fn']}, FP={cm['fp']})", fontsize=9, loc="left"); ax.grid(False)
    for ax in axes.ravel()[len(show):]:
        ax.axis("off")
    fig.suptitle("Confusion matrices on the test split", x=0.02, ha="left", fontweight="bold")
    _save(fig, "05_confusion_matrices.png")


def fig_cv_boxplot(cv: Dict[str, Any]):
    models = cv["models"]
    keys = sorted(models, key=lambda k: -models[k]["f1_mean"])
    data = [models[k]["f1_folds"] for k in keys]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    bp = ax.boxplot(data, vert=True, patch_artist=True, widths=0.55, medianprops={"color": INK, "linewidth": 1.5})
    for patch in bp["boxes"]:
        patch.set_facecolor(CAT[0]); patch.set_alpha(0.35); patch.set_edgecolor(CAT[0])
    for i, k in enumerate(keys):
        ax.scatter([i + 1], [models[k]["test_f1"]], marker="D", color=CAT[3], s=30, zorder=4)
        ax.text(i + 1, max(data[i]) + 0.002, f"{models[k]['f1_mean']:.4f}±{models[k]['f1_std']:.4f}", ha="center", fontsize=7, color=INK2)
    ax.set_xticks(range(1, len(keys) + 1)); ax.set_xticklabels([_pretty(k) for k in keys], rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("F1 per fold"); ax.set_title(f"{cv['n_splits']}-fold stratified cross-validation on the train split (◆ = test-split F1)", loc="left", fontweight="bold", fontsize=10)
    _save(fig, "06_cross_validation_boxplot.png")


def fig_feature_selection(fs: Dict[str, Any]):
    bench = [r for r in fs["benchmark"] if r["model"] == "random_forest"]
    names = [r["feature_set"] for r in bench]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    x = np.arange(len(bench))
    axes[0].bar(x, [r["f1"] for r in bench], color=CAT[0], width=0.6)
    lo = min(r["f1"] for r in bench) - 0.01
    axes[0].set_ylim(max(0, lo), 1.0)
    for i, r in enumerate(bench):
        axes[0].text(i, r["f1"] + 0.0005, f"{r['f1']:.4f}\n({r['n_features']} feats)", ha="center", fontsize=7.5, color=INK)
    axes[0].set_title("Random Forest F1 by feature set", loc="left", fontweight="bold", fontsize=10)
    axes[1].bar(x, [r["training_time_sec"] for r in bench], color=CAT[1], width=0.6)
    for i, r in enumerate(bench):
        axes[1].text(i, r["training_time_sec"], f"{r['training_time_sec']:.2f}s", ha="center", va="bottom", fontsize=7.5, color=INK)
    axes[1].set_title("Random Forest training time by feature set", loc="left", fontweight="bold", fontsize=10); axes[1].set_ylabel("seconds")
    for ax in axes:
        ax.set_xticks(x); ax.set_xticklabels([n.replace("_", " ") for n in names], rotation=20, ha="right", fontsize=8); ax.grid(axis="x", visible=False)
    _save(fig, "07_feature_selection_benchmark.png")


def fig_pca(pca: Dict[str, Any]):
    curve = pca["cumulative_variance_curve"]
    k = pca["n_components_retained"]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(range(1, len(curve) + 1), curve, color=CAT[0], linewidth=2, marker="o", markersize=4)
    ax.axhline(0.95, color=CAT[3], linestyle="--", linewidth=1); ax.axvline(k, color=CAT[3], linestyle="--", linewidth=1)
    ax.text(k + 0.5, 0.5, f"{k} components explain ≥95% variance", color=CAT[3], fontsize=9)
    ax.set_xlabel("number of principal components"); ax.set_ylabel("cumulative explained variance"); ax.set_ylim(0, 1.02)
    ax.set_title("PCA scree (cumulative) - fitted on the train split", loc="left", fontweight="bold", fontsize=10)
    _save(fig, "08_pca_cumulative_variance.png")


def fig_search_comparison(opt: Dict[str, Any]):
    cmp = opt["search_comparison"]
    algos = ["random_forest", "xgboost"]
    methods = ["GridSearchCV", "RandomizedSearchCV", "Optuna TPE"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    w = 0.26
    for mi, m in enumerate(methods):
        rows = [next((r for r in cmp if r["algorithm"] == a and r["method"] == m), None) for a in algos]
        x = np.arange(len(algos)) + (mi - 1) * w
        axes[0].bar(x, [r["best_cv_f1"] if r else 0 for r in rows], width=w, color=CAT[mi], label=m)
        axes[1].bar(x, [r["test_f1"] if r else 0 for r in rows], width=w, color=CAT[mi], label=m)
        axes[2].bar(x, [r["wall_time_sec"] if r else 0 for r in rows], width=w, color=CAT[mi], label=m)
        for ax_i, key, fmt in ((0, "best_cv_f1", "{:.4f}"), (1, "test_f1", "{:.4f}"), (2, "wall_time_sec", "{:.0f}s")):
            for xi, r in zip(x, rows):
                if r:
                    axes[ax_i].text(xi, r[key], fmt.format(r[key]) + (f"\n{r['n_configs_evaluated']} cfg" if ax_i == 2 else ""), ha="center", va="bottom", fontsize=7, color=INK)
    for ax, title, yl in zip(axes, ["Best cross-validated F1", "Test F1 of the winner", "Search wall-clock time"], ["F1", "F1", "seconds"]):
        ax.set_xticks(range(len(algos))); ax.set_xticklabels([_pretty(a) for a in algos]); ax.set_title(title, loc="left", fontweight="bold", fontsize=10); ax.set_ylabel(yl); ax.grid(axis="x", visible=False)
    f1s = [r["best_cv_f1"] for r in cmp] + [r["test_f1"] for r in cmp]
    axes[0].set_ylim(min(f1s) - 0.01, 1.0); axes[1].set_ylim(min(f1s) - 0.01, 1.0)
    axes[0].legend(frameon=False, fontsize=8, loc="lower left")
    _save(fig, "09_grid_vs_random_vs_optuna.png")


def fig_optuna_convergence(opt: Dict[str, Any]):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, key, title in zip(axes, ["optimized_random_forest", "optimized_xgboost"], ["Random Forest", "XGBoost"]):
        hist = opt[key]["trials_history"]
        vals = [h["value"] for h in hist]
        best = np.maximum.accumulate(vals)
        ax.scatter(range(len(vals)), vals, color=CAT[0], s=22, label="trial CV F1", zorder=3)
        ax.step(range(len(vals)), best, color=CAT[3], where="post", linewidth=2, label="best so far")
        for m_key, c, lab in (("grid_" + key.replace("optimized_", ""), CAT[2], "GridSearchCV best"), ("random_" + key.replace("optimized_", ""), CAT[1], "RandomizedSearchCV best")):
            if m_key in opt:
                ax.axhline(opt[m_key]["best_cv_f1"], color=c, linestyle="--", linewidth=1.2, label=lab)
        ax.set_xlabel("Optuna trial"); ax.set_ylabel("5-fold CV F1"); ax.set_title(f"Optuna (TPE) convergence - {title}", loc="left", fontweight="bold", fontsize=10)
        ax.legend(frameon=False, fontsize=8, loc="lower right")
    _save(fig, "10_optuna_convergence.png")


def fig_imbalance(imb: Dict[str, Any]):
    recs = imb["records"]
    strategies = imb["strategies"]; models = ["logistic_regression", "decision_tree", "random_forest"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    w = 0.2
    for si, s in enumerate(strategies):
        x = np.arange(len(models)) + (si - 1.5) * w
        rows = [next(r for r in recs if r["strategy"] == s and r["model"] == m) for m in models]
        for ax, key in zip(axes, ["recall", "false_positive_rate", "f1"]):
            ax.bar(x, [r[key] for r in rows], width=w, color=CAT[si], label=s.replace("_", " "))
    for ax, title in zip(axes, ["Attack recall", "False-positive rate", "F1-score"]):
        ax.set_xticks(range(len(models))); ax.set_xticklabels([_pretty(m) for m in models]); ax.set_title(title, loc="left", fontweight="bold", fontsize=10); ax.grid(axis="x", visible=False)
    axes[0].set_ylim(min(r["recall"] for r in recs) - 0.05, 1.0); axes[2].set_ylim(min(r["f1"] for r in recs) - 0.05, 1.0)
    axes[0].legend(frameon=False, fontsize=8, loc="lower left")
    fig.suptitle(f"Imbalance handling (train split is {imb['train_attack_ratio_before']*100:.1f}% attack; resampling applied to train only)", x=0.02, ha="left", fontsize=10)
    _save(fig, "11_imbalance_strategies.png")


def fig_per_class_recall(pcr: Dict[str, Any], master: Dict[str, Any]):
    models = pcr["models"]
    show = [k for k in ["logistic_regression", "support_vector_machine", "optimized_xgboost", "stacking", master.get("best_model")] if k in models]
    show = list(dict.fromkeys(show))
    classes = [r["class"] for r in models[show[0]] if r["is_attack"]]
    mat = np.array([[next(r["recall"] for r in models[m] if r["class"] == c) or 0 for c in classes] for m in show])
    fig, ax = plt.subplots(figsize=(10, 0.55 * len(show) + 2.2))
    im = ax.imshow(mat, cmap="Blues", vmin=0.5, vmax=1.0, aspect="auto")
    for (i, j), v in np.ndenumerate(mat):
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8, color="white" if v > 0.85 else INK)
    n_test = {r["class"]: r["n_test"] for r in models[show[0]]}
    ax.set_xticks(range(len(classes))); ax.set_xticklabels([f"{c}\n(n={n_test[c]})" for c in classes], rotation=35, ha="right", fontsize=7.5)
    ax.set_yticks(range(len(show))); ax.set_yticklabels([_pretty(m) for m in show], fontsize=9); ax.grid(False)
    ax.set_title("Per-attack-type recall on the test split (binary detector)", loc="left", fontweight="bold", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02, label="recall")
    _save(fig, "12_per_attack_type_recall.png")


def fig_shap(shap_data: Dict[str, Any]):
    items = shap_data["global_importance"][:15][::-1]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.barh(range(len(items)), [i["importance"] for i in items], color=CAT[0], height=0.65)
    ax.set_yticks(range(len(items))); ax.set_yticklabels([i["feature"] for i in items], fontsize=8.5)
    ax.set_xlabel("mean |SHAP value|  (average impact on the attack log-odds)")
    ax.set_title(f"SHAP global feature importance - {_pretty(shap_data['model_name'])}", loc="left", fontweight="bold", fontsize=10); ax.grid(axis="y", visible=False)
    _save(fig, "13_shap_global_importance.png")


def fig_latency(master: Dict[str, Any]):
    models = master["models"]
    keys = sorted(models, key=lambda k: models[k]["inference_time_ms_per_1k"])
    fig, ax = plt.subplots(figsize=(9, 5.5))
    vals = [models[k]["inference_time_ms_per_1k"] for k in keys]
    ax.barh(range(len(keys)), vals, color=[PARADIGM_COLOR[models[k]["paradigm"]] for k in keys], height=0.66)
    for i, v in enumerate(vals):
        ax.text(v * 1.08, i, f"{v:.2f} ms", va="center", fontsize=8, color=INK)
    ax.set_xscale("log"); ax.set_yticks(range(len(keys))); ax.set_yticklabels([_pretty(k) for k in keys]); ax.invert_yaxis()
    ax.set_xlabel("inference time per 1,000 flows (ms, log scale)"); ax.set_title("Inference latency", loc="left", fontweight="bold", fontsize=10); ax.grid(axis="y", visible=False)
    _save(fig, "14_inference_latency.png")


def fig_correlation_heatmap():
    splits_file = ML_DIR / "data" / "splits" / "dataset_splits.joblib"
    if not splits_file.exists():
        return
    import joblib
    s = joblib.load(splits_file)
    df: pd.DataFrame = s["X_train_df_raw"].sample(n=min(20000, len(s["X_train_df_raw"])), random_state=42)
    corr = df.corr().abs()
    fig, ax = plt.subplots(figsize=(11, 9.5))
    im = ax.imshow(corr.values, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(corr))); ax.set_xticklabels(corr.columns, rotation=90, fontsize=5.5)
    ax.set_yticks(range(len(corr))); ax.set_yticklabels(corr.columns, fontsize=5.5); ax.grid(False)
    ax.set_title("Absolute Pearson correlation between features (train split) - blocks show redundancy the r>0.90 filter removes", loc="left", fontsize=9, fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    _save(fig, "15_feature_correlation_heatmap.png")


def main():
    print(">>> Generating figures into docs/figures/")
    master = _load("master_comparison.json")
    summary = _load("dataset_summary.json")
    fig_class_distribution(summary)
    fig_model_leaderboard(master)
    fig_recall_vs_fpr(master)
    fig_roc_pr(master)
    fig_confusion_matrices(master)
    fig_cv_boxplot(_load("cv_results.json"))
    fig_feature_selection(_load("feature_selection_summary.json"))
    fig_pca(_load("pca_summary.json"))
    opt = _load("optimization_study.json")
    fig_search_comparison(opt)
    fig_optuna_convergence(opt)
    fig_imbalance(_load("imbalance_study.json"))
    if (METRICS / "per_class_recall.json").exists():
        fig_per_class_recall(_load("per_class_recall.json"), master)
    fig_shap(_load("shap_global.json"))
    fig_latency(master)
    fig_correlation_heatmap()
    print(">>> Done.")


if __name__ == "__main__":
    main()
