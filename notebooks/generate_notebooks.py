"""
Generator for the 7 research notebooks. Each notebook is a thin, readable
front-end over the same modules the experiment runner uses, so the notebook
and the runner can never disagree.

    python notebooks/generate_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb   # optional
"""

import json
from pathlib import Path

NOTEBOOKS_DIR = Path(__file__).resolve().parent
NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)

BOOTSTRAP = """import sys, os
from pathlib import Path
ROOT = Path.cwd() if (Path.cwd() / "ml").exists() else Path.cwd().parent
sys.path.insert(0, str(ROOT)); os.chdir(ROOT)
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, matplotlib.pyplot as plt, joblib, json
print("project root:", ROOT)"""


def make_notebook(cells):
    return {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                         "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip()}


def code(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src.strip()}


def save(name, cells):
    with open(NOTEBOOKS_DIR / name, "w", encoding="utf-8") as f:
        json.dump(make_notebook(cells), f, indent=1)


def build_01_eda():
    save("01_eda.ipynb", [
        md("""# 01 · Exploratory Data Analysis — CIC-IDS2017
What is in the data, how imbalanced it is, which columns are broken (Inf/NaN), and how redundant the features are.
Run `python -m ml.data.build_dataset --download --build` first."""),
        code(BOOTSTRAP),
        code("""from ml.data.loader import DatasetLoader
df, meta = DatasetLoader().load_dataset()
print({k: v for k, v in meta.items() if k not in ("class_distribution", "sampling_manifest")})
df.shape"""),
        code("""# Class distribution in the sample vs the full 2.83M-row corpus
man = meta.get("sampling_manifest") or {}
dist = pd.DataFrame({"sample": pd.Series(meta["class_distribution"]), "full_corpus": pd.Series(man.get("full_corpus_class_distribution", {}))}).fillna(0).astype(int)
dist.sort_values("sample", ascending=False)"""),
        code("""ax = dist["sample"].sort_values().plot.barh(figsize=(8, 5), logx=True, color="#0072B2")
ax.set_xlabel("rows (log)"); ax.set_title("Sample class distribution"); plt.tight_layout()"""),
        code("""# Missing / infinite values: only two CICFlowMeter columns are affected
num = df.drop(columns=[meta["label_column"]]).apply(pd.to_numeric, errors="coerce")
bad = (num.isna() | np.isinf(num)).sum()
bad[bad > 0]"""),
        code("""# Constant columns (zero variance) and the most redundant feature pairs
const = [c for c in num.columns if num[c].nunique(dropna=True) <= 1]
print("constant columns:", const)
corr = num.drop(columns=const).sample(min(20000, len(num)), random_state=42).corr().abs()
pairs = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool)).stack().sort_values(ascending=False)
pairs.head(15)"""),
        code("""# Do attacks and benign flows look different? Compare medians of a few flow-shape features
feats = ["Flow Duration", "Total Fwd Packets", "Fwd Packet Length Mean", "Init_Win_bytes_forward", "Flow Packets/s", "SYN Flag Count"]
df["is_attack"] = (df[meta["label_column"]] != "BENIGN").astype(int)
df.groupby("is_attack")[feats].median().T"""),
    ])


def build_02_preprocessing():
    save("02_preprocessing.ipynb", [
        md("""# 02 · Preprocessing with a zero-leakage protocol
Clean → encode → **split** → fit imputer/scaler on train only → transform every split. Duplicates are removed *before* the split."""),
        code(BOOTSTRAP),
        code("""from ml.preprocessing.pipeline import PipelineOrchestrator
summary = PipelineOrchestrator().run()
summary["cleaning_audit"]"""),
        code("""pd.DataFrame({k: summary["split_info"][k] for k in ("train_class_counts", "val_class_counts", "test_class_counts")}).fillna(0).astype(int)"""),
        code("""# Proof that the scaler was fitted on train only: train columns have mean≈0, std≈1; test columns do not exactly
s = joblib.load("ml/data/splits/dataset_splits.joblib")
print("train mean/std:", s["X_train"].mean().round(3), s["X_train"].std().round(3))
print("test  mean/std:", s["X_test"].mean().round(3), s["X_test"].std().round(3))
print("features:", len(s["feature_names"]))"""),
    ])


def build_03_feature_selection():
    save("03_feature_selection.ipynb", [
        md("""# 03 · Feature selection & PCA (RQ2)
Correlation filter, mutual information, RF importance, ANOVA SelectKBest, and PCA — each fitted on the train split and **benchmarked** with a Random Forest."""),
        code(BOOTSTRAP),
        code("""from ml.feature_engineering.selection import FeatureSelector
from ml.feature_engineering.dimensionality_reduction import PCAReducer
from sklearn.ensemble import RandomForestClassifier
from ml.training.evaluate import evaluate_model
import time
s = joblib.load("ml/data/splits/dataset_splits.joblib")
X_train, y_train, X_test, y_test, names = s["X_train"], s["y_train"], s["X_test"], s["y_test"], s["feature_names"]
fs = FeatureSelector(random_state=42)
subsets = {
    "all": names,
    "corr<0.90": fs.correlation_filter(s["X_train_df_raw"], threshold=0.90),
    "MI top25": fs.mutual_information(X_train, y_train, names, top_k=25),
    "RF top25": fs.tree_importance(X_train, y_train, names, top_k=25),
    "ANOVA k25": fs.select_k_best(X_train, y_train, names, k=25),
}
{k: len(v) for k, v in subsets.items()}"""),
        code("""idx = {n: i for i, n in enumerate(names)}
rows = []
for name, feats in subsets.items():
    cols = [idx[f] for f in feats]
    rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced", n_jobs=-1)
    t = time.perf_counter(); rf.fit(X_train[:, cols], y_train); tt = time.perf_counter() - t
    ev = evaluate_model(rf, X_test[:, cols], y_test, name, training_time=tt)
    rows.append({"feature_set": name, "n": len(cols), "f1": ev["f1"], "recall": ev["recall"], "train_s": round(tt, 2)})
pca = PCAReducer(variance_threshold=0.95, random_state=42)
Xp_tr, Xp_te = pca.fit_transform(X_train), pca.transform(X_test)
rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced", n_jobs=-1)
t = time.perf_counter(); rf.fit(Xp_tr, y_train); tt = time.perf_counter() - t
ev = evaluate_model(rf, Xp_te, y_test, "pca", training_time=tt)
rows.append({"feature_set": "PCA 95%", "n": pca.n_components_retained_, "f1": ev["f1"], "recall": ev["recall"], "train_s": round(tt, 2)})
pd.DataFrame(rows)"""),
        code("""plt.plot(np.arange(1, len(pca.cumulative_variance_[:40]) + 1), pca.cumulative_variance_[:40], marker="o", color="#0072B2")
plt.axhline(0.95, ls="--", color="#D55E00"); plt.xlabel("components"); plt.ylabel("cumulative explained variance"); plt.title("PCA scree"); plt.tight_layout()"""),
        code("""pd.Series(fs.feature_scores_["tree_importance"]).sort_values(ascending=False).head(15)"""),
    ])


def build_04_baselines():
    save("04_baseline_models.ipynb", [
        md("""# 04 · Baseline classifiers + 5-fold cross-validation (RQ1)
Logistic Regression, Decision Tree, Random Forest, Extra Trees, SVM (RBF), KNN, XGBoost with default hyperparameters."""),
        code(BOOTSTRAP),
        code("""from ml.models.baselines import get_baseline_models
from ml.training.evaluate import evaluate_model, cross_validate_classifier
from sklearn.base import clone
import time
s = joblib.load("ml/data/splits/dataset_splits.joblib")
X_train, y_train, X_test, y_test = s["X_train"], s["y_train"], s["X_test"], s["y_test"]
rows = []
for name, model in get_baseline_models(random_state=42).items():
    t = time.perf_counter(); model.fit(X_train, y_train); tt = time.perf_counter() - t
    ev = evaluate_model(model, X_test, y_test, name, training_time=tt)
    cvm = clone(model)
    if name == "support_vector_machine": cvm.set_params(probability=False)
    cv = cross_validate_classifier(cvm, X_train, y_train, name, n_splits=5)
    rows.append({"model": name, "test_f1": ev["f1"], "cv_f1": cv["f1_mean"], "cv_std": cv["f1_std"], "recall": ev["recall"], "fpr": ev["false_positive_rate"], "train_s": round(tt, 1), "infer_ms_1k": ev["inference_time_ms_per_1k"]})
    print(f"{name:26s} test F1 {ev['f1']:.4f} | CV {cv['f1_mean']:.4f} ± {cv['f1_std']:.4f}")
pd.DataFrame(rows).sort_values("test_f1", ascending=False)"""),
    ])


def build_05_imbalance():
    save("05_imbalance_handling.ipynb", [
        md("""# 05 · Imbalanced-dataset handling (RQ7)
none vs `class_weight="balanced"` vs random under-sampling vs SMOTE — resampling applied to the **training split only**."""),
        code(BOOTSTRAP),
        code("""from ml.preprocessing.resampling import run_imbalance_study
s = joblib.load("ml/data/splits/dataset_splits.joblib")
res = run_imbalance_study(s["X_train"], s["y_train"], s["X_test"], s["y_test"], random_state=42)
print("train class counts before:", res["train_class_counts_before"])
df = pd.DataFrame(res["records"])[["strategy", "model", "train_rows_after_resampling", "recall", "precision", "f1", "false_positive_rate"]]
df"""),
        code("""df.pivot(index="model", columns="strategy", values="recall").plot.bar(figsize=(8, 4), color=["#0072B2", "#E69F00", "#009E73", "#D55E00"])
plt.ylabel("attack recall"); plt.ylim(df["recall"].min() - 0.05, 1); plt.title("Recall by imbalance strategy"); plt.tight_layout()
res["best_strategy_by_model"]"""),
    ])


def build_06_tuning():
    save("06_hyperparameter_tuning.ipynb", [
        md("""# 06 · Hyperparameter tuning: Grid Search vs Random Search vs Optuna (RQ3)
Same 5-fold stratified CV and the same F1 objective for all three; only the search strategy changes. Budgets are reduced here so the notebook runs in minutes."""),
        code(BOOTSTRAP),
        code("""from ml.models.optimization import HyperparameterOptimizer
from ml.models import optimization as opt
from ml.training.evaluate import evaluate_model
s = joblib.load("ml/data/splits/dataset_splits.joblib")
X_train, y_train, X_test, y_test = s["X_train"], s["y_train"], s["X_test"], s["y_test"]
opt.RF_GRID = {"n_estimators": [100, 200], "max_depth": [10, 20, None]}     # 6 configs
optimizer = HyperparameterOptimizer(n_trials=8, cv_folds=5, random_state=42)
rows = []
for method, fn in [("GridSearchCV", optimizer.grid_search_random_forest), ("RandomizedSearchCV", lambda X, y: optimizer.random_search_random_forest(X, y, n_iter=6)), ("Optuna TPE", optimizer.optimize_random_forest)]:
    est, res = fn(X_train, y_train)
    ev = evaluate_model(est, X_test, y_test, method)
    rows.append({"method": method, "configs": res["n_trials"], "wall_s": res["wall_time_sec"], "best_cv_f1": round(res["best_cv_f1"], 4), "test_f1": ev["f1"], "best_params": res["best_params"]})
pd.DataFrame(rows)"""),
        code("""hist = optimizer.study_results_["optimized_random_forest"]["trials_history"]
vals = [h["value"] for h in hist]
plt.scatter(range(len(vals)), vals, color="#0072B2"); plt.step(range(len(vals)), np.maximum.accumulate(vals), where="post", color="#D55E00")
plt.xlabel("Optuna trial"); plt.ylabel("CV F1"); plt.title("TPE convergence"); plt.tight_layout()"""),
    ])


def build_07_ensembles():
    save("07_ensemble_comparison.ipynb", [
        md("""# 07 · Ensembles: Voting, Bagging, Boosting, Stacking (RQ4, RQ5)
Heterogeneous base learners (RF + XGB + LR + SVM). The weighted ensemble tunes its weights on the validation split. Also: per-attack-type recall (RQ8) and SHAP (RQ6)."""),
        code(BOOTSTRAP),
        code("""from ml.models.ensembles import build_ensemble_models
from ml.training.evaluate import evaluate_model, per_class_recall
import time
s = joblib.load("ml/data/splits/dataset_splits.joblib")
X_train, y_train, X_val, y_val, X_test, y_test = s["X_train"], s["y_train"], s["X_val"], s["y_val"], s["X_test"], s["y_test"]
opt_rf = joblib.load("ml/artifacts/models/optimized_random_forest.joblib") if Path("ml/artifacts/models/optimized_random_forest.joblib").exists() else None
opt_xgb = joblib.load("ml/artifacts/models/optimized_xgboost.joblib") if Path("ml/artifacts/models/optimized_xgboost.joblib").exists() else None
rows, fitted = [], {}
for name, model in build_ensemble_models(random_state=42, optimized_rf=opt_rf, optimized_xgb=opt_xgb).items():
    t = time.perf_counter()
    model.fit(X_train, y_train, X_val=X_val, y_val=y_val) if name == "weighted_ensemble" else model.fit(X_train, y_train)
    tt = time.perf_counter() - t
    ev = evaluate_model(model, X_test, y_test, name, training_time=tt); fitted[name] = model
    rows.append({"ensemble": name, "f1": ev["f1"], "recall": ev["recall"], "fpr": ev["false_positive_rate"], "roc_auc": ev["roc_auc"], "train_s": round(tt, 1), "infer_ms_1k": ev["inference_time_ms_per_1k"]})
    print(f"{name:22s} F1 {ev['f1']:.4f} recall {ev['recall']:.4f} FPR {ev['false_positive_rate']:.4f}")
pd.DataFrame(rows).sort_values("f1", ascending=False)"""),
        code("""# Per-attack-type recall of the stacking ensemble
pd.DataFrame(per_class_recall(fitted["stacking"], X_test, y_test, s["y_test_labels"]))"""),
        code("""# SHAP on the tuned XGBoost (exact TreeExplainer)
from ml.explainability.shap_analysis import IDSExplainer
if opt_xgb is not None:
    exp = IDSExplainer(opt_xgb, feature_names=s["feature_names"], background_data=X_train[:200])
    imp = pd.DataFrame(exp.explain_global(X_test[:500], max_display=15))
    imp.set_index("feature")["importance"].iloc[::-1].plot.barh(figsize=(7, 5), color="#0072B2"); plt.xlabel("mean |SHAP|"); plt.tight_layout()
    imp"""),
    ])


if __name__ == "__main__":
    build_01_eda(); build_02_preprocessing(); build_03_feature_selection(); build_04_baselines()
    build_05_imbalance(); build_06_tuning(); build_07_ensembles()
    print("Generated 7 notebooks in notebooks/")
