"""
Generator for the 6 Academic Jupyter Research Notebooks.
"""

import json
from pathlib import Path

NOTEBOOKS_DIR = Path(__file__).resolve().parent
NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)


def make_notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.11.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }


def md_cell(text):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.strip().split("\n")]
    }


def code_cell(code):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in code.strip().split("\n")]
    }


def build_01_eda():
    cells = [
        md_cell("""# 01 - Exploratory Data Analysis (EDA)
## Cyber Attack Intrusion Detection (CIC-IDS2017 Benchmark)

This notebook explores the dataset distribution, class imbalances, protocol features, and correlation profiles of network flow captures.
"""),
        code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from ml.data.loader import DatasetLoader

loader = DatasetLoader()
df, meta = loader.load_dataset()
print("Dataset Metadata:")
for k, v in meta.items():
    print(f"  {k}: {v}")
"""),
        code_cell("""# Inspect Shape and Column Types
print("Dimensions:", df.shape)
df.info()
df.head()
"""),
        code_cell("""# Class Distribution Analysis
label_col = meta["label_column"]
class_counts = df[label_col].value_counts()
print(class_counts)

plt.figure(figsize=(10, 4))
sns.barplot(x=class_counts.values, y=class_counts.index, palette="viridis")
plt.title("CIC-IDS2017 Class Distribution")
plt.xlabel("Sample Count")
plt.ylabel("Traffic Category")
plt.tight_layout()
plt.show()
"""),
        code_cell("""# Check Missing Values and Inf Count
inf_nan_count = df.isin([np.nan, np.inf, -np.inf, "Infinity", "inf"]).sum().sum()
print("Total Infinite/NaN values detected:", inf_nan_count)
""")
    ]
    with open(NOTEBOOKS_DIR / "01_eda.ipynb", "w") as f:
        json.dump(make_notebook(cells), f, indent=2)


def build_02_preprocessing():
    cells = [
        md_cell("""# 02 - Leakage-Free Data Preprocessing & Pipeline
## Strict Train / Val / Test Partitioning

Protocol:
1. Stratified split (70% Train, 15% Val, 15% Test) BEFORE fitting transformers.
2. Imputation and StandardScaler fitted strictly on Train only.
3. Validation and Test splits transformed using pre-fitted parameters.
"""),
        code_cell("""import joblib
from ml.preprocessing.pipeline import PipelineOrchestrator

pipeline = PipelineOrchestrator()
summary = pipeline.run()
print("Preprocessing summary:")
print(summary["split_info"])
"""),
        code_cell("""# Verify Zero Data Leakage
splits = joblib.load("ml/data/splits/dataset_splits.joblib")
print("Train shape:", splits["X_train"].shape)
print("Val shape:", splits["X_val"].shape)
print("Test shape:", splits["X_test"].shape)
print("Retained Features Count:", len(splits["feature_names"]))
""")
    ]
    with open(NOTEBOOKS_DIR / "02_preprocessing.ipynb", "w") as f:
        json.dump(make_notebook(cells), f, indent=2)


def build_03_feature_selection():
    cells = [
        md_cell("""# 03 - Feature Selection & Dimensionality Reduction
## Investigating RQ2: Does Feature Selection Improve Model Performance and Efficiency?

Methods Evaluated:
- Correlation Filtering (Collinearity threshold r > 0.90)
- Mutual Information Ranking
- Tree Importance (Random Forest Gini)
- PCA (95% variance)
"""),
        code_cell("""import joblib
from ml.feature_engineering.selection import FeatureSelector
from ml.feature_engineering.dimensionality_reduction import PCAReducer

splits = joblib.load("ml/data/splits/dataset_splits.joblib")
X_train = splits["X_train"]
y_train = splits["y_train"]
feature_names = splits["feature_names"]
X_train_df = splits["X_train_df_raw"]

fs = FeatureSelector(random_state=42)
corr_feats = fs.correlation_filter(X_train_df, threshold=0.90)
mi_feats = fs.mutual_information(X_train, y_train, feature_names, top_k=20)
tree_feats = fs.tree_importance(X_train, y_train, feature_names, top_k=20)

print(f"Original Features: {len(feature_names)}")
print(f"Correlation Filtered: {len(corr_feats)} (Pruned {len(feature_names) - len(corr_feats)} collinear attributes)")
print(f"Top 5 Mutual Info: {mi_feats[:5]}")
print(f"Top 5 Tree Importance: {tree_feats[:5]}")
"""),
        code_cell("""# PCA Cumulative Variance Analysis
pca = PCAReducer(variance_threshold=0.95, random_state=42)
X_pca = pca.fit_transform(X_train)
summary = pca.get_variance_summary()
print(f"Principal components required for 95% variance: {summary['n_components_retained']}")
""")
    ]
    with open(NOTEBOOKS_DIR / "03_feature_selection.ipynb", "w") as f:
        json.dump(make_notebook(cells), f, indent=2)


def build_04_baseline_models():
    cells = [
        md_cell("""# 04 - Baseline Machine Learning Classifiers
## Investigating RQ1: How Accurately Can Individual ML Models Detect Attacks?

Algorithms:
1. Logistic Regression
2. Decision Tree
3. Random Forest
4. Extra Trees
5. Support Vector Machine (RBF)
6. K-Nearest Neighbors
7. XGBoost (Default)
"""),
        code_cell("""import joblib
from ml.models.baselines import get_baseline_models
from ml.training.evaluate import evaluate_model

splits = joblib.load("ml/data/splits/dataset_splits.joblib")
X_train, y_train = splits["X_train"], splits["y_train"]
X_test, y_test = splits["X_test"], splits["y_test"]

models = get_baseline_models(random_state=42)
results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    ev = evaluate_model(model, X_test, y_test, model_name=name)
    results[name] = ev
    print(f"{name:25s} | F1: {ev['f1']:.4f} | Recall: {ev['recall']:.4f} | Acc: {ev['accuracy']:.4f}")
""")
    ]
    with open(NOTEBOOKS_DIR / "04_baseline_models.ipynb", "w") as f:
        json.dump(make_notebook(cells), f, indent=2)


def build_05_hyperparameter_optimization():
    cells = [
        md_cell("""# 05 - Hyperparameter Optimization via Optuna
## Investigating RQ3: Comparing Unoptimized Baselines vs Optuna Tuned Models

Bayesian Tree-structured Parzen Estimator (TPE) 5-Fold Stratified Cross-Validation on Train Split.
"""),
        code_cell("""import joblib
from ml.models.optimization import HyperparameterOptimizer
from ml.training.evaluate import evaluate_model

splits = joblib.load("ml/data/splits/dataset_splits.joblib")
X_train, y_train = splits["X_train"], splits["y_train"]
X_test, y_test = splits["X_test"], splits["y_test"]

optimizer = HyperparameterOptimizer(n_trials=15, cv_folds=5, random_state=42)
opt_rf, rf_study = optimizer.optimize_random_forest(X_train, y_train)
opt_xgb, xgb_study = optimizer.optimize_xgboost(X_train, y_train)

ev_rf = evaluate_model(opt_rf, X_test, y_test, "optimized_rf")
ev_xgb = evaluate_model(opt_xgb, X_test, y_test, "optimized_xgb")

print("Optimized Random Forest F1:", ev_rf["f1"], "Best CV:", rf_study["best_cv_f1"])
print("Optimized XGBoost F1:", ev_xgb["f1"], "Best CV:", xgb_study["best_cv_f1"])
""")
    ]
    with open(NOTEBOOKS_DIR / "05_hyperparameter_optimization.ipynb", "w") as f:
        json.dump(make_notebook(cells), f, indent=2)


def build_06_ensemble_comparison():
    cells = [
        md_cell("""# 06 - Ensemble Learning & Research Conclusions
## Investigating RQ4 & RQ5: Ensemble Superiority and Trade-off Analysis

Paradigms:
- Voting (Hard & Soft)
- Bagging (Extra Trees)
- Boosting (AdaBoost, Gradient Boosting, XGBoost)
- Stacking Classifier (Meta-Learner: Calibrated Logistic Regression)
"""),
        code_cell("""import joblib
from ml.models.ensembles import build_ensemble_models
from ml.training.evaluate import evaluate_model

splits = joblib.load("ml/data/splits/dataset_splits.joblib")
X_train, y_train = splits["X_train"], splits["y_train"]
X_test, y_test = splits["X_test"], splits["y_test"]

ensembles = build_ensemble_models(random_state=42)
for name, model in ensembles.items():
    model.fit(X_train, y_train)
    ev = evaluate_model(model, X_test, y_test, model_name=name)
    print(f"{name:20s} | F1: {ev['f1']:.4f} | Recall: {ev['recall']:.4f} | FPR: {ev['false_positive_rate']:.4f}")
""")
    ]
    with open(NOTEBOOKS_DIR / "06_ensemble_comparison.ipynb", "w") as f:
        json.dump(make_notebook(cells), f, indent=2)


if __name__ == "__main__":
    build_01_eda()
    build_02_preprocessing()
    build_03_feature_selection()
    build_04_baseline_models()
    build_05_hyperparameter_optimization()
    build_06_ensemble_comparison()
    print("Generated all 6 Jupyter Research Notebooks in notebooks/")
