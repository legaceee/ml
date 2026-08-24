# AI-Based Network Intrusion Detection System

**Detecting cyber-attacks in network flows with optimized machine-learning models and a comparison of ensemble methods — on the real CIC-IDS2017 dataset.**

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.7-F7931E.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.x-189FDD.svg)](https://xgboost.readthedocs.io/)
[![Optuna](https://img.shields.io/badge/Optuna-TPE-3B5BDB.svg)](https://optuna.org/)
[![SHAP](https://img.shields.io/badge/Explainability-SHAP-FF6F00.svg)](https://shap.readthedocs.io/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688.svg)](https://fastapi.tiangolo.com/)
[![React 18](https://img.shields.io/badge/React-18-61DAFB.svg)](https://reactjs.org/)

> **Start here:** [`docs/walkthrough.md`](docs/walkthrough.md) explains what was done, in what order, and what came out — with the numbers. [`docs/syllabus-mapping.md`](docs/syllabus-mapping.md) maps every course topic to the exact code and result. [`docs/results.md`](docs/results.md) is generated from the experiment artifacts.

---

## 1. What this project does

A network flow is a summary of one conversation between two hosts — how many packets, how big, how fast, which TCP flags. The task: given 70 such statistics from CICFlowMeter, decide **BENIGN or ATTACK**.

The project compares **seven individual algorithms** (Logistic Regression, KNN, Decision Tree, SVM, Random Forest, Extra Trees, XGBoost), **three hyperparameter-search strategies** (Grid, Random, Bayesian/Optuna), **four imbalance strategies** (none, class weights, under-sampling, SMOTE), **five feature-selection/reduction methods** (correlation filter, mutual information, tree importance, ANOVA, PCA) and **seven ensembles** (hard/soft voting, bagging, AdaBoost, gradient boosting, stacking, weighted blend) — all on identical, leakage-free splits of a 55,720-flow sample of the real CIC-IDS2017 corpus that keeps all 15 traffic classes.

Every number in this README, the docs and the dashboard is produced by code from the experiment artifacts. Nothing is typed in by hand.

## 2. Pipeline

```
CIC-IDS2017 (8 day files, 2.83 M flows)
   └─ build_dataset.py     class-capped stratified sample → 55,720 flows, 28 % attack, 15 classes
       └─ pipeline.py      clean (Inf/NaN, duplicates, constants) → encode → split 70/15/15 → fit scaler on TRAIN only
           └─ run_all_experiments.py
                 1  feature selection benchmark (+ PCA, + Destination-Port ablation)
                 2  7 baselines + 5-fold stratified CV
                 3  imbalance study: none / class_weight / under-sample / SMOTE
                 4  Grid vs Random vs Optuna-TPE tuning of RF and XGBoost (same CV, same objective)
                 5  ensembles: voting ×2, bagging, AdaBoost, gradient boosting, stacking, weighted
                 6  per-attack-type recall for every model
                 7  bootstrap 95 % CIs + multi-seed retraining
                 8  SHAP global explanation
                 9  master_comparison.json · results.csv · presets.json · research_conclusions.json
               └─ generate_figures.py → docs/figures/     render_results_docs.py → docs/results.md + table below
                    └─ FastAPI (/api/predict, /api/explain, /api/experiments/…) → React dashboard
```

## 3. Leaderboard

<!-- LEADERBOARD:START -->
_Run the experiments and `python -m ml.training.render_results_docs` to fill this table._
<!-- LEADERBOARD:END -->

Full tables (CV, feature selection, search comparison, imbalance, per-attack-type recall, stability, SHAP) and all 15 figures: [`docs/results.md`](docs/results.md).

## 4. Key findings

<!-- KEY-FINDINGS:START -->
_Filled from `research_conclusions.json` after the run._
<!-- KEY-FINDINGS:END -->

## 5. Quick start

```bash
pip install -r requirements.txt

# 1. data — the 55,720-flow CIC-IDS2017 sample is committed (ml/data/raw/cicids2017_sample.csv).
#    Only rebuild it if you want different caps (one-off ~885 MB download):
#    python -m ml.data.build_dataset --download --build --benign-rows 40000 --attack-cap 1500

# 2. preprocess + split (leakage-free)
python -m ml.preprocessing.pipeline

# 3. every experiment (~1 h on 8 cores; add --quick for a 4-minute smoke test)
python -m ml.training.run_all_experiments

# 4. figures + docs
python -m ml.training.generate_figures
python -m ml.training.render_results_docs

# 5. tests
python -m pytest tests/ -q

# 6. app
uvicorn backend.app.main:app --port 8000 --reload      # API + Swagger at http://localhost:8000/docs
cd frontend && npm install && npm run dev               # dashboard at http://localhost:5173
```

Details: [`docs/reproducibility.md`](docs/reproducibility.md), [`DATASET_SETUP.md`](DATASET_SETUP.md).

## 6. Documentation

| Document | What it answers |
|:--|:--|
| [walkthrough.md](docs/walkthrough.md) | **What happened and how** — the narrative, step by step, with numbers |
| [syllabus-mapping.md](docs/syllabus-mapping.md) | Where each course topic (missing values, encoding, scaling, PCA, feature selection, imbalance, LR/KNN/DT/SVM, bagging/boosting, metrics, CV, grid search, tuning) lives in the code |
| [results.md](docs/results.md) | Every table and figure, generated from the artifacts |
| [presentation-script.md](docs/presentation-script.md) | A 10-minute talk with slide-by-slide notes |
| [viva-defense-guide.md](docs/viva-defense-guide.md) | Likely examiner questions and the answers the evidence supports |
| [methodology.md](docs/methodology.md) | Workflow, design principles, research questions, statistical protocol |
| [dataset.md](docs/dataset.md) · [preprocessing.md](docs/preprocessing.md) | CIC-IDS2017, its quirks, the zero-leakage protocol |
| [feature-selection.md](docs/feature-selection.md) · [model-training.md](docs/model-training.md) · [ensemble-methods.md](docs/ensemble-methods.md) | The methods, with the maths |
| [evaluation.md](docs/evaluation.md) · [explainable-ai.md](docs/explainable-ai.md) | Metrics, uncertainty, SHAP |
| [architecture.md](docs/architecture.md) · [reproducibility.md](docs/reproducibility.md) | System design; the four commands that rebuild everything |
| [CLOUD_HANDOFF.md](docs/CLOUD_HANDOFF.md) | How to finish the 1–2 h benchmark run unattended on Claude Code (web), with the exact prompt |

Notebooks mirroring the pipeline stages are in [`notebooks/`](notebooks/) (`01_eda` … `07_ensemble_comparison`).

## 7. Repository layout

```
ml/            data/ · preprocessing/ · feature_engineering/ · models/ · training/ · explainability/ · artifacts/
backend/app/   FastAPI service (api/, services/ml_service.py, schemas/, database/)
frontend/src/  React + Tailwind dashboard (pages/, components/, services/api.ts)
docs/          documentation + figures/
notebooks/     7 notebooks
tests/         unit tests (run anywhere) + integration tests (need trained artifacts)
```

## 8. License & scope

MIT. Defensive research and coursework only: the repository contains no exploits or traffic generators — only a detector trained on a public benchmark.
