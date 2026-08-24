# Project Overview: AI-Based Network Intrusion Detection

## 1. Problem

Networks are attacked by denial-of-service floods, port scans, brute-force logins, botnets and web exploits. Signature-based intrusion detection (Snort-style rules) matches known byte patterns and misses anything new. **Anomaly-based ML detection** instead learns what benign and malicious *flows* look like — packet counts, byte rates, inter-arrival times, TCP flags, window sizes — and classifies traffic by behaviour.

**Task**: given one bidirectional network flow described by 70 CICFlowMeter statistics, predict `BENIGN` or `ATTACK`.

## 2. Data

Real **CIC-IDS2017** (University of New Brunswick): 5 days of captured traffic, 2,830,743 flows, 14 attack types. A class-capped stratified sample of 55,720 flows (28 % attack, every attack class present) is used so that all 18 models train in about an hour on a laptop. See `DATASET_SETUP.md`.

## 3. Hypothesis

> Ensembles that combine tree, linear and kernel learners (voting, bagging, boosting, stacking) detect attacks with higher recall and lower false-positive rate than any individually tuned model — and the gain is measurable beyond seed noise.

The project tests this hypothesis rather than assuming it: RQ4 reports the actual gap between the best ensemble and the best single model with bootstrap confidence intervals, whichever way it comes out.

## 4. What was built

| Layer | Content |
|:--|:--|
| **Data** | downloader + class-capped sampler; cleaning (Inf/NaN, duplicates, constant columns); leakage-free 70/15/15 split |
| **Feature engineering** | correlation filter, mutual information, RF importance, ANOVA, PCA — each benchmarked; `Destination Port` ablation |
| **Imbalance** | none vs class weights vs under-sampling vs SMOTE, train-only |
| **Models** | LR, KNN, DT, SVM, RF, ET, XGBoost; Grid / Random / Optuna-TPE tuning; hard/soft voting, bagging, AdaBoost, gradient boosting, stacking, weighted blend |
| **Evaluation** | accuracy, precision, recall, F1, ROC/PR-AUC, FPR/FNR, latency; 5-fold CV; bootstrap CIs; multi-seed; per-attack-type recall; SHAP |
| **Serving** | FastAPI (`/api/predict`, `/api/explain`, `/api/experiments/…`), React + Tailwind dashboard with a Course-Topics Lab page |
| **Reproducibility** | four commands regenerate data, models, figures and docs; results docs are rendered from JSON |

## 5. Headline findings

Numbers live in `docs/results.md` (generated) and the narrative in `docs/walkthrough.md`. In short: tree-based models separate attacks almost perfectly, the linear model does not; tuning and ensembling yield small but consistent gains at very different costs; the practical deployment choice is the tuned XGBoost, with stacking when recall matters more than latency; the rarest web-attack categories are where residual misses concentrate.

## 6. Repository map

```
ml/data/               build_dataset.py (download + sample), loader.py, generate_dev_sample.py (synthetic fallback)
ml/preprocessing/      cleaning.py, encoding.py, scaling.py, resampling.py, pipeline.py
ml/feature_engineering/ selection.py, dimensionality_reduction.py
ml/models/             baselines.py, optimization.py (grid / random / optuna), ensembles.py
ml/training/           evaluate.py, run_all_experiments.py, generate_figures.py, render_results_docs.py
ml/explainability/     shap_analysis.py
ml/artifacts/          models/, metrics/*.json, experiments/results.csv + research_conclusions.json
backend/app/           FastAPI service (api/, services/ml_service.py, schemas/, database/)
frontend/src/          React pages (Dashboard, Predictor, Leaderboard, Course Lab, Research Findings, …)
notebooks/             7 notebooks mirroring the pipeline stages
docs/                  this documentation + figures/
tests/                 unit + integration tests
```
