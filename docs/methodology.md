# Research Methodology

## 1. Experimental workflow

Implemented end-to-end in `ml/training/run_all_experiments.py`; every stage writes a JSON artifact that the docs, figures and dashboard read.

```
 CIC-IDS2017 corpus (8 day files, 2,830,743 flows, 15 classes)
      │  ml/data/build_dataset.py
      ▼
 Class-capped stratified sample (55,720 flows, 28 % attack, all 15 classes)
      │  ml/preprocessing/pipeline.py
      ▼
 Clean (Inf/NaN → NaN, drop duplicates, drop constant cols) → encode label
      │
      ▼
 Stratified split  train 70 % │ val 15 % │ test 15 %        ← split FIRST
      │
      ▼
 Fit median imputer + StandardScaler on TRAIN only → transform all three
      │
      ├─ Step 1  Feature selection benchmark (corr filter, MI, RF importance, ANOVA, PCA, port ablation)
      ├─ Step 2  7 baselines  + 5-fold stratified CV on the train split
      ├─ Step 3  Imbalance study (none / class_weight / under-sample / SMOTE)
      ├─ Step 4  Grid vs Random vs Optuna-TPE tuning of RF and XGBoost (same CV, same objective)
      ├─ Step 5  Ensembles: hard/soft voting, bagging, AdaBoost, gradient boosting, stacking, weighted
      ├─ Step 6  Per-attack-type recall for every model
      ├─ Step 7  Bootstrap 95 % CIs (test rows) + multi-seed retraining
      ├─ Step 8  SHAP global importance (TreeExplainer)
      └─ Step 9  master_comparison.json · results.csv · presets.json · research_conclusions.json
      │
      ▼
 generate_figures.py → docs/figures/    render_results_docs.py → docs/results.md + README table
      │
      ▼
 FastAPI backend (/api/predict, /api/explain, /api/experiments/…)  →  React dashboard
```

## 2. Design principles

| Principle | How it is enforced |
|:--|:--|
| **No leakage** | Split before fitting anything; duplicates removed before the split; SMOTE / selectors / scalers only ever see train arrays; stacking uses out-of-fold predictions; test split scored once per final model. |
| **Same conditions for every comparison** | All models see the same scaled 70-feature matrix; all search strategies share one `StratifiedKFold(5, seed 42)` and one objective (mean F1); all imbalance strategies are scored on the same test rows. |
| **Numbers are generated, never typed** | `research_conclusions.json` is written by code from the metrics; `docs/results.md` and the README leaderboard are rendered from the JSON. |
| **Uncertainty is reported** | CV mean ± std, bootstrap 95 % CIs, multi-seed retraining. |
| **Cost is reported** | Training time, inference latency per 1,000 flows and search wall-time sit next to every accuracy number, because an IDS runs at line rate. |
| **Failures are visible** | Per-attack-type recall exposes rare classes that a binary F1 hides; the port ablation tests the known shortcut feature. |

## 3. Research questions

| RQ | Question | Primary evidence |
|:-:|:--|:--|
| RQ1 | How accurately can individual ML algorithms (LR, KNN, DT, SVM, RF, ET, XGB) detect malicious flows? | leaderboard, CV table |
| RQ2 | Does feature selection / PCA improve performance or training efficiency? | feature benchmark (F1 + time per subset) |
| RQ3 | How do tuned models compare with defaults, and how do Grid, Random and Bayesian search compare in accuracy and cost? | search comparison table, Optuna convergence |
| RQ4 | Do ensembles (voting, bagging, boosting, stacking) outperform the best single model? | leaderboard, bootstrap CIs |
| RQ5 | Which model gives the best trade-off between recall, false alarms and latency? | recall-vs-FPR and latency figures |
| RQ6 | Can the model explain its decisions? | SHAP global importance + per-flow drivers in the API |
| RQ7 | Which imbalance strategy works best? | imbalance study |
| RQ8 | Which attack categories are still missed? | per-attack-type recall |

## 4. Metrics

Accuracy, precision, recall, F1, ROC-AUC, PR-AUC, specificity, FPR, FNR, confusion matrix, training time and inference latency (ms per 1,000 flows). F1 is the model-selection criterion; recall and FPR are reported alongside because their operational costs are asymmetric (a missed intrusion ≫ a false alarm). See `docs/evaluation.md`.

## 5. Statistical protocol

* **Cross-validation**: `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` on the 70 % train split; reported as mean ± std and train-vs-validation gap.
* **Bootstrap**: 200 resamples of the 8,081 test rows with replacement; 2.5/97.5 percentiles give a 95 % CI for F1, recall, accuracy and FPR (model fixed).
* **Multi-seed**: LR, DT, RF and XGB retrained with seeds 42 / 123 / 2024 on the same split.
* **Decision rule for "best model"**: highest F1; ties within 0.0005 broken by lower FPR, then lower latency.

## 6. Compute budget

Optuna 20 trials, RandomizedSearchCV 12 iterations, GridSearchCV 18-point grids, 5 folds, 200 bootstrap resamples. Full run ≈ 1 hour on an 8-core laptop; `--quick` mode ≈ 4 minutes. Step timings are recorded in `master_comparison.json`.
