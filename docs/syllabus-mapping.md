# Syllabus → Code Mapping

The project brief lists these topics: *Introduction to ML, Data Preprocessing, Missing Value Handling, Encoding, Scaling, PCA, Feature Selection, Imbalanced Dataset Handling, Logistic Regression, KNN, Decision Tree, SVM, Ensemble Learning (Bagging & Boosting), Evaluation Metrics, Cross Validation, Grid Search, Hyperparameter Tuning.*

This page shows **exactly where each topic is implemented**, what it does on this dataset, and where the result is recorded. Every result file lives in `ml/artifacts/`; every figure in `docs/figures/`.

| # | Syllabus topic | Where it is implemented | What it does here | Where the result lands |
|:-:|:--|:--|:--|:--|
| 1 | **Introduction to ML** (supervised classification) | `ml/training/run_all_experiments.py` (whole pipeline) | Binary supervised classification: *is this network flow BENIGN or an ATTACK?* 70 numeric features → 1 label. | `docs/walkthrough.md` |
| 2 | **Data Preprocessing** | `ml/preprocessing/pipeline.py` | Fixed order: load → clean → encode → **split** → fit transformers on train only → transform all splits. | `ml/artifacts/metrics/dataset_summary.json` |
| 3 | **Missing Value Handling** | `ml/preprocessing/cleaning.py` (`DataCleaner`), `ml/preprocessing/scaling.py` (`SimpleImputer(strategy="median")`) | `Flow Bytes/s` and `Flow Packets/s` contain `Infinity`/`NaN` (division by a zero-duration flow). They are coerced to NaN, then imputed with the **train-split median**. Median (not mean) because these columns are heavy-tailed. | `cleaning_audit.nan_inf_cells_found` in `dataset_summary.json` |
| 4 | **Encoding** | `ml/preprocessing/encoding.py` (`LabelEncoderIDS`) | Target encoding: 15 string labels → `0 = BENIGN`, `1 = ATTACK` (binary mode). A multiclass map is kept for the per-attack-type analysis. All input features are already numeric (CICFlowMeter output), so no one-hot encoding is needed — this is stated, not skipped. | `classes` in `dataset_summary.json` |
| 5 | **Scaling** | `ml/preprocessing/scaling.py` (`NetworkFlowPreprocessor` → `StandardScaler`) | z-score scaling `(x − μ_train) / σ_train`. Required for distance/margin models (KNN, SVM, Logistic Regression); harmless for trees. Fitted on train only. | `ml/artifacts/preprocessors/preprocessor.joblib` |
| 6 | **PCA** | `ml/feature_engineering/dimensionality_reduction.py` (`PCAReducer`) | Fits PCA on the train split, keeps the smallest number of components explaining ≥ 95% variance, then trains RF/LR on the projected data to measure the accuracy cost of compression. | `pca_summary.json`, figure 08, benchmark row `pca_95pct_variance` |
| 7 | **Feature Selection** | `ml/feature_engineering/selection.py` (`FeatureSelector`) | Four methods, all fitted on train only: Pearson correlation filter (r > 0.90), mutual information top-25, Random-Forest Gini importance top-25, ANOVA `SelectKBest(f_classif)` top-25. Each subset is **benchmarked** (F1 + train time) — not just listed. Plus an ablation removing `Destination Port`. | `feature_selection_summary.json` → `benchmark`, figure 07 |
| 8 | **Imbalanced Dataset Handling** | `ml/preprocessing/resampling.py` (`run_imbalance_study`) | Compares **none / class_weight="balanced" / RandomUnderSampler / SMOTE** on LR, DT and RF with identical test data. Resampling is applied to the train split only. | `imbalance_study.json`, figure 11, RQ7 |
| 9 | **Logistic Regression** | `ml/models/baselines.py` | Linear baseline (`max_iter=2000`, balanced class weights). Deliberately the weakest model — it shows the decision boundary is non-linear. | leaderboard row |
| 10 | **KNN** | `ml/models/baselines.py` | `KNeighborsClassifier(k=5)` on scaled features. Near-zero training time, slowest inference (it stores the whole train set). | leaderboard row, figure 14 |
| 11 | **Decision Tree** | `ml/models/baselines.py` | CART with Gini impurity, unlimited depth, balanced class weights. Its overfit gap in the CV table motivates ensembles. | leaderboard + CV table |
| 12 | **SVM** | `ml/models/baselines.py`, also a base learner in Voting/Stacking | RBF-kernel `SVC` with Platt-scaled probabilities. Its O(n²) training cost is why the sample is 55k rows and why timing is reported. | leaderboard row |
| 13 | **Ensemble Learning — Bagging** | `ml/models/ensembles.py` (`bagging_extra_trees`), Random Forest / Extra Trees in baselines | Bootstrap samples + random feature subsets, averaged. Reduces variance of deep trees. | leaderboard rows |
| 14 | **Ensemble Learning — Boosting** | `ml/models/ensembles.py` (`adaboost`, `gradient_boosting`), XGBoost in baselines and Optuna | Sequential learners that fit the previous learner's errors. XGBoost adds second-order gradients and L2 leaf regularisation. | leaderboard rows |
| 15 | **Ensemble Learning — Voting / Stacking** (beyond syllabus) | `ml/models/ensembles.py` (`voting_hard`, `voting_soft`, `stacking`, `weighted_ensemble`) | Heterogeneous base learners (RF + XGB + LR + SVM). Stacking trains a logistic-regression meta-learner on 5-fold **out-of-fold** predictions to avoid leakage. The weighted blend tunes its weights on the *validation* split — the first time the val split is consumed. | leaderboard rows, RQ4/RQ5 |
| 16 | **Evaluation Metrics** | `ml/training/evaluate.py` (`evaluate_model`) | Accuracy, precision, recall, F1, ROC-AUC, PR-AUC, specificity, FPR, FNR, confusion matrix, inference latency. Recall/FNR are emphasised because a missed attack costs more than a false alarm. | `master_comparison.json`, figures 02–05 |
| 17 | **Cross Validation** | `ml/training/evaluate.py` (`cross_validate_classifier`), used inside every search in `optimization.py` and inside `StackingClassifier(cv=5)` | `StratifiedKFold(5, shuffle=True, seed=42)` on the **train split only**. Reports mean ± std per fold and the train-vs-validation gap. The test split is never part of any fold. | `cv_results.json`, figure 06 |
| 18 | **Grid Search** | `ml/models/optimization.py` (`grid_search_random_forest`, `grid_search_xgboost`) | `GridSearchCV` over an explicit 18-point grid (RF: n_estimators × max_depth × max_features; XGB: n_estimators × max_depth × learning_rate). Records configs evaluated, wall time, best CV F1, then test F1. | `optimization_study.json` → `search_comparison`, figure 09 |
| 19 | **Hyperparameter Tuning** (Random + Bayesian) | `ml/models/optimization.py` (`random_search_*`, `optimize_*` with Optuna TPE) | Same CV splitter and objective as Grid Search so the three strategies are directly comparable in accuracy *and* cost. Optuna's TPE sampler is Bayesian: it models which regions of the space produced good scores and samples there. | figures 09 and 10, RQ3 |

## Beyond the syllabus (why these are in the project)

| Addition | Why it matters for the examiner |
|:--|:--|
| **Zero-leakage protocol** (`docs/preprocessing.md`) | The most common way student projects produce fake 100% accuracy is fitting the scaler/SMOTE/feature selector on the whole dataset. Here every fitted object sees only the train split, and duplicates are removed *before* splitting. |
| **Per-attack-type recall** (`per_class_recall.json`) | A 0.99 F1 can hide a completely missed rare class (Heartbleed has 11 rows in 2.8M). This table proves which categories are caught. |
| **Bootstrap 95% CIs + multi-seed retraining** (`statistical_stability.json`) | Shows whether a 0.001 difference between two models is real or noise. |
| **SHAP explainability** (`shap_global.json`) | Answers "why did it flag this flow?" with Shapley values — and exposes the `Destination Port` shortcut, which the ablation then tests. |
| **Real-data presets + FastAPI + React dashboard** | The model is deployable, not just a notebook: `POST /api/predict` scores a flow and returns SHAP drivers. |

## Where each research question is answered

| RQ | Question | Answer source |
|:-:|:--|:--|
| RQ1 | How well do individual algorithms detect attacks? | `research_conclusions.json → rq1_individual_baselines`, leaderboard |
| RQ2 | Does feature selection / PCA help? | `rq2_feature_selection`, feature benchmark table |
| RQ3 | Tuned vs untuned, and Grid vs Random vs Bayesian? | `rq3_optimization_impact`, search comparison table |
| RQ4 | Do ensembles beat single models? | `rq4_ensemble_superiority` |
| RQ5 | Best trade-off (recall / false alarms / latency)? | `rq5_optimal_tradeoff` |
| RQ6 | Can the model explain itself? | `rq6_explainability`, SHAP table |
| RQ7 | Which imbalance strategy works? | `rq7_imbalance_handling`, imbalance table |
| RQ8 | Which attack types are still missed? | `rq8_per_attack_type`, per-class recall table |
