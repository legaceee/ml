# Capstone Research Methodology

## 1. Experimental Workflow

The project follows a rigorous, reproducible 10-stage machine learning lifecycle:

```
[Raw Network Traffic / PCAP Flow CSV]
                  │
                  ▼
         [Data Sanitization]
    (Strip whitespace, NaN/Inf imputation, constant column removal)
                  │
                  ▼
     [Stratified Partitioning]
    (70% Train, 15% Validation, 15% Test)
                  │
                  ▼
    [Pipeline Fitting (Train Only)]
    (StandardScaler + RobustScaler + Median Imputer)
                  │
                  ▼
        [Feature Selection]
    (Collinearity Filtering, Mutual Information, Tree Importance, PCA)
                  │
                  ▼
      [Baseline Model Training]
    (LR, DT, RF, ExtraTrees, SVM, KNN, XGBoost)
                  │
                  ▼
   [Bayesian Optuna Optimization]
    (5-Fold Stratified CV on Train split, maximizing F1)
                  │
                  ▼
     [Ensemble Model Synthesis]
    (Hard/Soft Voting, Bagging, Boosting, Stacking)
                  │
                  ▼
  [Untouched Test Set Evaluation]
    (Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, FPR, FNR, Latency)
                  │
                  ▼
   [SHAP Explainability & Deployment]
    (Global Summary Plots, Local Waterfall Attributions, FastAPI Backend, React SOC UI)
```

---

## 2. Research Questions Formulation

- **RQ1**: How accurately can individual machine-learning algorithms detect malicious network traffic?
- **RQ2**: Does feature selection / dimensionality reduction improve model performance and training efficiency?
- **RQ3**: How do optimized models compare with their unoptimized versions?
- **RQ4**: Do ensemble methods outperform individual classifiers?
- **RQ5**: Which ensemble strategy provides the best trade-off between Accuracy, Precision, Recall, F1, ROC-AUC, FPR, and Inference Latency?
- **RQ6**: Can the final model provide interpretable explanations for why traffic was classified as malicious?
