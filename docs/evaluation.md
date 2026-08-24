# Evaluation Metrics & Performance Analysis in Intrusion Detection

## 1. Metric Definitions

In Cyber Attack Detection, class distributions are typically imbalanced (Benign >> Attacks). Relying solely on raw Accuracy is misleading because a trivial classifier predicting 100% Benign can achieve high accuracy while failing completely as a security defense.

| Metric | Formula | Cybersecurity Significance |
| :--- | :--- | :--- |
| **Accuracy** | $\frac{TP + TN}{TP + TN + FP + FN}$ | Overall percentage of correct flow classifications. |
| **Precision** | $\frac{TP}{TP + FP}$ | Fraction of flagged alarms that are genuine cyber attacks (minimizes analyst alert fatigue). |
| **Recall (Sensitivity)** | $\frac{TP}{TP + FN}$ | **Crucial Metric**: Fraction of all actual cyber attacks detected (minimizes missed breaches). |
| **$F_1$-Score** | $2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$ | Harmonic mean of Precision and Recall; primary optimization target. |
| **Specificity (TNR)** | $\frac{TN}{TN + FP}$ | Fraction of legitimate traffic correctly permitted without false alarms. |
| **False Positive Rate (FPR)** | $\frac{FP}{FP + TN} = 1 - \text{Specificity}$ | Percentage of legitimate user sessions mistakenly blocked. |
| **False Negative Rate (FNR)** | $\frac{FN}{FN + TP} = 1 - \text{Recall}$ | **Highest Risk**: Percentage of malicious flows that bypass the IDS undetected. |
| **ROC-AUC** | $\int_{0}^{1} \text{TPR}(t) \, d(\text{FPR}(t))$ | Discriminative capability across all classification probability thresholds. |
| **Inference Latency** | $\frac{\text{Total Prediction Time (ms)}}{\text{Number of Flows}} \times 1000$ | Time required to score 1,000 network flows; critical for line-rate packet filtering. |

---

## 2. The Cost Asymmetry of Cyber Intrusion Detection
In defensive cybersecurity, errors carry highly asymmetric operational costs:
- **Cost of False Positive ($C_{FP}$)**: Operational inconvenience, security analyst verification overhead, temporary legitimate user throttling.
- **Cost of False Negative ($C_{FN}$)**: Severe enterprise compromise, unauthorized data exfiltration, ransomware encryption, compliance penalties.

$$\text{Operational Imperative: } C_{FN} \gg C_{FP}$$

Consequently, this capstone explicitly optimizes and selects models that maximize **Recall** and **$F_1$-Score** rather than raw classification accuracy alone.

---

## 3. Uncertainty and coverage

| Report | How it is computed | Why |
|:--|:--|:--|
| **5-fold CV mean ± std** (`cv_results.json`) | `StratifiedKFold(5, shuffle, seed 42)` on the train split | shows the variance of each estimate and the train-vs-validation overfit gap |
| **Bootstrap 95 % CI** (`statistical_stability.json`, leaderboard) | 200 resamples of the test rows with replacement, model fixed; 2.5/97.5 percentiles | tells whether a 0.001 F1 difference between models is real |
| **Multi-seed retraining** | LR/DT/RF/XGB refit with seeds 42/123/2024 | separates seed noise from model differences |
| **Per-attack-type recall** (`per_class_recall.json`) | recall of the binary detector for each original category | a 0.99 F1 can hide a missed rare class (11 Heartbleed rows in 2.8M) |
| **Score source** (`score_source` in every evaluation) | `predict_proba` → `decision_function` → hard labels | hard-voting has no probabilities; its ROC-AUC is computed from labels (one operating point) and flagged, never reported as a placeholder 0.5 |

## 4. Model-selection rule

Highest test F1; ties within 0.0005 broken by lower FPR, then lower inference latency. The rule is code (`MasterExperimentOrchestrator._pick_best`), not judgement, so the "best model" in every document is the same one.
