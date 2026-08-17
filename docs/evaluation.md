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
