# Academic Project Overview: Cyber Attack Detection via Machine Learning

## 1. Problem Statement
Modern computer networks are continuously targeted by sophisticated, multi-vector cyber intrusions, including Distributed Denial-of-Service (DDoS) floods, brute-force credential stuffing (FTP/SSH Patator), automated port-scanning reconnaissance, and application-layer web attacks. Traditional Signature-Based Intrusion Detection Systems (SIDS) rely on deterministic pattern matching against known vulnerability signatures (e.g., Snort rules), rendering them ineffective against zero-day exploits, mutated payloads, and polymorphic attack vectors.

**Anomaly-based Machine Learning Intrusion Detection Systems (AIDS)** analyze continuous flow-based statistical properties (e.g., inter-arrival packet times, packet length distributions, TCP signaling flags) to detect behavioral deviations characteristic of malicious network traffic.

---

## 2. Core Research Hypothesis

$$\text{Research Hypothesis: } \mathcal{H}_1$$

> *"Ensemble machine learning architectures—specifically Stacking and Soft Voting meta-classifiers combining tree-based, linear, and margin-based base learners—provide significantly higher intrusion detection recall, lower false-positive rates (FPR), and superior generalization on network traffic than individual optimized machine learning models."*

---

## 3. Key Contributions
1. **Zero Data Leakage Pipeline**: Implements a strict train-only fitting protocol where preprocessing transformers, imputers, scalers, and feature selectors are fitted exclusively on the 70% training split.
2. **Systematic Hyperparameter Optimization**: Employs Optuna with the Tree-structured Parzen Estimator (TPE) algorithm under 5-fold stratified cross-validation maximizing the $F_1$-score.
3. **Comprehensive Ensemble Taxonomy**: Evaluates four core ensemble paradigms (Voting, Bagging, Boosting, and Stacking) against 7 baseline algorithms.
4. **Explainable AI (XAI) Integration**: Deploys SHAP (SHapley Additive exPlanations) to provide game-theoretic local and global feature attribution for SOC analyst verification.
5. **Statistical Stability Verification**: Assesses model performance stability across multiple random seeds ($\text{seeds} \in \{42, 123, 2024\}$) to ensure empirical findings are statistically sound.
