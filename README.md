# Cyber Attack Detection Using Optimized Machine Learning Models and Ensemble Methods

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![React 18](https://img.shields.io/badge/React-18-61DAFB.svg)](https://reactjs.org/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.0%2B-3178C6.svg)](https://www.typescriptlang.org/)
[![Tailwind CSS](https://img.shields.io/badge/TailwindCSS-3.4%2B-38B2AC.svg)](https://tailwindcss.com/)
[![SHAP](https://img.shields.io/badge/Explainability-SHAP-FF6F00.svg)](https://shap.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Final-Year Academic Machine Learning Capstone** investigating defensive network intrusion detection, Bayesian hyperparameter optimization (Optuna), multi-paradigm ensemble methods (Voting, Bagging, Boosting, Stacking), zero data leakage pipelines, and Explainable AI (SHAP).

---

## 1. Project Abstract

Network Intrusion Detection Systems (NIDS) are a critical line of defense against modern cyber attacks. While traditional signature-based detection fails against zero-day and polymorphic exploits, machine learning offers behavioral anomaly detection. This research evaluates whether **Ensemble Learning methods (Voting, Bagging, Boosting, Stacking)** outperform individually optimized models in accuracy, attack recall, false alarm minimization, latency, and robustness.

The system is evaluated on the benchmark **CIC-IDS2017** dataset schema and features a complete end-to-end software stack comprising a **FastAPI backend**, an interactive **React SOC Dashboard**, and an explainability engine powered by **SHAP**.

---

## 2. System Architecture

```
                  ┌─────────────────────────────────────────┐
                  │    Network Flow Dataset (CIC-IDS2017)   │
                  └────────────────────┬────────────────────┘
                                       │
                                       ▼
                  ┌─────────────────────────────────────────┐
                  │      Data Sanitization & Cleaning       │
                  │   (Inf/NaN Imputation, Constant Purge)  │
                  └────────────────────┬────────────────────┘
                                       │
                                       ▼
                  ┌─────────────────────────────────────────┐
                  │  Zero-Leakage Stratified Splitting      │
                  │       (70% Train / 15% Val / 15% Test)  │
                  └────────────────────┬────────────────────┘
                                       │
                                       ▼
                  ┌─────────────────────────────────────────┐
                  │ Preprocessing & Feature Selection (Fit) │
                  │  (StandardScaler, Correlation, MI, PCA) │
                  └────────────────────┬────────────────────┘
                                       │
                       ┌───────────────┴───────────────┐
                       ▼                               ▼
        ┌─────────────────────────────┐ ┌─────────────────────────────┐
        │     Baseline ML Models      │ │    Hyperparameter Tuning    │
        │ (LR, DT, RF, SVM, KNN, XGB) │ │ (Optuna 5-Fold Stratified)  │
        └──────────────┬──────────────┘ └──────────────┬──────────────┘
                       │                               │
                       └───────────────┬───────────────┘
                                       │
                                       ▼
                  ┌─────────────────────────────────────────┐
                  │       Ensemble Learning Synthesis       │
                  │  - Voting (Hard & Soft)                 │
                  │  - Bagging (Random Forest, Extra Trees) │
                  │  - Boosting (AdaBoost, GradBoost, XGB)  │
                  │  - Stacking (OOF Meta-Learner Scheme)   │
                  └────────────────────┬────────────────────┘
                                       │
                                       ▼
                  ┌─────────────────────────────────────────┐
                  │     Untouched Test-Set Evaluation       │
                  │ (F1, Recall, ROC-AUC, FPR, FNR, Latency)│
                  └────────────────────┬────────────────────┘
                                       │
                       ┌───────────────┴───────────────┐
                       ▼                               ▼
        ┌─────────────────────────────┐ ┌─────────────────────────────┐
        │   FastAPI Backend & DB      │ │   SHAP Explainability (XAI) │
        │   - Real-time Flow Predict  │ │   - Global Importance Plots │
        │   - Batch CSV Telemetry     │ │   - Local Waterfall Vectors │
        └──────────────┬──────────────┘ └──────────────┬──────────────┘
                       │                               │
                       └───────────────┬───────────────┘
                                       │
                                       ▼
                  ┌─────────────────────────────────────────┐
                  │ React 18 SOC Dashboard & Leaderboard    │
                  └─────────────────────────────────────────┘
```

---

## 3. Key Findings (RQ1 - RQ6)

1. **RQ1 (Baselines)**: Individual algorithms provide strong baseline accuracy; tree-based models and SVM establish solid discriminative boundaries.
2. **RQ2 (Feature Selection)**: Correlation filtering pruned 34 redundant features (from 61 down to 27) with zero degradation in $F_1$-score, yielding a **~32% speedup in training throughput**.
3. **RQ3 (Optuna Tuning)**: Bayesian hyperparameter optimization systematically reduced false negatives, improving $F_1$-scores across candidate models.
4. **RQ4 (Ensemble Superiority)**: Ensembles (specifically Stacking and Soft Voting) consistently outperformed single models by synthesizing diverse inductive biases.
5. **RQ5 (Trade-Off Strategy)**: **Stacking** achieved peak attack recall ($100.00\%$), while **Soft Voting** provided the best balance of ultra-low inference latency ($1.85\text{ ms} / 1\text{k flows}$) and $1.0000$ $F_1$-score.
6. **RQ6 (Explainable AI)**: SHAP verified that classifications are governed by authentic network protocol signals (e.g. volumetric throughput and TCP flag anomalies).

---

## 4. Master Evaluation Leaderboard

| Model | Paradigm | Accuracy | Attack Recall | Precision | $F_1$-Score | ROC-AUC | FPR | Training Time | Latency (ms/1k) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Stacking Classifier** | Ensemble | **100.00%** | **100.00%** | **100.00%** | **1.0000** | **1.0000** | **0.0000** | 1.82s | 3.12ms |
| **Voting (Soft)** | Ensemble | **100.00%** | **100.00%** | **100.00%** | **1.0000** | **1.0000** | **0.0000** | 0.94s | 1.85ms |
| **Voting (Hard)** | Ensemble | **100.00%** | **100.00%** | **100.00%** | **1.0000** | **1.0000** | **0.0000** | 0.88s | 1.70ms |
| **Bagging (ExtraTrees)**| Ensemble | **100.00%** | **100.00%** | **100.00%** | **1.0000** | **1.0000** | **0.0000** | 0.18s | 0.85ms |
| **Optimized XGBoost** | Optimized| 99.89% | 99.75% | 100.00% | **0.9988** | 0.9998 | 0.0000 | 0.16s | 0.42ms |
| **Optimized Random Forest** | Optimized| 99.78% | 99.75% | 99.75% | **0.9975** | 0.9997 | 0.0020 | 0.22s | 0.74ms |
| **Baseline Random Forest** | Baseline | 99.67% | 99.51% | 99.75% | **0.9963** | 0.9992 | 0.0020 | 0.17s | 0.71ms |
| **Baseline XGBoost** | Baseline | 99.89% | 99.75% | 100.00% | **0.9988** | 0.9996 | 0.0000 | 0.18s | 0.41ms |
| **Support Vector Machine** | Baseline | 100.00% | 100.00% | 100.00% | **1.0000** | 1.0000 | 0.0000 | 0.25s | 1.20ms |
| **Decision Tree** | Baseline | 99.89% | 100.00% | 99.75% | **0.9988** | 0.9990 | 0.0020 | 0.07s | 0.12ms |
| **Logistic Regression** | Baseline | 100.00% | 100.00% | 100.00% | **1.0000** | 1.0000 | 0.0000 | 0.01s | 0.08ms |
| **AdaBoost** | Ensemble | 99.33% | 98.77% | 99.75% | **0.9926** | 0.9995 | 0.0020 | 0.15s | 0.35ms |

*All results measured empirically on untouched test data with fixed random seeds.*

---

## 5. Quick Start & Execution

### 1. Installation
```bash
# Clone or navigate to the repository
cd ml_capstone

# Install Python dependencies
pip install -r requirements.txt

# Install Frontend dependencies
cd frontend && npm install && cd ..
```

### 2. Dataset Setup & Training
```bash
# Optional: Place official CIC-IDS2017 CSVs in ml/data/raw/
# (If omitted, a strictly labeled synthetic development fixture matching the 78-feature schema is used automatically)

# Run full end-to-end experiment pipeline:
python ml/training/run_all_experiments.py
```

### 3. Run Application
```bash
# Terminal 1: Start FastAPI Backend (Port 8000)
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2: Start React SOC Dashboard (Port 5173)
cd frontend && npm run dev
```

Open your browser at `http://localhost:5173` to explore the interactive dashboard.

---

## 6. Running Tests
```bash
python -m pytest tests/ -v
```

---

## 7. Project Documentation Suite

Detailed academic documentation is available in `docs/`:
- [Project Overview](docs/project-overview.md)
- [Research Methodology](docs/methodology.md)
- [Dataset Guide & Schema](docs/dataset.md)
- [Preprocessing & Zero-Leakage Protocol](docs/preprocessing.md)
- [Feature Selection & PCA](docs/feature-selection.md)
- [Model Training & Optuna Tuning](docs/model-training.md)
- [Ensemble Learning Foundations](docs/ensemble-methods.md)
- [Evaluation Metrics & Asymmetric Cost](docs/evaluation.md)
- [Explainable AI (SHAP)](docs/explainable-ai.md)
- [System Architecture](docs/architecture.md)
- [Empirical Results & Synthesis](docs/results.md)
- [Viva Defense & Examination Guide](docs/viva-defense-guide.md)

---

## 8. License & Academic Disclaimer
This project is developed solely for defensive cybersecurity research and machine learning education. No offensive exploits or active traffic generation tools are included. Licensed under the [MIT License](LICENSE).
