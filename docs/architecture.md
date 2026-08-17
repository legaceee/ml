# System Architecture & Component Interaction

## 1. End-to-End Architectural Blueprint

```
+-------------------------------------------------------------------------------+
|                             CLIENT / USER LAYER                               |
|   React 18 + Vite + TypeScript + Tailwind CSS (SOC Defensive Cyber Theme)     |
|   - Real-time Flow Inspector          - Drag-and-Drop Batch CSV Predictor     |
|   - Benchmark Model Leaderboard       - Optuna Optimization Visualizer        |
|   - Ensemble Paradigms Deep-Dive      - SHAP Waterfall Attributions Explorer   |
|   - Research Conclusions Generator    - Dataset Schema & Split Auditor        |
+---------------------------------------+---------------------------------------+
                                        | (REST JSON / Multipart Form-Data)
                                        v
+-------------------------------------------------------------------------------+
|                            FASTAPI BACKEND SERVICE                            |
|   - Input Pydantic Schema Validation  - Session Management / CORS Middleware  |
|   - Batch CSV Processing Stream       - Health Checks & Error Handling        |
|   - Model Registry Service            - SHAP Attributions Caching Engine      |
+---------------------------------------+---------------------------------------+
                    |                                       |
                    v                                       v
+---------------------------------------+   +-----------------------------------+
|          SQLITE DATABASE              |   |       SERIALIZED ML ARTIFACTS     |
|  - PredictionLog (Audit trail)        |   |  - Preprocessor Pipeline (.joblib)|
|  - ModelRegistryEntry (Metadata)      |   |  - 12 Trained Models (.joblib)    |
|  - ExperimentRecord (Metrics history) |   |  - SHAP Background Profiles (.json|
+---------------------------------------+   +-----------------------------------+
```
