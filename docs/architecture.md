# System Architecture

```
+------------------------------------------------------------------------------+
|                          REACT DASHBOARD  (frontend/)                        |
|  Vite + React 18 + TypeScript + Tailwind + Recharts                          |
|  SOC Dashboard · Live Flow Inspector · Batch CSV · Model Leaderboard          |
|  Optuna Optimization · Ensemble Deep-Dive · SHAP Explainability              |
|  Research Findings (RQ1-8) · Course Topics Lab · Dataset Explorer            |
+-------------------------------------+----------------------------------------+
                                      | REST / JSON  (http://localhost:8000/api)
                                      v
+------------------------------------------------------------------------------+
|                          FASTAPI BACKEND  (backend/app/)                     |
|  api/predict.py      POST /predict, /predict/batch, GET /predict/test-samples|
|  api/explain.py      GET  /explain/global/{model}, POST /explain/local       |
|  api/models.py       GET  /models, /models/{id}                              |
|  api/metrics.py      GET  /metrics, /metrics/{id}                            |
|  api/experiments.py  GET  /experiments, /conclusions, /optimization-study,   |
|                           /feature-selection, /artifact/{cv-results |        |
|                           imbalance-study | per-class-recall | search-       |
|                           comparison | shap-global | pca-summary}            |
|  api/dataset.py      GET  /dataset/summary, /features, /presets              |
|  services/ml_service.py  singleton: loads preprocessor + 18 models, builds   |
|                          feature vectors, runs inference + SHAP              |
+------------------+---------------------------------------+-------------------+
                   |                                       |
                   v                                       v
+------------------------------+   +------------------------------------------+
|  SQLITE (backend/app/database)|   |  ML ARTIFACTS  (ml/artifacts/)           |
|  PredictionLog (audit trail)  |   |  preprocessors/preprocessor.joblib       |
|  ModelRegistryEntry           |   |  models/*.joblib  (18 models)            |
|  ExperimentRecord             |   |  metrics/*.json   (11 result files)      |
+------------------------------+   |  experiments/results.csv, conclusions    |
                                   +------------------------------------------+
                                                       ^
                                                       | written by
+------------------------------------------------------------------------------+
|                          ML PIPELINE  (ml/)                                  |
|  data/build_dataset.py  -> download 8 day files, class-capped sample         |
|  preprocessing/pipeline.py -> clean, encode, split, fit-on-train transforms   |
|  training/run_all_experiments.py -> 9 steps (see docs/methodology.md)         |
|  training/generate_figures.py -> docs/figures/*.png                          |
|  training/render_results_docs.py -> docs/results.md, README leaderboard      |
+------------------------------------------------------------------------------+
```

## Request flow for a live prediction

1. Dashboard posts `{model_name, features: {...}}` to `POST /api/predict`.
2. `MLService.prepare_feature_vector` starts from the **benign median profile** of the train split, overlays the provided features, and recomputes physically dependent fields (bytes/s, packets/s, segment sizes, subflow mirrors) so a partial input is still a coherent flow.
3. The frozen `NetworkFlowPreprocessor` (median imputer + StandardScaler fitted on train) transforms the row.
4. The chosen model predicts; class indices are read from `model.classes_` (never assumed).
5. `IDSExplainer` returns the top SHAP drivers toward the predicted class.
6. Response: label, probabilities, latency, heuristic attack family, SHAP drivers; a `PredictionLog` row is written.

## Why the artifacts are JSON

Every result file is plain JSON so that three consumers read the same numbers: the docs renderer, the figure generator and the dashboard. Regenerating the experiments updates all three without editing any text by hand.

## Deployment

`Dockerfile` + `docker-compose.yml` build the API (uvicorn) and the static frontend. Models are loaded once at startup (`MLService` singleton); SHAP explainers are built lazily per model and cached.
