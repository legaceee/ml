# Cloud Hand-off: finishing the benchmark on Claude Code (web)

This page exists because the full experiment run takes **1–2 hours of CPU** and the laptop it was started on
cannot stay open overnight. Everything needed to finish is in the repository; a cloud Claude Code session
(claude.ai/code) can complete the remaining work unattended and push the result.

Section 1 is for you (the human). Section 2 is the exact prompt to paste. Section 3 is the technical
checklist the agent follows — keep it in the repo so the agent can read it.

---

## 1. What you do (5 minutes)

1. Open **https://claude.ai/code** and sign in with the same account that owns Claude Code.
2. Connect the GitHub repository **`legaceee/ml`** (branch `main`). The web version clones the repo into a cloud sandbox with its own CPU — your laptop can be off.
3. Start a new session on that repo and paste the **entire prompt in Section 2** as the first message.
4. Leave it. The session runs the ~1–2 hour benchmark, regenerates every figure and table, writes the two missing documents, runs the tests and pushes.
5. In the morning, open the repo on GitHub:
   * If the agent pushed straight to `main`, pull it: `git pull origin main`.
   * If it opened a **pull request** instead (the web sandbox sometimes works on a `claude/...` branch), read the PR summary, then **Merge**.
6. Then read, in this order: `docs/walkthrough.md` → `docs/syllabus-mapping.md` → `docs/presentation-script.md` → `docs/viva-defense-guide.md`.

If the cloud session cannot install packages or cannot push, it will say so in its final message — the fallback is Section 4 (run the same four commands locally when you have ~2 hours of laptop time).

---

## 2. The prompt to paste into Claude Code (web)

Copy everything inside the fence, unchanged.

```text
You are finishing a final-year ML capstone: "Cyber Attack Detection Using Optimized Machine Learning Models and Ensemble Methods" (network intrusion detection on the real CIC-IDS2017 dataset). The code, data sample and documentation scaffolding are complete; what is missing is the FULL experiment run and the documents that are written from its numbers. The student submits tomorrow morning, so finish everything in this session and push.

First read docs/CLOUD_HANDOFF.md (section 3 is your checklist), then docs/reproducibility.md, then README.md. Do not change the methodology, the data sample, the seeds, the search budgets or the model set — the point of the run is to produce the honest numbers for the pipeline exactly as written.

Tasks, in order:

1. Environment: python 3.11 or 3.12. `pip install -r requirements.txt` and `pip install jupyter nbconvert`. The 19 MB dataset sample ml/data/raw/cicids2017_sample.csv is committed — do NOT download the 885 MB corpus.
2. Smoke test first: `python -m ml.preprocessing.pipeline` then `python -m ml.training.run_all_experiments --quick` (~5 min). If anything crashes, fix the bug minimally, keep the fix, and re-run the smoke test until it passes.
3. Full run: `python -m ml.preprocessing.pipeline` (regenerates the splits at full size) then `python -m ml.training.run_all_experiments` with stdout going to ml/artifacts/experiments/run_log.txt. Expect 1–2 hours; the RBF-SVM baseline alone is ~9 minutes and the SVC-based ensembles ~30 minutes. Run it in the background and keep working on task 5 while it runs. Do not shorten it unless it has been running for more than 4 hours — in that case, and only then, stop it, run `python -m ml.data.build_dataset --build --benign-rows 25000 --attack-cap 1500`, re-run steps 3 onwards, and state the reduced sample size explicitly in docs/walkthrough.md and README.md.
4. After the run: `python -m ml.training.generate_figures` (docs/figures/*.png, 15 files) and `python -m ml.training.render_results_docs` (docs/results.md + the LEADERBOARD and KEY-FINDINGS blocks in README.md). Then `python notebooks/generate_notebooks.py` and `jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 notebooks/*.ipynb` so the notebooks carry real outputs (if a notebook needs more than 30 minutes, skip executing that one and say so).
5. Write the two missing documents, following the specification in docs/CLOUD_HANDOFF.md section 3.4: docs/walkthrough.md and docs/presentation-script.md. Every number in them must be read from ml/artifacts/metrics/*.json, ml/artifacts/experiments/results.csv or research_conclusions.json — never typed from memory. Embed the figures. Then search all of docs/ and README.md for leftover placeholder text ("after the run", "Run the experiments and", "TBD", "TODO", "{{") and replace it with the real content.
6. Sanity-check the story before publishing (section 3.5 of the hand-off has the checklist): no model may show a perfect 1.0000 on every metric; ROC-AUC must not be 0.5 for hard voting; the search-comparison table must have Grid, Random and Optuna rows for both RF and XGBoost; the per-attack-type table must list all 14 attack classes; research_conclusions.json must not describe a negative delta as an "improvement". If a check fails, find the cause in the code, fix it, and re-run only what is necessary.
7. `python -m pytest tests/ -q` must pass (integration tests use the trained artifacts). Fix failures; do not delete tests.
8. Commit with a clear message ("Full CIC-IDS2017 benchmark run: results, figures, walkthrough, presentation script") and push to origin main. If you are not allowed to push to main, push a branch and open a pull request to main with a summary of the run (leaderboard top 5, run time, any deviation from the plan).
9. Final message: list exactly what was run, the wall-clock time of the benchmark, the top-5 leaderboard, anything that deviated from this plan and why, and any check from step 6 that could not be satisfied.

Constraints: never fabricate or hand-edit a number; if a metric looks suspicious, investigate rather than smooth it over. Do not add new models or new experiments. Do not touch frontend/ except to fix a build error. Keep docs in the same plain, factual tone as docs/reproducibility.md.
```

---

## 3. Technical checklist (what the agent actually does)

### 3.1 State of the repository at hand-off

| Area | State |
|:--|:--|
| Data | Real CIC-IDS2017, all 8 day-files merged (2,830,743 flows), class-capped stratified sample of **55,720 rows** (40,000 benign + up to 1,500 per attack class, all 14 attack types present, seed 42). Committed as `ml/data/raw/cicids2017_sample.csv` with `cicids2017_sample_manifest.json`. |
| Preprocessing | `ml/preprocessing/` — cleaning (inf/NaN, duplicates, constant columns), label encoding, median imputation + `StandardScaler` fitted on the training split only; 70/15/15 stratified split. `ml/preprocessing/resampling.py` — class-weight / random under-sampling / SMOTE (train split only). |
| Experiments | `ml/training/run_all_experiments.py` — 9 steps: feature-selection benchmark (measured, 5 subsets), 7 baselines + 5-fold CV, imbalance study, Grid vs Random vs Optuna (RF and XGBoost), 7 ensembles, per-attack-type recall, bootstrap CIs + 3-seed retraining, SHAP, conclusions RQ1–RQ8. |
| Figures / docs | `ml/training/generate_figures.py` (15 PNGs) and `ml/training/render_results_docs.py` (results.md + README blocks) are written and were tested on a `--quick` run. The PNGs currently in `docs/figures/` are from that quick run and **must be regenerated**. |
| App | FastAPI backend (`backend/`), React dashboard (`frontend/`) incl. new *Course Topics Lab* page reading the new JSON artifacts. |
| Tests | 26 tests in `tests/`; the integration ones skip when artifacts are missing. |
| Last local run | Reached step 4/9 (search comparison) and was stopped. `ml/artifacts/metrics/*.json` therefore hold a **mix** of quick-run and partial full-run output — treat them as invalid until the full run overwrites them. Trained models are git-ignored. |

### 3.2 Runtime expectations (8-core laptop, 37,706 training rows)

| Step | Approx. time | Bottleneck |
|:--|--:|:--|
| 1 Feature-selection benchmark | 4 min | RF/XGB on 5 subsets |
| 2 Baselines + CV | 12 min | RBF-SVM with `probability=True` ≈ 9 min |
| 3 Imbalance study | 3 min | SMOTE + 4 strategies × 3 models |
| 4 Grid / Random / Optuna | 25–35 min | 5-fold CV × (36 grid + 12 random + 20 Optuna configs) × 2 algorithms |
| 5 Ensembles | 25–35 min | soft voting, stacking and weighted blend each fit an SVC |
| 6–9 Per-class recall, bootstrap, seeds, SHAP, conclusions | 10–15 min | 3-seed retraining of the candidates |

### 3.3 Commands

```bash
pip install -r requirements.txt && pip install jupyter nbconvert
python -m ml.preprocessing.pipeline
python -m ml.training.run_all_experiments --quick                 # smoke test, ~5 min
python -m ml.preprocessing.pipeline                               # full-size splits again
python -m ml.training.run_all_experiments > ml/artifacts/experiments/run_log.txt 2>&1
python -m ml.training.generate_figures
python -m ml.training.render_results_docs
python notebooks/generate_notebooks.py
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 notebooks/*.ipynb
python -m pytest tests/ -q
```

### 3.4 Specification of the two documents to write

**`docs/walkthrough.md` — "what happened and how"** (the document the student reads to explain the project). Plain language, every claim backed by a number from the artifacts and, where one exists, the matching figure embedded with a relative path (`figures/NN_*.png`). Sections:

1. **The problem in one paragraph** — flows, attacks, why signatures fail, why behaviour works.
2. **The data** — full corpus vs sample (numbers from `cicids2017_sample_manifest.json`), why class-capped sampling (RBF-SVM/KNN/stacking cost), what 70/15/15 gives (counts from `dataset_summary.json`), the leakage protocol.
3. **Cleaning, encoding, imputation, scaling** — what was found (inf/NaN cells, duplicates, constant columns dropped) and what was done.
4. **Feature selection and PCA** — the measured benchmark table (`feature_selection_summary.json`), PCA components for 95 % variance (`pca_summary.json`), and the honest verdict (did it help accuracy, speed, both, neither?).
5. **Imbalanced data** — the four strategies and the table from `imbalance_study.json`; which won and why it makes sense.
6. **Seven baseline models** — test F1 next to 5-fold CV mean ± std (`cv_results.json`, `master_comparison.json`); explain the spread (linear vs tree vs distance models).
7. **Grid vs Random vs Optuna** — table from `optimization_study.json` (best CV F1, test F1, configs evaluated, wall time) for RF and XGBoost, plus the convergence figure; a one-paragraph verdict on cost vs benefit.
8. **Ensembles** — voting/bagging/boosting/stacking/weighted results and whether they beat the best tuned single model (with the bootstrap CI from `statistical_stability.json` saying whether the gap is real).
9. **Per-attack-type recall** — the table from `per_class_recall.json`; call out the rare classes (Heartbleed 11 flows, SQL injection 21, Infiltration 36) honestly.
10. **Stability** — bootstrap 95 % CIs and 3-seed retraining.
11. **Explainability** — top SHAP features and what they mean physically; the Destination-Port ablation result.
12. **Limitations** — sample not full corpus, binary not multiclass, one dataset, latency measured on one machine.
13. **File map** — which file produced which table.

**`docs/presentation-script.md` — a 10-minute talk.** About 10 slides; for each: title, the one figure or table to show (by filename), 3–5 speaker-note sentences with the real numbers, and the transition. End with a "questions you will be asked" list of 8 items with two-sentence answers drawn from the results (why not 100 %? why a sample? isn't Destination Port a shortcut? why F1 not accuracy? what did tuning actually buy? what did SMOTE do? which model would you deploy and why? what would you do with more time?).

### 3.5 Sanity checklist before pushing

* No model has 1.0000 on accuracy, precision, recall, F1 **and** ROC-AUC simultaneously.
* Hard voting reports a real ROC-AUC (computed from vote fractions), not 0.5.
* `optimization_study.json` → `comparison` has six rows: {RF, XGB} × {grid, random, optuna}, each with `wall_time_sec > 0`.
* `per_class_recall.json` lists 14 attack categories for every model.
* `research_conclusions.json` never says "improved" for a negative delta; RQ answers read correctly against the numbers.
* `docs/results.md`, README leaderboard and `results.csv` agree (same F1 for the same model).
* No placeholder text remains in `README.md` or `docs/*.md`.
* `docs/figures/` has 15 PNGs newer than `master_comparison.json`.
* `python -m pytest tests/ -q` passes.

---

## 4. Fallback: run it locally instead

Same commands as 3.3, on a plugged-in laptop with sleep disabled (`powercfg /change standby-timeout-ac 0` on Windows). Start the run detached so a closed terminal does not kill it:

```powershell
Start-Process python -ArgumentList "-u","-m","ml.training.run_all_experiments" -RedirectStandardOutput ml\artifacts\experiments\run_log.txt -RedirectStandardError ml\artifacts\experiments\run_err.txt -WindowStyle Hidden
```

Then, once `run_log.txt` ends with `BENCHMARK COMPLETE`, run the figure/doc/notebook/test commands and commit.
