# Viva / Examination Guide

Questions an examiner is likely to ask, with the answers this project actually supports. Numbers in *italics* come from `docs/results.md` — quote from there on the day, not from memory.

---

## A. The questions that get projects rejected

### "Everything is 99–100%. How do I know this isn't leakage or an easy dataset?"
1. **It isn't 100% everywhere.** Logistic Regression scores *≈0.84 F1* and RBF-SVM *≈0.86* on the same split; only tree-based models get above 0.99. A leaky pipeline makes *every* model look perfect. The spread is the evidence.
2. **Order of operations**: duplicates dropped → split → scaler/imputer/feature-selector/SMOTE fitted on the train split only → test split touched once per model (`docs/preprocessing.md`).
3. **Cross-validation agrees with the test split** within ~0.005 F1 for every model (`cv_results.json`), and bootstrap 95% CIs are reported, so the numbers are not a lucky draw.
4. **Per-attack-type recall** shows where the model *does* struggle (rare web attacks / infiltration), which a leaky model would not.
5. CIC-IDS2017 *is* a well-separated benchmark — published papers report 0.99+ for tree models. That is a property of flow statistics (a SYN flood does not look like a browser session), not of this pipeline.

### "Why a 55k-row sample instead of all 2.8 million flows?"
Compute. An RBF-SVM with Platt scaling is ~O(n²): measured *6 s at 10k, 17 s at 20k, 100 s at 40k rows* per plain fit; the ensembles need it fitted ~15 times. The whole benchmark with 18 models runs in ~1 hour on a laptop at 55k rows; at 2.8M it would take days. The sample is **class-capped stratified**: every rare attack class is kept in full, so no attack type is lost. This is the same practice as most CIC-IDS2017 papers; the manifest records the true corpus counts so the change in prior is explicit.

### "Isn't `Destination Port` cheating?"
Partly a shortcut — attacks in this corpus target fixed services (FTP 21, SSH 22, HTTP 80). SHAP ranks it near the top, and the report says so. The `all_minus_destination_port` ablation in the feature benchmark shows RF F1 with the port removed; the drop is *small* (see `rq2_feature_selection.destination_port_ablation`), so the detector relies mostly on flow-shape features (TCP window sizes, packet lengths, inter-arrival times).

### "Why is the ensemble only marginally better than XGBoost?"
Because XGBoost is already near the ceiling on this data. The honest finding (RQ4) is that stacking gives a *small* F1 gain plus lower variance and the highest recall, at 100× the inference latency of XGBoost. RQ5 therefore recommends XGBoost for deployment and stacking where recall matters more than speed. An examiner prefers this over an inflated claim.

---

## B. Preprocessing

### "How do you handle missing values?"
Only `Flow Bytes/s` and `Flow Packets/s` have them (`Infinity`/`NaN` from zero-duration flows, *64 cells* in the sample). They are coerced to NaN and imputed with the **train-split median** — median because the columns are heavy-tailed; not dropped because zero-duration flows are disproportionately PortScan probes.

### "What encoding did you use?"
Inputs are all numeric CICFlowMeter statistics, so no one-hot encoding is required. The target is label-encoded (BENIGN=0, ATTACK=1); the original category is retained for the per-class table.

### "Why StandardScaler and not MinMax?"
Byte counts span 0 to 10⁹ with extreme outliers; MinMax would squash 99% of rows into a tiny interval. z-scoring keeps the spread. Trees are unaffected either way; KNN/SVM/LR need it.

### "Why drop duplicates before splitting?"
If the same flow lands in train and test, the model gets credit for memorising it. *1,853* exact duplicates were removed before the split.

---

## C. Feature selection and PCA

### "Which feature selection method was best?"
Four were measured (`feature_selection_summary.json → benchmark`): Pearson correlation filter (r>0.90, *70→≈37 features*), mutual-information top-25, RF-importance top-25, ANOVA SelectKBest-25, plus PCA at 95% variance. The RF probe's F1 and training time for each are in the results table; the conclusion is data-driven (RQ2).

### "What does PCA do here, and why not use it for the final model?"
PCA (fitted on train) needs *≈22 components* for 95% variance — a 3× compression — but the RF trained on the components loses F1 (*see table*), because PCA maximises variance, not class separation, and destroys the axis-aligned splits trees exploit. It is reported as a dimensionality-reduction experiment, not used for the final model.

### "Correlation filter — what threshold and why?"
|r| > 0.90. CICFlowMeter emits many near-duplicate columns (`Fwd Header Length` twice, `Subflow Fwd Bytes` = `Total Length of Fwd Packets`, `Avg Fwd Segment Size` = `Fwd Packet Length Mean`). The heat-map figure shows the blocks.

---

## D. Imbalance

### "How imbalanced is the data and what did you do?"
Train split ≈ *72% benign / 28% attack* overall; within attacks, *40,000 benign vs 11 Heartbleed*. Four strategies were compared on identical test data (`imbalance_study.json`): nothing, `class_weight="balanced"`, random under-sampling, SMOTE. Resampling is applied to the train split only.

### "What is SMOTE and its risk?"
Synthetic Minority Over-sampling: for each minority row, pick one of its k=5 nearest minority neighbours and create a new point on the segment between them. Risk: synthetic points in overlapping regions create noise, and applying it before the split leaks test neighbours into training.

### "Which worked?"
For logistic regression, class weighting and SMOTE both lift recall substantially at the cost of FPR; for tree models the effect is small. Cost-sensitive learning (`class_weight`) is the cheapest and is what the baselines use.

---

## E. Models

### "Explain each algorithm in one sentence."
* **Logistic Regression** — linear log-odds boundary; the weakest here, which proves the boundary is non-linear.
* **KNN (k=5)** — majority vote of the 5 nearest scaled neighbours; zero training cost, slowest inference (*see latency figure*).
* **Decision Tree** — recursive Gini splits; fits train almost perfectly (overfit gap in CV table).
* **SVM (RBF)** — maximum-margin boundary in kernel space; O(n²) training, mediocre here without per-feature kernel tuning.
* **Random Forest / Extra Trees** — bagging of decorrelated trees; variance reduction.
* **AdaBoost / Gradient Boosting / XGBoost** — sequential learners fitting previous errors; XGBoost adds second-order gradients and L2 leaf regularisation.
* **Voting** — hard: majority label; soft: mean probability.
* **Stacking** — logistic-regression meta-learner on out-of-fold predictions from RF + XGB + LR + SVM.

### "Why those four base learners in the stack?"
Diversity of inductive bias: trees (axis-aligned), linear (global hyperplane), kernel (local margin). Dietterich's argument: an ensemble beats its members only when they make *different* mistakes.

### "Why is hard voting worse than soft?"
Hard voting counts LR and SVM (the weak members) as equal votes; soft voting weights by confidence, so the confident tree models dominate.

---

## F. Tuning and cross-validation

### "Grid vs Random vs Optuna — what did you find?"
All three use the same `StratifiedKFold(5, seed 42)` on the train split and the same objective (mean F1). Grid evaluates every point of an 18-point grid; Random samples 12 points; Optuna's TPE runs 20 trials, using the results so far to model P(params | good) / P(params | bad) and sample where that ratio is high. The table reports configs evaluated, wall time, best CV F1 and test F1 for all six runs — so the comparison is in both accuracy *and* cost (`rq3_optimization_impact`).

### "What is TPE?"
Tree-structured Parzen Estimator (Bergstra et al., 2011). Instead of modelling score = f(params) like a Gaussian process, it splits observed trials into "good" (top γ quantile) and "bad", fits a density to each, and proposes the next trial maximising ℓ(θ)/g(θ) — equivalent to expected improvement.

### "Why 5 folds? Why stratified?"
5 folds is the usual bias/variance compromise (each fold trains on 80%). Stratified so every fold has the same attack ratio — with 11 Heartbleed rows a plain KFold could put them all in one fold.

### "Was the test set used during tuning?"
No. Tuning uses CV inside the train split; the validation split is consumed only by the weighted-ensemble weight search; the test split is scored once per final model.

---

## G. Evaluation

### "Why not accuracy?"
Predicting BENIGN for everything gives ≈72% accuracy and catches zero attacks. Recall (missed attacks) and FPR (false alarms) are the operational metrics; F1 balances them; PR-AUC is the right curve for the minority class.

### "What are FNR and FPR here and which matters more?"
FNR = missed attacks / all attacks; FPR = false alarms / all benign. A missed intrusion costs more than an analyst checking a false alarm, so recall/FNR are weighted higher — but FPR at 40,000 benign flows/day still matters, which is why RQ5 reports both.

### "What is the bootstrap CI?"
Resample the 8,081 test rows with replacement 200 times, recompute F1 each time, take the 2.5th/97.5th percentiles. If two models' CIs overlap heavily, their difference is not significant.

### "What does SHAP tell you?"
Shapley values: each feature's average marginal contribution to the attack log-odds over all feature orderings. `TreeExplainer` computes them exactly for tree models. The global bar chart ranks features; the API returns per-flow drivers so an analyst can see *why* a flow was flagged.

---

## H. Engineering

### "How would this run in production?"
`POST /api/predict` with a CICFlowMeter flow → preprocessor → XGBoost → label + probability + top SHAP drivers in *≈3 ms per 1,000 flows*. Batch CSV endpoint for offline scoring. Models are versioned `.joblib` artifacts; the whole benchmark is reproducible with four commands (`docs/reproducibility.md`).

### "Limitations?"
1. Class-capped sampling changes the prior vs. the raw corpus.
2. CIC-IDS2017 is a 2017 lab capture; real traffic drifts — a deployed model needs retraining.
3. `Destination Port` shortcut (mitigated, not removed).
4. Binary detector — the attack *type* is inferred heuristically in the API, not by a multiclass model.
5. Latency measured in Python on one laptop, not a line-rate appliance.
