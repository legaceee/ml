# Feature Selection & Dimensionality Reduction

Code: `ml/feature_engineering/selection.py`, `ml/feature_engineering/dimensionality_reduction.py`. Measured results: `docs/results.md` §4 and figure 07/08/15.

All selectors are **fitted on the train split only**; the test split is transformed with the selected column indices. Each subset is then benchmarked with the same Random Forest (and Logistic Regression) so the question "does it help?" gets a measured answer rather than a list of feature names.

## 1. Methods

### A. Pearson correlation filter (unsupervised, redundancy)
Compute |r| between every pair of training columns; walk the upper triangle and drop any column with |r| > 0.90 against an earlier column. CICFlowMeter emits many near-duplicates (`Fwd Header Length` appears twice; `Subflow Fwd Bytes` = `Total Length of Fwd Packets`; `Avg Fwd Segment Size` = `Fwd Packet Length Mean`), so this roughly halves the feature count. It ignores the label, so it can never remove a feature *because* it is predictive.

$$r_{xy} = \frac{\sum (x_i-\bar x)(y_i-\bar y)}{\sqrt{\sum (x_i-\bar x)^2 \sum (y_i-\bar y)^2}}$$

### B. Mutual information (supervised, non-linear)
$I(X;Y)=\sum_y\int p(x,y)\log\frac{p(x,y)}{p(x)p(y)}dx$ estimated with k-NN entropy estimators (`mutual_info_classif`). Captures any dependency, not just linear. Top-25 kept.

### C. Random-Forest Gini importance (supervised, embedded)
Mean decrease in Gini impurity attributed to each feature across 100 trees. Fast and aligned with the tree models we deploy, but biased toward high-cardinality continuous features. Top-25 kept.

### D. ANOVA F-test `SelectKBest(f_classif)` (supervised, univariate, linear)
Ratio of between-class to within-class variance per feature. Cheap; assumes a difference in means is what matters. Top-25 kept.

### E. Ablation: drop `Destination Port`
Not a selection method but a control. Attacks in CIC-IDS2017 target fixed services, so the port is a partial shortcut; training without it shows how much detection depends on flow *behaviour* rather than the target service.

## 2. PCA

`PCAReducer` fits a full PCA on the scaled train split, reads the cumulative explained-variance curve, keeps the smallest k with ≥ 95 % variance, and projects train/test. Σ v_k = λ_k v_k on the training covariance Σ = XᵀX/N.

PCA is **variance-preserving, not class-preserving**: the directions of largest spread in benign traffic are not necessarily the directions that separate attacks, and rotating the axes destroys the axis-aligned splits trees exploit. That is why the RF trained on principal components scores lower than on raw features in the benchmark, and why PCA is reported as an experiment and not used for the final model. Its legitimate use here is visualisation and a compression baseline.

## 3. How to read the benchmark table

For each subset the table (`feature_selection_summary.json → benchmark`) gives the number of features, RF / LR test F1, recall, ROC-AUC, training time and inference latency. The data-driven conclusion (RQ2) names the best reduced subset, the F1 change versus all features, the training-time change, and the port-ablation delta. Speed-ups are *measured* on this machine; on an 8-core CPU with `n_jobs=-1` a 100-tree forest is fast enough that a halved feature count does not always translate into a proportionally shorter wall time — the honest finding is whatever the table says.
