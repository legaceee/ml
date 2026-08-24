# Model Training & Hyperparameter Tuning

Code: `ml/models/baselines.py`, `ml/models/optimization.py`, `ml/models/ensembles.py`. Numbers: `docs/results.md` §2, §3, §5.

## 1. Baseline models (defaults + balanced class weights)

| Model | Formulation | Why it is here |
|:--|:--|:--|
| **Logistic Regression** | P(y=1\|x) = σ(wᵀx + b); L2-regularised, `max_iter=2000` | The linear reference. Its low F1 on this data is evidence that the boundary is non-linear. |
| **K-Nearest Neighbours** | majority vote of the k=5 nearest rows in scaled feature space | Instance-based; no training cost, expensive inference (stores the train set). |
| **Decision Tree** | recursive Gini-impurity splits, unlimited depth | Interpretable; high variance — motivates bagging. |
| **Random Forest** | B=100 bootstrap trees, √p features per split | Bagging: averages decorrelated trees. |
| **Extra Trees** | as RF but random split thresholds | More randomness, faster, slightly higher bias. |
| **SVM (RBF)** | max-margin boundary in kernel space, K(xᵢ,xⱼ)=exp(−γ‖xᵢ−xⱼ‖²), C=1 | The kernel method of the syllabus; O(n²) training. |
| **XGBoost** | regularised second-order gradient boosting, `tree_method="hist"` | Strongest single learner on tabular data. |

`class_weight="balanced"` (LR, DT, RF, ET, SVM) reweights the loss by inverse class frequency — the cheapest form of imbalance handling — so the baseline comparison is fair across models.

## 2. Cross-validation

Every baseline is scored with `StratifiedKFold(5, shuffle=True, random_state=42)` on the train split (`ml/training/evaluate.py::cross_validate_classifier`). The CV table reports mean ± std for F1/recall/precision/accuracy, the train-fold F1 and the **overfit gap** (train − validation). A decision tree with gap ≈ +0.01 and an RF with gap ≈ +0.005 is exactly the variance reduction bagging is supposed to buy.

## 3. Hyperparameter tuning — three strategies, one protocol

All three share the same splitter, the same objective (mean 5-fold F1) and the same estimator defaults; only the search strategy differs, so the comparison isolates *how* the space is searched.

| Strategy | Implementation | Space (Random Forest) | Space (XGBoost) | Budget |
|:--|:--|:--|:--|:--|
| **Grid Search** | `GridSearchCV` | n_estimators {100,200,300} × max_depth {10,20,None} × max_features {sqrt,log2} | n_estimators {100,200,300} × max_depth {4,6,8} × learning_rate {0.05,0.1} | 18 configs (exhaustive) |
| **Random Search** | `RandomizedSearchCV` | n_estimators U[50,300], max_depth U[6,30], min_samples_split U[2,10], min_samples_leaf U[1,6], max_features {sqrt,log2,0.5} | n_estimators U[60,300], max_depth U[3,10], learning_rate U[0.01,0.25], subsample U[0.6,1], colsample U[0.5,1], min_child_weight U[1,7] | 12 configs |
| **Bayesian (Optuna TPE)** | `optuna.create_study(sampler=TPESampler(seed=42))` | same ranges as Random; log-uniform learning rate | same | 20 trials |

Grid search is exhaustive but its cost is the product of the grid sizes (kᵈ). Random search covers the same ranges with a fixed budget and, per Bergstra & Bengio (2012), usually finds better optima than a grid of equal size because it does not waste evaluations on unimportant dimensions. TPE goes further: after a few random trials it fits two densities — ℓ(θ) over the parameters of the best trials and g(θ) over the rest — and proposes the θ maximising ℓ(θ)/g(θ), which is equivalent to maximising expected improvement. The comparison table (`search_comparison` in `optimization_study.json`) records configs evaluated, wall time, best CV F1 and the test F1 of each winner, plus the best-so-far after an equal budget so the strategies can be compared fairly.

The Optuna winners are the `optimized_random_forest` / `optimized_xgboost` models used as base learners in the ensembles; the Grid winners are also saved (`grid_*.joblib`).

## 4. What tuning changes on this data

Because the strongest tree models already sit near the ceiling of this benchmark, the *test* F1 gain from tuning is small in absolute terms (see RQ3). What tuning does change is the false-negative rate and, more visibly, the cost: a grid that takes minutes versus a TPE study that reaches the same CV score in a fraction of the evaluations — or does not, which is equally worth reporting.

## 5. Ensembles

See `docs/ensemble-methods.md` for the maths. In code (`build_ensemble_models`):

* **Voting (hard/soft)** over RF (tuned) + XGB (tuned) + LR + SVM.
* **Bagging**: Extra-Trees with `bootstrap=True`, 150 trees.
* **Boosting**: AdaBoost (100 stumps-by-default, lr 0.5) and Gradient Boosting (150 trees, depth 4).
* **Stacking**: the same four base learners → 5-fold out-of-fold predictions → logistic-regression meta-learner. The SVM inside stacking/hard-voting is a plain `SVC` (stacking uses `decision_function`), avoiding the 6× cost of Platt calibration where probabilities are not needed.
* **Weighted soft vote**: blend weights chosen by searching the simplex on the **validation split** (uniform, one-hot and 60 Dirichlet draws), the only place the validation split is consumed.
