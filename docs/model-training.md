# Model Training & Hyperparameter Optimization

## 1. Baseline Model Formulation

### Logistic Regression
Models the log-odds of a malicious event as a linear combination of flow features:

$$P(y=1|x) = \sigma(w^T x + b) = \frac{1}{1 + e^{-(w^T x + b)}}$$

### Decision Tree
Recursively partitions feature space using Gini impurity criteria:

$$\text{Gini}(p) = 1 - \sum_{k=1}^{K} p_k^2$$

### Random Forest
Ensemble of $B=100$ decorrelated decision trees with feature sub-sampling $m = \sqrt{p}$.

### Support Vector Machine (SVM)
Finds the maximum-margin hyperplane separating classes in reproducing kernel Hilbert space using the Radial Basis Function (RBF) kernel:

$$K(x_i, x_j) = \exp\left(-\gamma \|x_i - x_j\|^2\right)$$

### XGBoost
Optimizes regularized objective function using exact greedy split finding and weighted quantile sketch.

---

## 2. Bayesian Hyperparameter Optimization with Optuna
Optuna models the objective distribution $P(\theta|y)$ using the **Tree-structured Parzen Estimator (TPE)**, optimizing:

$$\text{EI}_{y^*}(\theta) = \int_{-\infty}^{y^*} (y^* - y) p(y|\theta) \, dy = \frac{\gamma y^* \ell(\theta) - \ell(\theta) \int_{-\infty}^{y^*} P(y) \, dy}{\gamma \ell(\theta) + (1-\gamma) g(\theta)}$$

5-fold stratified cross-validation on the training set was used to maximize the $F_1$-score.
