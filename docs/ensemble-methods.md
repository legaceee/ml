# Mathematical Foundations of Ensemble Learning in Cyber Attack Detection

## 1. Voting Classifiers

### Hard Voting (Majority Rule)
For $M$ base classifiers $\{h_1, h_2, \dots, h_M\}$, the hard voting ensemble prediction $\hat{y}$ is given by:

$$\hat{y} = \text{mode}\{h_1(x), h_2(x), \dots, h_M(x)\} = \arg\max_{c \in \{0, 1\}} \sum_{m=1}^{M} \mathbb{I}(h_m(x) = c)$$

### Soft Voting (Weighted Probability Average)
When base estimators output calibrated posterior probabilities $P_m(y=c|x)$:

$$\hat{y} = \arg\max_{c \in \{0, 1\}} \frac{1}{M} \sum_{m=1}^{M} w_m P_m(y=c|x)$$

where $w_m$ represents the weight assigned to classifier $m$ ($\sum w_m = 1$). Soft voting reduces prediction variance by accounting for classifier confidence.

---

## 2. Bagging (Bootstrap Aggregation)
Bagging reduces the variance of high-variance, low-bias estimators (such as unconstrained decision trees).

Given training dataset $\mathcal{D}$ of size $N$:
1. Generate $B$ bootstrap samples $\mathcal{D}_1, \mathcal{D}_2, \dots, \mathcal{D}_B$ by sampling $N$ observations with replacement.
2. Train base estimator $h_b(x)$ on $\mathcal{D}_b$.
3. Aggregate predictions:

$$\hat{f}_{\text{bag}}(x) = \frac{1}{B} \sum_{b=1}^{B} h_b(x)$$

**Random Forest** extends Bagging by introducing feature sub-sampling: at each split in a decision tree, only a random subset $k = \sqrt{p}$ of all $p$ features is considered, decorrelating individual trees.

---

## 3. Boosting (Sequential Error Correction)

### Gradient Boosting
Gradient Boosting constructs an additive model by sequentially fitting base estimators to the negative gradient (pseudo-residuals) of the loss function:

$$F_m(x) = F_{m-1}(x) + \gamma_m h_m(x)$$

For binary log-loss:

$$\mathcal{L}(y, p) = -\left[y \ln(p) + (1-y)\ln(1-p)\right]$$

The pseudo-residual for sample $i$ at iteration $m$ is:

$$r_{im} = -\left[\frac{\partial \mathcal{L}(y_i, F(x_i))}{\partial F(x_i)}\right]_{F=F_{m-1}} = y_i - \sigma(F_{m-1}(x_i))$$

### XGBoost (Extreme Gradient Boosting)
XGBoost adds explicit second-order Taylor expansion and regularizes tree complexity:

$$\text{Obj}^{(m)} = \sum_{i=1}^{n} \left[ g_i f_m(x_i) + \frac{1}{2} h_i f_m^2(x_i) \right] + \gamma T + \frac{1}{2}\lambda \sum_{j=1}^{T} w_j^2$$

where $g_i$ and $h_i$ are first and second order gradients, $T$ is the number of terminal leaves, and $\lambda$ is the $L_2$ leaf weight penalty.

---

## 4. Stacking Classifier (Meta-Learning)

Stacking combines heterogeneous algorithms (e.g., Random Forest, XGBoost, SVM, and Logistic Regression) through a meta-learner:

```
Level 0: Base Estimators
  ├── Random Forest (Tree Bagging)        ──>  z_1 = P(Attack | RF)
  ├── XGBoost (Gradient Boosting)         ──>  z_2 = P(Attack | XGB)
  ├── Support Vector Machine (RBF Margin) ──>  z_3 = P(Attack | SVM)
  └── Logistic Regression (Linear Log-Odds)──> z_4 = P(Attack | LR)
                                                     │
Level 1: Meta-Learner                                ▼
  Meta-Features Z = [z_1, z_2, z_3, z_4] ──> Logistic Regression Meta-Model ──> Final Prediction
```

### Out-Of-Fold (OOF) Leakage Prevention
To prevent the meta-learner from overfitting to base learner predictions on the training set, Level-0 training features are generated strictly using **5-Fold Stratified Out-Of-Fold Cross-Validation**:
- The training set is split into 5 folds.
- For each fold $k$, base models are trained on the remaining 4 folds and used to predict on fold $k$.
- The resulting OOF predictions form the clean training matrix for the Level-1 meta-model.
