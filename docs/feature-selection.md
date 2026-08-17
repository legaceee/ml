# Feature Selection & Dimensionality Reduction

## 1. Feature Selection Methods Evaluated

### Method A: Collinearity Filtering (Pearson Correlation)
Calculates pairwise Pearson correlation matrix on training set features:

$$r_{xy} = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum (x_i - \bar{x})^2 \sum (y_i - \bar{y})^2}}$$

Features with $|r| > 0.90$ are pruned, reducing redundant collinear features from 61 down to 27 without degrading classification performance.

### Method B: Mutual Information Ranking
Quantifies the non-linear dependency between continuous feature $X$ and discrete label $Y$:

$$I(X; Y) = \sum_{y \in Y} \int_{X} p(x, y) \log \frac{p(x, y)}{p(x)p(y)} \, dx$$

### Method C: Tree-Based Importance (Gini Impurity Decrease)
Measures the mean decrease in node impurity across all decision trees in a Random Forest:

$$\Delta I(t) = i(t) - \frac{N_{t_L}}{N_t} i(t_L) - \frac{N_{t_R}}{N_t} i(t_R)$$

---

## 2. Principal Component Analysis (PCA)
Computes orthogonal projections along directions of maximal variance:
- Calculates covariance matrix $\Sigma = \frac{1}{N} X^T X$.
- Solves eigenvalue decomposition $\Sigma v_k = \lambda_k v_k$.
- 18 principal components capture $\ge 95\%$ of cumulative variance.
