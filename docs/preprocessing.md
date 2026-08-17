# Preprocessing Pipeline & Leakage Prevention Protocol

## 1. Zero Data Leakage Rule

```
[Raw Dataset]
      │
      ▼
[Stratified Train / Val / Test Split (70 / 15 / 15)]
      │
      ├──> TRAIN SPLIT (70%) ──> Fit SimpleImputer & StandardScaler
      │                               │
      ├──> VAL SPLIT (15%)   ─────────┼──> Transform using Train parameters
      │                               │
      └──> TEST SPLIT (15%)  ─────────┘──> Transform using Train parameters
```

### Preprocessing Operations:
1. **Header Stripping**: Cleans whitespace and non-standard characters from CSV headers.
2. **Infinite and Missing Value Imputation**: Replaces `np.inf`, `-np.inf`, and string `"Infinity"` with `np.nan`, then imputes using the median computed exclusively on training data.
3. **Zero-Variance Column Removal**: Detects columns where variance is 0 across the training set (e.g. constant protocol fields) and removes them consistently across all splits.
4. **Z-Score Normalization**: Standardizes features to mean $\mu = 0$ and standard deviation $\sigma = 1$:

$$z = \frac{x - \mu_{\text{train}}}{\sigma_{\text{train}}}$$
