# Capstone Viva Defense & Examination Guide

This document prepares you for final-year defense questions from academic evaluators and industry examiners.

---

### Q1: Why is raw classification accuracy misleading for cyber attack intrusion detection?
**Answer:**
Network intrusion datasets are heavily imbalanced (e.g. 80-99% legitimate traffic, 1-20% attacks). A naive model that predicts every flow as "BENIGN" would achieve 80-99% accuracy while having an Attack Recall of 0.0%, failing completely as a security defense. In cybersecurity, False Negatives (missed intrusions) are catastrophic. Therefore, we evaluate systems using **Recall (Sensitivity)**, **$F_1$-score**, **Precision-Recall AUC**, and **False Negative Rate (FNR)**.

---

### Q2: How did you ensure ZERO Data Leakage in your machine learning pipeline?
**Answer:**
1. **Partition-First Rule**: The raw dataset is split into Stratified Train (70%), Validation (15%), and Test (15%) partitions **before** any feature scaling, imputation, or transformation.
2. **Train-Only Parameter Fitting**: The `StandardScaler`, `SimpleImputer`, and `FeatureSelector` were fitted exclusively on the 70% training split. Validation and Test splits are transformed using these pre-fitted parameters without refitting.
3. **Out-of-Fold (OOF) Stacking**: Base models generating level-1 meta features for the stacking classifier were trained using 5-fold cross-validation inside the training split, preventing the meta-learner from overfitting.
4. **Untouched Test Evaluation**: The test split was evaluated once at the very end of the experiment lifecycle.

---

### Q3: Why do ensemble methods outperform single optimized models?
**Answer:**
According to Dietterich's ensemble theory, ensembles reduce error through three fundamental mechanisms:
1. **Statistical Reduction (Variance)**: When multiple models have similar training accuracy, averaging their predictions reduces the risk of choosing a poor local optimum.
2. **Computational Reduction**: Algorithms like gradient descent or greedy tree splitting may get trapped in local minima; bagging and voting restart from multiple random initializations.
3. **Representational Expansion (Bias Reduction)**: Stacking combines distinct hypothesis spaces (e.g., linear hyperplanes from Logistic Regression, orthogonal decision boundaries from Random Forest, and maximum-margin hyperplanes from SVM), allowing the ensemble to approximate complex decision boundaries that no single model family can represent.

---

### Q4: How does Optuna optimize hyperparameters compared to GridSearchCV?
**Answer:**
GridSearchCV performs exhaustive grid searches ($\mathcal{O}(k^d)$ complexity), wasting compute on unpromising parameter regions. Optuna implements **Bayesian Optimization using the Tree-structured Parzen Estimator (TPE)**:
- It models the probability distribution of hyperparameters conditioned on past evaluation scores: $P(\theta | y)$.
- It samples candidates that maximize the **Expected Improvement (EI)**, concentrating trials around optimal parameter basins and achieving superior convergence in a fraction of the computational time.

---

### Q5: How does SHAP calculate feature contributions?
**Answer:**
SHAP is grounded in cooperative game theory (Shapley values). It measures the marginal contribution of each feature across all possible feature subsets. Because of the **efficiency property**, the sum of all individual feature attributions plus the baseline expected value $\mathbb{E}[f(X)]$ strictly equals the model's exact prediction probability, ensuring mathematical consistency.
