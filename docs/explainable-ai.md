# Explainable AI (XAI) using SHAP (SHapley Additive exPlanations)

## 1. The Need for Interpretability in Cybersecurity
Machine learning models are frequently criticized as "black-box" systems. In a Security Operations Center (SOC), security analysts cannot blindly trust a high-confidence alert without actionable context. 

SHAP provides mathematically rigorous, game-theoretic feature attribution answering:
1. Which specific protocol attributes contributed most to classifying a packet flow as an intrusion?
2. Did the feature increase or decrease the malicious probability score?

---

## 2. Shapley Values Formulation
Originating from cooperative game theory (Lloyd Shapley, 1953), the Shapley value $\phi_i$ of feature $i$ represents its marginal contribution averaged over all possible feature subsets $S \subseteq F \setminus \{i\}$:

$$\phi_i(x) = \sum_{S \subseteq F \setminus \{i\}} \frac{|S|! (|F| - |S| - 1)!}{|F|!} \left[ f_x(S \cup \{i\}) - f_x(S) \right]$$

where:
- $F$ is the total set of input features.
- $S$ is a subset of features excluding feature $i$.
- $f_x(S)$ is the model prediction conditioned only on feature subset $S$.

### Additive Feature Attribution Property (Efficiency)
The sum of all feature attributions equals the difference between the model output $f(x)$ and the baseline expected value $\mathbb{E}[f(X)]$:

$$f(x) = \phi_0 + \sum_{i=1}^{M} \phi_i(x)$$

where $\phi_0 = \mathbb{E}[f(X)]$ is the base value (average dataset prediction).

---

## 3. Implementation in CyberGuard IDS
- **TreeExplainer**: Utilized for Tree-based models (Random Forest, XGBoost, ExtraTrees) with polynomial runtime $\mathcal{O}(TLD^2)$, enabling sub-millisecond local explanations.
- **Global Feature Summary**: Ranks overall protocol importance across test sets.
- **Local Waterfall Attributions**: Returns positive attack drivers (e.g. abnormally high Flow Bytes/s, SYN Flag set) and negative benign drivers for individual prediction events.
