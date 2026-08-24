# Explainable AI with SHAP

Code: `ml/explainability/shap_analysis.py` (`IDSExplainer`), served by `GET /api/explain/global/{model}` and inside every `POST /api/predict` response. Results: `ml/artifacts/metrics/shap_global.json`, figure 13, RQ6.

## 1. Why an IDS needs explanations

A SOC analyst who receives "ATTACK, p = 0.97" cannot act on it. They need to know *which* properties of the flow drove the score — a SYN without ACK and a 256-byte initial window says *scan*; 50 identical 60-byte packets per millisecond says *flood*. Explanations also expose shortcuts: if the model keyed only on `Destination Port`, it would be brittle. Here SHAP does both jobs.

## 2. Shapley values

For a feature set F and a prediction function f, the Shapley value of feature i for instance x is its average marginal contribution over all subsets S that exclude it:

$$\phi_i(x)=\sum_{S\subseteq F\setminus\{i\}}\frac{|S|!\,(|F|-|S|-1)!}{|F|!}\Big[f_x(S\cup\{i\})-f_x(S)\Big]$$

**Efficiency**: $f(x)=\phi_0+\sum_i\phi_i(x)$ where $\phi_0=\mathbb E[f(X)]$ is the base value — the attributions sum exactly to the prediction, so they can be read as "this feature pushed the log-odds up by 1.3".

`TreeExplainer` (Lundberg et al., 2020) computes these exactly for tree ensembles in polynomial time, which is why the global analysis is run on the tuned XGBoost even when a heterogeneous stacking ensemble has the top F1 (its SVM member would require slow kernel-SHAP approximations).

## 3. What is reported

* **Global importance** — mean |φᵢ| over 500 test flows; top-25 in `shap_global.json` and figure 13.
* **Local drivers** — for each live prediction the API returns the top positive (toward ATTACK) and negative (toward BENIGN) contributors with the raw feature values, so the dashboard can show "why".

## 4. Reading the global ranking honestly

`Destination Port` typically ranks first or second. That is expected — attacks in CIC-IDS2017 hit fixed services — and it is a known shortcut risk. Two things in the project address it: the ranking is reported as is, and the feature-selection benchmark includes an `all_minus_destination_port` ablation whose F1 delta is quoted in RQ2/RQ6. The remaining top features (`Init_Win_bytes_forward/backward`, `min_seg_size_forward`, packet-length and IAT statistics) are behavioural: attack tools use characteristic TCP window defaults and send packets of uniform size at uniform intervals, which no legitimate browser session does.

## 5. Limits

SHAP explains the *model*, not the *world*: a high attribution says the model used the feature, not that the feature causes an attack. Correlated features (the many duplicated CICFlowMeter columns) share credit, so individual rankings among near-duplicates are unstable even when the group ranking is stable.
