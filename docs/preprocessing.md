# Preprocessing Pipeline & Leakage Prevention

Implemented in `ml/preprocessing/pipeline.py` (`PipelineOrchestrator`), `cleaning.py`, `encoding.py`, `scaling.py`.

## 1. Order of operations — why it matters

```
raw CSV (55,720 rows × 79 cols)
   │
   ▼  DataCleaner
   ├─ strip header whitespace, collapse double spaces
   ├─ "Infinity"/"inf"/"NaN" strings and ±inf  →  NaN
   ├─ drop exact duplicate rows            (1,853 removed)   ← BEFORE the split
   └─ drop constant (zero-variance) columns (8 removed)      → 70 features
   │
   ▼  LabelEncoderIDS
   └─ 15 string labels → 0 (BENIGN) / 1 (ATTACK); original label kept for per-class analysis
   │
   ▼  Stratified split (on the ORIGINAL category, seed 42)
   ├─ train 70 %  (37,706)
   ├─ val   15 %  ( 8,080)   ← only consumed by the weighted-ensemble weight search
   └─ test  15 %  ( 8,081)   ← touched once per final model
   │
   ▼  NetworkFlowPreprocessor  — fit(train) only
   ├─ SimpleImputer(median)   64 Inf/NaN cells
   └─ StandardScaler          z = (x − μ_train) / σ_train
   │
   ▼  transform(train), transform(val), transform(test) with the frozen parameters
```

### The three leakage traps this order avoids

| Trap | What goes wrong | How it is avoided here |
|:--|:--|:--|
| Scaling before splitting | μ and σ include test rows — the model has "seen" test statistics | scaler fitted on train only (`pipeline.py` step 5) |
| Duplicates across splits | identical flows in train and test → memorisation looks like generalisation | duplicates dropped before the split (`cleaning.py` step 4) |
| Resampling / feature selection on the full data | SMOTE creates synthetic points from test neighbours; MI/importance scores use test labels | `resampling.py` and `selection.py` are only ever called with train arrays |

The stacking ensemble adds a fourth guard: its meta-learner is trained on **out-of-fold** predictions (`StackingClassifier(cv=5)`), so no base learner ever predicts a row it was trained on.

## 2. Missing value handling

The only missing values in CIC-IDS2017 are in `Flow Bytes/s` and `Flow Packets/s`, produced when CICFlowMeter divides by a zero flow duration. They arrive as the strings `Infinity` / `NaN`.

* Coerced to `NaN` (`DataCleaner`).
* Imputed with the **median of the training split** (`SimpleImputer(strategy="median")`). Median rather than mean because both columns are extremely heavy-tailed (a single DoS flow can be 10⁸ bytes/s); the mean would be dominated by outliers.
* Dropping the rows would bias the sample: zero-duration flows are disproportionately PortScan probes.

## 3. Encoding

All 78 input features are numeric counts, byte totals, timings and flag counts — CICFlowMeter output — so no categorical encoding is needed for the inputs. The **target** is label-encoded:

* Binary mode (used for all experiments): `BENIGN → 0`, anything else → `1`.
* Multiclass map (`LabelEncoderIDS.STANDARD_MULTICLASS_MAP`) groups the 14 attack strings into 9 families and is used only for the per-attack-type recall table.

`Destination Port` is technically categorical but is treated as numeric by every published CIC-IDS2017 baseline; the `all_minus_destination_port` ablation in the feature benchmark shows what happens without it.

## 4. Scaling

`StandardScaler` (z-score). It is essential for the distance- and margin-based models (KNN, SVM, logistic regression converge badly on raw byte counts spanning 0–10⁹) and neutral for tree models, which split on rank order. Applying one scaler to every model keeps the comparison fair.

## 5. Audit trail

`ml/artifacts/metrics/dataset_summary.json` records the initial/final shape, duplicates removed, Inf/NaN count, dropped constant columns, the class counts in each split and the seed — so every preprocessing claim in the report can be checked.
