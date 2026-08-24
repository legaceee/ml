# Dataset Setup: CIC-IDS2017

This project trains on the **real** CIC-IDS2017 intrusion-detection benchmark from the Canadian Institute for Cybersecurity (University of New Brunswick).

## 1. One command

```bash
python -m ml.data.build_dataset --download --build
```

That downloads the 8 official *MachineLearningCSV* files (~885 MB) from a public mirror and builds the training sample. Nothing else is needed.

| Day file | Rows | Traffic |
|:--|--:|:--|
| Monday-WorkingHours | 529,918 | 100% benign |
| Tuesday-WorkingHours | 445,909 | FTP-Patator, SSH-Patator brute force |
| Wednesday-workingHours | 692,703 | DoS Hulk, GoldenEye, slowloris, Slowhttptest, Heartbleed |
| Thursday-Morning-WebAttacks | 170,366 | Web brute force, XSS, SQL injection |
| Thursday-Afternoon-Infilteration | 288,602 | Infiltration |
| Friday-Morning | 191,033 | Bot (ARES) |
| Friday-Afternoon-PortScan | 286,467 | PortScan |
| Friday-Afternoon-DDos | 225,745 | DDoS (LOIC) |
| **Total** | **2,830,743** | 15 classes, 78 features + label |

Sources: [UNB official page](https://www.unb.ca/cic/datasets/ids-2017.html) (requires a form), [Kaggle mirror](https://www.kaggle.com/datasets/cicdataset/cicids2017) (requires login), [Hugging Face mirror `c01dsnap/CIC-IDS2017`](https://huggingface.co/datasets/c01dsnap/CIC-IDS2017) (direct download — used by the script).

## 2. Why a sample, and how it is drawn

Training 18 models — including an RBF-SVM, KNN and a 5-fold stacking ensemble — on 2.8M rows is not feasible on a laptop (an RBF-SVM alone would take hours per fit). The builder therefore draws a **class-capped stratified sample**:

* `BENIGN` is randomly down-sampled to `--benign-rows` (default 40,000).
* Every attack class is randomly down-sampled to `--attack-cap` (default 1,500) **or kept in full if smaller** — so Heartbleed (11 rows), SQL Injection (21) and Infiltration (36) are never lost.
* Seed 42; the manifest `ml/data/raw/cicids2017_sample_manifest.json` records both the full-corpus and the sample class counts.

Result with defaults: 55,720 rows, 28% attack, all 15 classes. The overall imbalance (≈3:1) and the extreme within-attack imbalance (40,000 benign vs 11 Heartbleed) keep the *imbalanced data* question meaningful.

This is a standard practice in CIC-IDS2017 papers, but it does change the class prior relative to the original 80/20 corpus — the docs say so explicitly rather than hiding it.

## 3. Known quirks of the raw files (all handled in code)

* Column names have leading spaces → stripped.
* `Flow Bytes/s` and `Flow Packets/s` contain `Infinity` and `NaN` (zero-duration flows) → coerced to NaN and median-imputed (train-fitted).
* The three *Web Attack* labels contain a mis-encoded en-dash (`Web Attack � Brute Force`) → normalised to `Web Attack - Brute Force`.
* `Fwd Header Length` appears twice (`Fwd Header Length.1`) → kept, the correlation filter removes the duplicate.
* 8 columns are constant (all zero: bulk-rate features, `Bwd PSH Flags`, `Bwd URG Flags`, …) → dropped.
* Exact duplicate rows exist → removed **before** splitting so the same flow can't appear in both train and test.

## 4. Files that stay out of git

`ml/data/raw/cicids2017_full/`, `ml/data/raw/cicids2017_sample.csv` and `ml/data/splits/` are git-ignored (hundreds of MB). Re-create them with the one command above.

## 5. Without the real data

If `ml/data/raw/` is empty the loader generates a small **synthetic** fixture so the code can be smoke-tested. Every artifact produced from it is tagged `is_synthetic: true`, and the dashboard shows a warning banner. Synthetic results are not research results.
