# Dataset Setup Guide: CIC-IDS2017 & Benchmark Datasets

This document provides complete instructions for downloading, placing, and preparing official network intrusion datasets for the **Cyber Attack Detection ML Capstone**.

---

## 1. Primary Dataset: CIC-IDS2017

The Canadian Institute for Cybersecurity (UNB) **CIC-IDS2017** dataset contains realistic benign and common network attacks captured over 5 days (Monday through Friday).

### Download Source:
- **Official UNB Portal**: [UNB CIC-IDS2017 Dataset](https://www.unb.ca/cic/datasets/ids-2017.html)
- **Kaggle Mirror (Pre-extracted CSVs)**: [Kaggle CIC-IDS2017 Collection](https://www.kaggle.com/datasets/cicdataset/cicids2017)

### Key Files in CIC-IDS2017:
1. `Monday-WorkingHours.pcap_ISCX.csv` (100% Benign baseline)
2. `Tuesday-WorkingHours.pcap_ISCX.csv` (FTP-Patator, SSH-Patator brute force)
3. `Wednesday-workingHours.pcap_ISCX.csv` (DoS Hulk, DoS GoldenEye, DoS slowloris, DoS Slowhttptest, Heartbleed)
4. `Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv` (Brute Force, XSS, SQL Injection)
5. `Thursday-WorkingHours-Afternoon-Infiltration.pcap_ISCX.csv` (Infiltration, PortScan)
6. `Friday-WorkingHours-Morning.pcap_ISCX.csv` (Botnet ARES)
7. `Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv` (PortScan)
8. `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv` (DDoS LOIC)

---

## 2. Directory Placement

Place downloaded CSV files in the `ml/data/raw/` folder:

```text
ml/
└── data/
    └── raw/
        ├── Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
        ├── Wednesday-workingHours.pcap_ISCX.csv
        └── ... (or combined CIC-IDS2017.csv)
```

> **Note**: If `ml/data/raw/` is empty, the pipeline will automatically generate a **strictly labeled synthetic development sample** (`cicids2017_dev_sample.csv`) that matches the exact 78-feature schema so you can develop, test, and run the complete system immediately.

---

## 3. Expected Feature Schema

The dataset contains 78+ network flow features calculated with CICFlowMeter:

| Feature Category | Examples | Description |
| :--- | :--- | :--- |
| **Flow Identifiers** | `Destination Port`, `Flow Duration` | Port and flow lifetime in microseconds |
| **Packet Counts & Lengths** | `Total Fwd Packets`, `Total Backward Packets`, `Total Length of Fwd Packets`, `Fwd Packet Length Mean`, `Bwd Packet Length Std` | Statistical packet sizes and volumetric properties |
| **Inter-Arrival Times (IAT)** | `Flow IAT Mean`, `Flow IAT Std`, `Fwd IAT Total`, `Bwd IAT Mean` | Time intervals between consecutive packets in microseconds |
| **TCP Flag Counts** | `FIN Flag Count`, `SYN Flag Count`, `RST Flag Count`, `PSH Flag Count`, `ACK Flag Count`, `URG Flag Count` | TCP handshake and connection state signaling flags |
| **Flow Rates** | `Flow Bytes/s`, `Flow Packets/s` | Throughput and traffic density rates |
| **Window & Segment Sizes** | `Init_Win_bytes_forward`, `Init_Win_bytes_backward`, `min_seg_size_forward` | TCP window sizes and segment metadata |
| **Active / Idle Times** | `Active Mean`, `Active Max`, `Idle Mean`, `Idle Min` | Flow activity vs idle period statistics |
| **Ground Truth Label** | `Label` | `BENIGN`, `DoS Hulk`, `PortScan`, `DDoS`, `FTP-Patator`, `SSH-Patator`, `Bot`, etc. |

---

## 4. Alternative Datasets Supported

The preprocessing pipeline is also compatible with:
- **CIC-IDS2018 (CSE-CIC-IDS2018)**: Updated version on AWS with similar 80-feature schema.
- **UNSW-NB15**: Network flow dataset from the Australian Cyber Security Centre (requires label column standard mapping).

---

## 5. Running the Data Pipeline & Training

Once raw CSV files are placed in `ml/data/raw/`:

```bash
# 1. Run full preprocessing and leakage-free train/val/test splitting:
python ml/preprocessing/pipeline.py

# 2. Run feature selection and hyperparameter optimization:
python ml/training/optimize_models.py

# 3. Train all baseline and ensemble models:
python ml/training/run_all_experiments.py
```
