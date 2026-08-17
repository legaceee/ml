"""
Synthetic Development Dataset Generator for CIC-IDS2017 Benchmark Structure.

NOTE: This script generates a synthetic dataset strictly matching the 78-feature schema
and behavioral statistical distributions of the CIC-IDS2017 Intrusion Detection dataset.
It is explicitly labeled as a DEVELOPMENT FIXTURE to allow complete pipeline verification,
unit testing, and frontend development when the multi-gigabyte raw dataset is not present.
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd

CIC_IDS2017_COLUMNS = [
    "Destination Port", "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
    "Total Length of Fwd Packets", "Total Length of Bwd Packets", "Fwd Packet Length Max",
    "Fwd Packet Length Min", "Fwd Packet Length Mean", "Fwd Packet Length Std",
    "Bwd Packet Length Max", "Bwd Packet Length Min", "Bwd Packet Length Mean",
    "Bwd Packet Length Std", "Flow Bytes/s", "Flow Packets/s", "Flow IAT Mean",
    "Flow IAT Std", "Flow IAT Max", "Flow IAT Min", "Fwd IAT Total", "Fwd IAT Mean",
    "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min", "Bwd IAT Total", "Bwd IAT Mean",
    "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min", "Fwd PSH Flags", "Bwd PSH Flags",
    "Fwd URG Flags", "Bwd URG Flags", "Fwd Header Length", "Bwd Header Length",
    "Fwd Packets/s", "Bwd Packets/s", "Min Packet Length", "Max Packet Length",
    "Packet Length Mean", "Packet Length Std", "Packet Length Variance",
    "FIN Flag Count", "SYN Flag Count", "RST Flag Count", "PSH Flag Count",
    "ACK Flag Count", "URG Flag Count", "CWE Flag Count", "ECE Flag Count",
    "Down/Up Ratio", "Average Packet Size", "Avg Fwd Segment Size", "Avg Bwd Segment Size",
    "Fwd Header Length.1", "Fwd Avg Bytes/Bulk", "Fwd Avg Packets/Bulk", "Fwd Avg Bulk Rate",
    "Bwd Avg Bytes/Bulk", "Bwd Avg Packets/Bulk", "Bwd Avg Bulk Rate", "Subflow Fwd Packets",
    "Subflow Fwd Bytes", "Subflow Bwd Packets", "Subflow Bwd Bytes", "Init_Win_bytes_forward",
    "Init_Win_bytes_backward", "act_data_pkt_fwd", "min_seg_size_forward", "Active Mean",
    "Active Std", "Active Max", "Active Min", "Idle Mean", "Idle Std", "Idle Max", "Idle Min",
    "Label"
]

ATTACK_CLASSES = {
    "BENIGN": 0.55,
    "DoS Hulk": 0.15,
    "PortScan": 0.12,
    "DDoS": 0.08,
    "FTP-Patator": 0.04,
    "SSH-Patator": 0.03,
    "Web Attack \u2013 Brute Force": 0.02,
    "Bot": 0.01
}


def generate_synthetic_dataset(
    num_samples: int = 5000,
    random_seed: int = 42,
    output_path: str = None
) -> pd.DataFrame:
    """
    Generate synthetic network flow dataset matching CIC-IDS2017.
    
    Args:
        num_samples: Total number of rows to generate.
        random_seed: Random seed for reproducibility.
        output_path: If specified, saves the DataFrame to CSV.
        
    Returns:
        pd.DataFrame containing synthetic flows with realistic attributes.
    """
    np.random.seed(random_seed)
    
    labels_pool = []
    for label, weight in ATTACK_CLASSES.items():
        count = int(num_samples * weight)
        labels_pool.extend([label] * count)
    
    # Fill any rounding discrepancy
    while len(labels_pool) < num_samples:
        labels_pool.append("BENIGN")
    labels_pool = np.array(labels_pool[:num_samples])
    np.random.shuffle(labels_pool)
    
    rows = []
    for i, label in enumerate(labels_pool):
        # Base flow features depending on traffic profile
        if label == "BENIGN":
            dest_port = np.random.choice([80, 443, 8080, 53, 22, 445, 8443], p=[0.3, 0.4, 0.1, 0.1, 0.04, 0.03, 0.03])
            flow_duration = float(np.random.exponential(scale=350000) + 100)
            tot_fwd_pkts = int(np.random.poisson(lam=8) + 1)
            tot_bwd_pkts = int(np.random.poisson(lam=9) + 1)
            syn_flag = int(np.random.rand() < 0.15)
            ack_flag = int(np.random.rand() < 0.85)
            rst_flag = 0
            flow_bytes_s = float(np.random.normal(loc=12000, scale=4000))
            fwd_pkt_mean = float(np.random.normal(loc=250, scale=80))
            bwd_pkt_mean = float(np.random.normal(loc=650, scale=200))
            init_win_fwd = int(np.random.choice([8192, 29200, 65535, 14600]))
            init_win_bwd = int(np.random.choice([8192, 29200, 65535, 28960]))
            
        elif label in ["DoS Hulk", "DDoS"]:
            dest_port = np.random.choice([80, 443, 8080])
            flow_duration = float(np.random.exponential(scale=15000) + 20)
            tot_fwd_pkts = int(np.random.poisson(lam=45) + 15)
            tot_bwd_pkts = int(np.random.poisson(lam=2) + 1)
            syn_flag = int(np.random.rand() < 0.90)
            ack_flag = int(np.random.rand() < 0.10)
            rst_flag = int(np.random.rand() < 0.30)
            flow_bytes_s = float(np.random.normal(loc=185000, scale=45000))
            fwd_pkt_mean = float(np.random.normal(loc=60, scale=15))
            bwd_pkt_mean = float(np.random.normal(loc=40, scale=10))
            init_win_fwd = int(np.random.choice([256, 1024, 0]))
            init_win_bwd = 0
            
        elif label == "PortScan":
            dest_port = int(np.random.randint(1, 65535))
            flow_duration = float(np.random.uniform(10, 800))
            tot_fwd_pkts = int(np.random.choice([1, 2, 3]))
            tot_bwd_pkts = int(np.random.choice([0, 1]))
            syn_flag = 1
            ack_flag = 0
            rst_flag = int(np.random.rand() < 0.45)
            flow_bytes_s = float(np.random.normal(loc=2500, scale=800))
            fwd_pkt_mean = float(np.random.normal(loc=44, scale=5))
            bwd_pkt_mean = 0.0 if tot_bwd_pkts == 0 else 40.0
            init_win_fwd = 1024
            init_win_bwd = 0
            
        elif label in ["FTP-Patator", "SSH-Patator", "Web Attack \u2013 Brute Force"]:
            dest_port = 21 if "FTP" in label else (22 if "SSH" in label else 80)
            flow_duration = float(np.random.exponential(scale=85000) + 500)
            tot_fwd_pkts = int(np.random.poisson(lam=18) + 5)
            tot_bwd_pkts = int(np.random.poisson(lam=14) + 4)
            syn_flag = int(np.random.rand() < 0.40)
            ack_flag = 1
            rst_flag = int(np.random.rand() < 0.10)
            flow_bytes_s = float(np.random.normal(loc=45000, scale=12000))
            fwd_pkt_mean = float(np.random.normal(loc=180, scale=50))
            bwd_pkt_mean = float(np.random.normal(loc=220, scale=60))
            init_win_fwd = 29200
            init_win_bwd = 28960
            
        else:  # Bot / Infiltration
            dest_port = int(np.random.choice([6667, 8080, 4444, 1337, 53]))
            flow_duration = float(np.random.exponential(scale=600000) + 1000)
            tot_fwd_pkts = int(np.random.poisson(lam=25) + 6)
            tot_bwd_pkts = int(np.random.poisson(lam=22) + 5)
            syn_flag = int(np.random.rand() < 0.30)
            ack_flag = 1
            rst_flag = 0
            flow_bytes_s = float(np.random.normal(loc=8500, scale=2500))
            fwd_pkt_mean = float(np.random.normal(loc=110, scale=30))
            bwd_pkt_mean = float(np.random.normal(loc=140, scale=40))
            init_win_fwd = 8192
            init_win_bwd = 8192

        tot_fwd_len = float(max(0, tot_fwd_pkts * fwd_pkt_mean))
        tot_bwd_len = float(max(0, tot_bwd_pkts * bwd_pkt_mean))
        flow_pkts_s = float((tot_fwd_pkts + tot_bwd_pkts) / max(0.0001, flow_duration / 1e6))
        flow_bytes_s = max(0.0, flow_bytes_s)
        
        fwd_iat_mean = float(flow_duration / max(1, tot_fwd_pkts))
        bwd_iat_mean = float(flow_duration / max(1, tot_bwd_pkts))
        
        fwd_header_len = int(tot_fwd_pkts * 20)
        bwd_header_len = int(tot_bwd_pkts * 20)
        
        pkt_len_mean = float((tot_fwd_len + tot_bwd_len) / max(1, (tot_fwd_pkts + tot_bwd_pkts)))
        pkt_len_std = float(abs(fwd_pkt_mean - bwd_pkt_mean) / 2.0 + 10.0)
        
        row = {
            "Destination Port": int(dest_port),
            "Flow Duration": max(1.0, flow_duration),
            "Total Fwd Packets": max(1, tot_fwd_pkts),
            "Total Backward Packets": max(0, tot_bwd_pkts),
            "Total Length of Fwd Packets": tot_fwd_len,
            "Total Length of Bwd Packets": tot_bwd_len,
            "Fwd Packet Length Max": max(fwd_pkt_mean * 1.5, 40.0),
            "Fwd Packet Length Min": min(fwd_pkt_mean * 0.5, 20.0),
            "Fwd Packet Length Mean": fwd_pkt_mean,
            "Fwd Packet Length Std": max(0.0, fwd_pkt_mean * 0.25),
            "Bwd Packet Length Max": max(bwd_pkt_mean * 1.5, 0.0),
            "Bwd Packet Length Min": min(bwd_pkt_mean * 0.5, 0.0),
            "Bwd Packet Length Mean": bwd_pkt_mean,
            "Bwd Packet Length Std": max(0.0, bwd_pkt_mean * 0.3),
            "Flow Bytes/s": flow_bytes_s,
            "Flow Packets/s": flow_pkts_s,
            "Flow IAT Mean": fwd_iat_mean,
            "Flow IAT Std": fwd_iat_mean * 0.4,
            "Flow IAT Max": fwd_iat_mean * 2.0,
            "Flow IAT Min": max(0.0, fwd_iat_mean * 0.1),
            "Fwd IAT Total": flow_duration,
            "Fwd IAT Mean": fwd_iat_mean,
            "Fwd IAT Std": fwd_iat_mean * 0.35,
            "Fwd IAT Max": fwd_iat_mean * 1.8,
            "Fwd IAT Min": max(0.0, fwd_iat_mean * 0.1),
            "Bwd IAT Total": flow_duration if tot_bwd_pkts > 0 else 0.0,
            "Bwd IAT Mean": bwd_iat_mean if tot_bwd_pkts > 0 else 0.0,
            "Bwd IAT Std": bwd_iat_mean * 0.35 if tot_bwd_pkts > 0 else 0.0,
            "Bwd IAT Max": bwd_iat_mean * 1.8 if tot_bwd_pkts > 0 else 0.0,
            "Bwd IAT Min": max(0.0, bwd_iat_mean * 0.1) if tot_bwd_pkts > 0 else 0.0,
            "Fwd PSH Flags": int(np.random.rand() < 0.05),
            "Bwd PSH Flags": 0,
            "Fwd URG Flags": 0,
            "Bwd URG Flags": 0,
            "Fwd Header Length": fwd_header_len,
            "Bwd Header Length": bwd_header_len,
            "Fwd Packets/s": float(tot_fwd_pkts / max(0.0001, flow_duration / 1e6)),
            "Bwd Packets/s": float(tot_bwd_pkts / max(0.0001, flow_duration / 1e6)),
            "Min Packet Length": 0.0,
            "Max Packet Length": max(1500.0, fwd_pkt_mean * 2.0),
            "Packet Length Mean": pkt_len_mean,
            "Packet Length Std": pkt_len_std,
            "Packet Length Variance": pkt_len_std ** 2,
            "FIN Flag Count": int(np.random.rand() < 0.06),
            "SYN Flag Count": syn_flag,
            "RST Flag Count": rst_flag,
            "PSH Flag Count": int(np.random.rand() < 0.20),
            "ACK Flag Count": ack_flag,
            "URG Flag Count": int(np.random.rand() < 0.05),
            "CWE Flag Count": 0,
            "ECE Flag Count": 0,
            "Down/Up Ratio": float(tot_bwd_pkts / max(1, tot_fwd_pkts)),
            "Average Packet Size": pkt_len_mean * 1.05,
            "Avg Fwd Segment Size": fwd_pkt_mean,
            "Avg Bwd Segment Size": bwd_pkt_mean,
            "Fwd Header Length.1": fwd_header_len,
            "Fwd Avg Bytes/Bulk": 0.0,
            "Fwd Avg Packets/Bulk": 0.0,
            "Fwd Avg Bulk Rate": 0.0,
            "Bwd Avg Bytes/Bulk": 0.0,
            "Bwd Avg Packets/Bulk": 0.0,
            "Bwd Avg Bulk Rate": 0.0,
            "Subflow Fwd Packets": tot_fwd_pkts,
            "Subflow Fwd Bytes": tot_fwd_len,
            "Subflow Bwd Packets": tot_bwd_pkts,
            "Subflow Bwd Bytes": tot_bwd_len,
            "Init_Win_bytes_forward": init_win_fwd,
            "Init_Win_bytes_backward": init_win_bwd,
            "act_data_pkt_fwd": max(0, tot_fwd_pkts - 1),
            "min_seg_size_forward": 20 if np.random.rand() < 0.7 else 32,
            "Active Mean": float(np.random.exponential(scale=20000)) if flow_duration > 1e6 else 0.0,
            "Active Std": 0.0,
            "Active Max": float(np.random.exponential(scale=20000)) if flow_duration > 1e6 else 0.0,
            "Active Min": 0.0,
            "Idle Mean": float(flow_duration * 0.8) if flow_duration > 1e6 else 0.0,
            "Idle Std": 0.0,
            "Idle Max": float(flow_duration * 0.8) if flow_duration > 1e6 else 0.0,
            "Idle Min": 0.0,
            "Label": label
        }
        rows.append(row)
        
    df = pd.DataFrame(rows)
    
    # Introduce a small, realistic rate of NaN / Inf values (0.1%) to test cleaning pipeline
    nan_indices = np.random.choice(df.index, size=int(num_samples * 0.002), replace=False)
    df.loc[nan_indices, "Flow Bytes/s"] = np.nan
    inf_indices = np.random.choice(df.index, size=int(num_samples * 0.002), replace=False)
    df.loc[inf_indices, "Flow Packets/s"] = np.inf
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"[Dev Dataset] Synthetic CIC-IDS2017 dev dataset created at: {output_path} ({len(df)} rows)")
        
    return df


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    raw_path = base_dir / "raw" / "cicids2017_dev_sample.csv"
    generate_synthetic_dataset(num_samples=6000, random_seed=42, output_path=str(raw_path))
