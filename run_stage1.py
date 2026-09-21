#!/usr/bin/env python3
"""
run_stage1.py - Automated Stage I Training and Evaluation Pipeline
Replicating the exact methodology and implementation of:
Hara & Sasabe, "Practicality of in-kernel/user-space packet processing empowered by
lightweight neural network and decision tree", Computer Networks 240 (2024) 110188.

Architectures & Methodology strictly mirroring:
- fixed-nn/src/train_mlp.py (MLP training with Adam lr=1e-3, batch_size=128, CrossEntropyLoss)
- fixed-nn/src/quantize.py (Post-Training Quantization via pytorch_quantization)
- fixed-nn/src/create_mlp_c_params.py (C parameter export)
- fixed-nn/src/test_mlp_c.py (C int8 inference & latency measurement)
- fixed-dt/learn_dt_float.py & fixed-dt/learn_dt.py (Decision Tree training & C inference)
"""

import os
import sys
import time
import json
import math
import pickle
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import random
import shutil
import copy
import argparse
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np

# PyTorch
try:
    import torch
    import torch.nn as nn
    from torch.optim import Adam
    from torch.utils.data import DataLoader, Dataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Scikit-learn
try:
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Ctypes
import ctypes
from ctypes import CDLL, POINTER, c_longlong, c_uint, c_int, c_int8

# Optional: NVIDIA PyTorch Quantization Toolkit
try:
    from pytorch_quantization import calib
    from pytorch_quantization import nn as quant_nn
    from pytorch_quantization import quant_modules
    from pytorch_quantization.tensor_quant import QuantDescriptor
    PYTORCH_QUANT_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False


# ==============================================================================
# 1. CONSTANTS & CATEGORY MAPPINGS (From Paper Table 5 & Table 7)
# ==============================================================================

MULTIPLIER_FXP = 2 ** 16  # Fixed-point 16-bit fractional multiplier

LABEL_NAMES_7 = [
    "Botnet",       # 0
    "Brute force",  # 1
    "DoS",          # 2
    "Infiltration", # 3
    "Normal",       # 4
    "Portscan",     # 5
    "Web attack",   # 6
]

RAW_TO_7_MAPPING = {
    0: 0,   # Botnet:ARES -> Botnet
    1: 1,   # Brute Force:FTP-Patator -> Brute force
    2: 1,   # Brute Force:SSH-Patator -> Brute force
    3: 2,   # DDoS:LOIT -> DoS
    4: 2,   # DoS / DDoS:DoS GoldenEye -> DoS
    5: 2,   # DoS / DDoS:DoS Hulk -> DoS
    6: 2,   # DoS / DDoS:DoS Slowhttptest -> DoS
    7: 2,   # DoS / DDoS:DoS slowloris -> DoS
    8: 2,   # DoS / DDoS:Heartbleed -> DoS
    9: 3,   # Infiltration:Dropbox download -> Infiltration
    10: 4,  # Normal -> Normal
    11: 5,  # PortScan:Firewall off -> Portscan
    12: 5,  # PortScan:Firewall on -> Portscan
    13: 6,  # Web Attack:Sql Injection -> Web attack
    14: 6,  # Web Attack:XSS -> Web attack
}


# ==============================================================================
# 2. MODEL ARCHITECTURE (fixed-nn/src/neural_nets.py)
# ==============================================================================

if TORCH_AVAILABLE:
    class MLP(nn.Module):
        """
        Three-layer MLP as defined in Section 5.1 of Hara & Sasabe (2024):
        Input 12 -> Hidden 16 -> ReLU -> Hidden 16 -> ReLU -> Output 2 or 7.
        All linear layers use bias=False to strictly match C kernel implementation.
        """
        def __init__(self, in_dim=12, hidden_sizes=[16, 16], out_dim=2):
            super().__init__()
            layer_list = [nn.Linear(in_dim, hidden_sizes[0], bias=False)]
            for i in range(1, len(hidden_sizes)):
                layer_list.extend([
                    nn.ReLU(inplace=True),
                    nn.Linear(hidden_sizes[i-1], hidden_sizes[i], bias=False)
                ])
            layer_list.extend([
                nn.ReLU(inplace=True),
                nn.Linear(hidden_sizes[-1], out_dim, bias=False)
            ])
            self.net = nn.Sequential(*layer_list)

        def forward(self, x):
            return self.net(x.flatten(start_dim=1))

    if PYTORCH_QUANT_AVAILABLE:
        class QuantMLP(nn.Module):
            """
            Quantized MLP using NVIDIA pytorch_quantization for PTQ int8 calibration.
            """
            def __init__(self, in_dim=12, hidden_sizes=[16, 16], out_dim=2):
                super().__init__()
                layer_list = [quant_nn.QuantLinear(in_dim, hidden_sizes[0], bias=False)]
                for i in range(1, len(hidden_sizes)):
                    layer_list.extend([
                        nn.ReLU(inplace=True),
                        quant_nn.QuantLinear(hidden_sizes[i-1], hidden_sizes[i], bias=False)
                    ])
                layer_list.extend([
                    nn.ReLU(inplace=True),
                    quant_nn.QuantLinear(hidden_sizes[-1], out_dim, bias=False)
                ])
                self.net = nn.Sequential(*layer_list)

            def forward(self, x):
                return self.net(x.flatten(start_dim=1))

    class FlowDataset(Dataset):
        def __init__(self, data: np.ndarray, labels: np.ndarray):
            self.data = torch.FloatTensor(data)
            self.labels = torch.LongTensor(labels)

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return self.data[idx], self.labels[idx]


# ==============================================================================
# 3. FEATURE EXTRACTION
# ==============================================================================

def calculate_flow_features(packets: np.ndarray, feature_mode: str = "mad") -> np.ndarray:
    """
    Computes 12-dimensional flow features from packet array.
    Packet layout: [sport, dport, protocol, tot_len, interval, direction]

    Features x1..x12:
    x1..x3: sport, dport, proto
    x4: len
    x5: interval
    x6: direction
    x7..x9: running average of len, interval, direction
    x10..x12:
      - 'mad': Mean Absolute Deviation |x - x̄| (Hara & Sasabe Baseline)
      - 'scv': Sample Squared Coefficient of Variation s² / x̄² with Bessel's N-1 (Thesis Proposed)
    """
    n_packets = packets.shape[0]

    if feature_mode == "mad":
        # Exactly matching train_mlp.py lines 145-160 of original repo:
        average = np.zeros(3, dtype=np.float64)
        deviation = np.zeros(3, dtype=np.float64)
        for i in range(n_packets):
            current_vector = packets[i, :6].astype(np.float64)
            average += current_vector[3:]
            current_average = average / (i + 1)
            deviation += np.abs(current_vector[3:] - current_average)
            current_deviation = deviation / (i + 1)
        return np.concatenate((current_vector, current_average, current_deviation))

    elif feature_mode == "scv":
        # Welford online sample variance with Bessel's (N-1) correction
        welford_avg = np.zeros(3, dtype=np.float64)
        M2 = np.zeros(3, dtype=np.float64)
        for i in range(n_packets):
            current_vector = packets[i, :6].astype(np.float64)
            x_stat = current_vector[3:]
            count = i + 1
            delta = x_stat - welford_avg
            welford_avg += delta / count
            delta2 = x_stat - welford_avg
            M2 += delta * delta2

        if n_packets < 2:
            sample_variance = np.zeros(3, dtype=np.float64)
        else:
            sample_variance = np.maximum(0.0, M2 / (n_packets - 1))

        avg_sq = np.square(welford_avg)
        scv = np.divide(sample_variance, avg_sq, out=np.zeros_like(sample_variance), where=(avg_sq != 0))
        return np.concatenate((current_vector, welford_avg, scv))

    else:
        raise ValueError(f"Unsupported feature mode: {feature_mode}")


def extract_all_features(raw_flows: list, feature_mode: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_flows = len(raw_flows)
    X_mat = np.zeros((n_flows, 12), dtype=np.float64)
    y_bin = np.zeros(n_flows, dtype=np.int64)
    y_cat = np.zeros(n_flows, dtype=np.int64)

    t0 = time.perf_counter()
    report_step = max(100000, n_flows // 10)
    for idx, (pkts, b_lbl, c_lbl) in enumerate(raw_flows):
        X_mat[idx] = calculate_flow_features(pkts, feature_mode=feature_mode)
        y_bin[idx] = int(b_lbl[0])
        y_cat[idx] = int(c_lbl[0])
        if (idx + 1) % report_step == 0 or idx == n_flows - 1:
            pct = (idx + 1) / n_flows * 100
            rate = (idx + 1) / max(0.1, time.perf_counter() - t0)
            print(f"    Extracted {idx+1:,} / {n_flows:,} flows ({pct:.1f}%) [{rate:,.0f} flows/s]")

    X_mat = np.nan_to_num(X_mat, nan=0.0, posinf=0.0, neginf=0.0)
    return X_mat, y_bin, y_cat


# ==============================================================================
# 4. DATA LOADING & 3-FOLD SPLITTING (fixed-nn/src/train_mlp.py)
# ==============================================================================

def generate_mock_dataset(n_samples: int = 3000, max_len: int = 10):
    raw_data = []
    rng = np.random.RandomState(42)
    for _ in range(n_samples):
        flow_len = rng.randint(2, max_len + 1)
        sport = rng.randint(1024, 65535)
        dport = rng.choice([80, 443, 22, 21, 53, 8080])
        proto = rng.choice([6, 17])
        packets = []
        for _ in range(flow_len):
            packets.append([sport, dport, proto, rng.randint(40, 1500), rng.exponential(scale=50.0), rng.choice([0, 1])])
        packets = np.array(packets, dtype=np.float64)
        is_attack = rng.rand() > 0.70
        bin_label = np.array([1 if is_attack else 0])
        cat_label = np.array([rng.choice([3, 11, 0, 13]) if is_attack else 10])
        raw_data.append((packets, bin_label, cat_label))
    return raw_data


def load_raw_flows(data_dir: Path, max_length: int = 100, seed: int = 0):
    pickle_path = data_dir / "flows.pickle"
    if not pickle_path.exists():
        print(f"[!] Error: flows.pickle not found at: {pickle_path}")
        sys.exit(1)

    print(f"[*] Loading raw flows from {pickle_path} ...")
    t0 = time.perf_counter()
    with open(pickle_path, "rb") as f:
        all_data = pickle.load(f)
    print(f"[+] Loaded in {time.perf_counter() - t0:.1f}s.")

    # Filter corrupt flows matching train_mlp.py
    filtered = []
    for item in all_data:
        if np.all(item[:, 4] >= 0):
            pkts = item[:max_length, :-2]
            bin_lbl = item[:max_length, -1:]
            cat_lbl = item[:max_length, -2:-1]
            filtered.append((pkts, bin_lbl[0], cat_lbl[0]))

    random.seed(seed)
    random.shuffle(filtered)
    print(f"[+] Filtered & shuffled {len(filtered):,} valid flows.")
    return filtered


def load_dataset(data_dir: Path, output_dir: Path, feature_mode: str,
                 max_length: int = 100, use_mock: bool = False,
                 seed: int = 0, no_cache: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads features from disk cache (.npz) or extracts them once from flows.pickle.
    """
    if use_mock:
        raw_flows = generate_mock_dataset(3000, min(max_length, 20))
        return extract_all_features(raw_flows, feature_mode)

    cache_file = data_dir / f"cache_flows_len{max_length}_{feature_mode}.npz"
    alt_cache = output_dir / f"cache_flows_len{max_length}_{feature_mode}.npz"

    # Check cache
    if not no_cache:
        for c_path in [cache_file, alt_cache]:
            if c_path.exists():
                print(f"[+] Found cached features: {c_path}")
                t0 = time.perf_counter()
                data = np.load(c_path)
                print(f"[+] Loaded {len(data['X_raw']):,} flows in {time.perf_counter() - t0:.2f}s! [Cache Hit]")
                return data["X_raw"], data["y_bin"], data["y_cat"]

    # Extract
    raw_flows = load_raw_flows(data_dir, max_length=max_length, seed=seed)
    print(f"[*] Extracting 12 features ({feature_mode.upper()})...")
    X_raw, y_bin, y_cat = extract_all_features(raw_flows, feature_mode)

    # Save cache
    save_path = cache_file if os.access(data_dir, os.W_OK) else alt_cache
    try:
        print(f"[*] Saving cache to: {save_path} ...")
        np.savez_compressed(save_path, X_raw=X_raw, y_bin=y_bin, y_cat=y_cat)
        print(f"[+] Cache saved ({save_path.stat().st_size / (1024*1024):.1f} MB). Future runs start in <1s!")
    except Exception as e:
        print(f"[!] Warning: Could not write cache: {e}")

    return X_raw, y_bin, y_cat


def split_3fold(X_raw: np.ndarray, y: np.ndarray, fold: int = 0, n_fold: int = 3,
                train_val_split: float = 0.8, seed: int = 0):
    """
    Replicates get_nth_split and train_val_split from train_mlp.py:
    Test: 1/3 of dataset (fold 0).
    Train/Val: Remaining 2/3 split into 80% train and 20% validation.
    """
    total = len(X_raw)
    bottom = int(math.floor(float(total) * fold / n_fold))
    top = int(math.floor(float(total) * (fold + 1) / n_fold))

    test_indices = list(range(bottom, top))
    remaining_indices = list(range(0, bottom)) + list(range(top, total))

    random.seed(seed)
    random.shuffle(remaining_indices)

    n_train = int(round(len(remaining_indices) * train_val_split))
    train_indices = remaining_indices[:n_train]
    val_indices = remaining_indices[n_train:]

    train_idx = np.array(train_indices, dtype=np.int64)
    val_idx = np.array(val_indices, dtype=np.int64)
    test_idx = np.array(test_indices, dtype=np.int64)

    print(f"[+] Split: Train={len(train_idx):,} ({len(train_idx)/total*100:.1f}%), "
          f"Val={len(val_idx):,} ({len(val_idx)/total*100:.1f}%), "
          f"Test={len(test_idx):,} ({len(test_idx)/total*100:.1f}%)")

    return (X_raw[train_idx], y[train_idx]), (X_raw[val_idx], y[val_idx]), (X_raw[test_idx], y[test_idx])


# ==============================================================================
# 5. DECISION TREE (fixed-dt/learn_dt_float.py & fixed-dt/learn_dt.py)
# ==============================================================================

def compile_dt_c(source_c_path: Path, output_so_path: Path) -> CDLL:
    cmd = ["gcc", "-Wall", "-O3", "-fPIC", "-shared", str(source_c_path), "-o", str(output_so_path)]
    subprocess.run(cmd, check=True)
    return CDLL(str(output_so_path))


def evaluate_dt_c(cdll: CDLL, X_test: np.ndarray, dt_model: Any) -> Tuple[np.ndarray, np.ndarray]:
    children_left = np.ascontiguousarray(dt_model.tree_.children_left, dtype=np.int64)
    children_right = np.ascontiguousarray(dt_model.tree_.children_right, dtype=np.int64)
    feature = np.ascontiguousarray(dt_model.tree_.feature, dtype=np.int64)
    threshold = np.ascontiguousarray(np.round(dt_model.tree_.threshold * MULTIPLIER_FXP), dtype=np.int64)
    value = np.ascontiguousarray(dt_model.tree_.value.squeeze().argmax(axis=1 if dt_model.tree_.value.ndim > 2 else 0), dtype=np.int64)

    c_longlong_p = POINTER(c_longlong)
    c_uint_p = POINTER(c_uint)
    dt_func = cdll.dt
    dt_func.argtypes = [c_longlong_p, c_longlong_p, c_longlong_p, c_longlong_p, c_longlong_p, c_longlong_p, c_uint_p]
    dt_func.restype = None

    left_p = children_left.ctypes.data_as(c_longlong_p)
    right_p = children_right.ctypes.data_as(c_longlong_p)
    feat_p = feature.ctypes.data_as(c_longlong_p)
    thresh_p = threshold.ctypes.data_as(c_longlong_p)
    val_p = value.ctypes.data_as(c_longlong_p)

    X_fxp = np.ascontiguousarray(np.round(X_test * MULTIPLIER_FXP), dtype=np.int64)
    n_samples = len(X_fxp)
    n_bench = min(10000, n_samples)

    preds = np.zeros(n_samples, dtype=np.int64)
    latencies = []
    class_buf = np.zeros(1, dtype=np.uintc)
    class_p = class_buf.ctypes.data_as(c_uint_p)

    for i in range(n_bench):
        sample_p = X_fxp[i].ctypes.data_as(c_longlong_p)
        start = time.perf_counter()
        dt_func(sample_p, left_p, right_p, val_p, feat_p, thresh_p, class_p)
        end = time.perf_counter()
        latencies.append((end - start) * 1e6)
        preds[i] = int(class_buf[0])

    for i in range(n_bench, n_samples):
        sample_p = X_fxp[i].ctypes.data_as(c_longlong_p)
        dt_func(sample_p, left_p, right_p, val_p, feat_p, thresh_p, class_p)
        preds[i] = int(class_buf[0])

    return preds, np.array(latencies)


def train_and_eval_dt(X_raw_train: np.ndarray, y_train: np.ndarray,
                      X_raw_test: np.ndarray, y_test: np.ndarray,
                      is_binary: bool, max_depth: int, dt_source_path: Path,
                      work_dir: Path, seed: int = 0) -> Dict[str, Any]:
    print(f"\n{'='*30} DECISION TREE (Depth={max_depth}, Binary={is_binary}) {'='*30}")
    t0 = time.perf_counter()
    dt = DecisionTreeClassifier(max_depth=max_depth, random_state=seed)
    dt.fit(X_raw_train, y_train)
    print(f"[+] Fitted Decision Tree in {time.perf_counter() - t0:.2f}s.")

    # 1. Evaluate DT (float) on raw unscaled features
    preds_py = dt.predict(X_raw_test)
    n_bench = min(10000, len(X_raw_test))
    latencies_py = []
    for i in range(n_bench):
        sample = X_raw_test[i:i+1]
        start = time.perf_counter()
        _ = dt.predict(sample)
        end = time.perf_counter()
        latencies_py.append((end - start) * 1e6)

    acc_py = accuracy_score(y_test, preds_py)
    prec_py = precision_score(y_test, preds_py, average="weighted" if not is_binary else "binary", zero_division=0)
    rec_py = recall_score(y_test, preds_py, average="weighted" if not is_binary else "binary", zero_division=0)
    f1_py = f1_score(y_test, preds_py, average="weighted" if not is_binary else "binary", zero_division=0)
    rep_py = classification_report(y_test, preds_py, output_dict=True, zero_division=0)

    # 2. Evaluate DT (fixed) in C
    so_path = work_dir / f"dt_{'binary' if is_binary else 'multi'}.so"
    cdll = compile_dt_c(dt_source_path, so_path)
    preds_c, latencies_c = evaluate_dt_c(cdll, X_raw_test, dt)

    acc_c = accuracy_score(y_test, preds_c)
    prec_c = precision_score(y_test, preds_c, average="weighted" if not is_binary else "binary", zero_division=0)
    rec_c = recall_score(y_test, preds_c, average="weighted" if not is_binary else "binary", zero_division=0)
    f1_c = f1_score(y_test, preds_c, average="weighted" if not is_binary else "binary", zero_division=0)
    rep_c = classification_report(y_test, preds_c, output_dict=True, zero_division=0)

    print(f"[*] DT (float)  Accuracy: {acc_py:.4f} | Prec: {prec_py:.4f} | Rec: {rec_py:.4f} | F1: {f1_py:.4f} | Latency: {np.mean(latencies_py):.2f} μs")
    print(f"[*] DT (fixed)  Accuracy: {acc_c:.4f} | Prec: {prec_c:.4f} | Rec: {rec_c:.4f} | F1: {f1_c:.4f} | Latency: {np.mean(latencies_c):.2f} μs")

    return {
        "dt_float": {
            "accuracy": acc_py, "precision": prec_py, "recall": rec_py, "f1": f1_py,
            "latency_mean_us": float(np.mean(latencies_py)), "latency_std_us": float(np.std(latencies_py)),
            "report": rep_py
        },
        "dt_fixed": {
            "accuracy": acc_c, "precision": prec_c, "recall": rec_c, "f1": f1_c,
            "latency_mean_us": float(np.mean(latencies_c)), "latency_std_us": float(np.std(latencies_c)),
            "report": rep_c
        }
    }


# ==============================================================================
# 6. NEURAL NETWORK TRAINING (fixed-nn/src/train_mlp.py)
# ==============================================================================

def train_mlp_model(X_train: np.ndarray, y_train: np.ndarray,
                    X_val: np.ndarray, y_val: np.ndarray, out_dim: int,
                    hidden_sizes: list = [16, 16], epochs: int = 10,
                    batch_size: int = 128, lr: float = 1e-3, seed: int = 0) -> MLP:
    """
    Trains MLP matching fixed-nn/src/train_mlp.py:
    - Optimizer: Adam(model.parameters(), lr=1e-3)
    - Loss: nn.CrossEntropyLoss()
    - No scheduler (constant learning rate matching the paper)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = MLP(in_dim=12, hidden_sizes=hidden_sizes, out_dim=out_dim).to(device)
    optimizer = Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    n_train = len(X_train)
    n_val = len(X_val)

    if device.type == "cuda":
        X_train_t = torch.FloatTensor(X_train).to(device)
        y_train_t = torch.LongTensor(y_train).to(device)
        X_val_t = torch.FloatTensor(X_val).to(device)
        y_val_t = torch.LongTensor(y_val).to(device)
    else:
        train_loader = DataLoader(FlowDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(FlowDataset(X_val, y_val), batch_size=min(4096, n_val), shuffle=False)

    best_loss = float("inf")
    best_weights = copy.deepcopy(model.state_dict())

    batches_per_epoch = (n_train + batch_size - 1) // batch_size
    print(f"[*] Training MLP (12 -> {hidden_sizes[0]} -> {hidden_sizes[1]} -> {out_dim}) on {device}")
    print(f"[*] Configuration: {epochs} epochs | Batch Size: {batch_size} ({batches_per_epoch:,} batches/epoch) | Adam lr={lr}")

    total_start = time.perf_counter()

    for epoch in range(epochs):
        ep_start = time.perf_counter()
        model.train()
        train_loss_acc = torch.zeros(1, device=device)

        if device.type == "cuda":
            perm = torch.randperm(n_train, device=device)
            X_shuffled = X_train_t[perm]
            y_shuffled = y_train_t[perm]
            for i in range(0, n_train, batch_size):
                batch_x = X_shuffled[i : i + batch_size]
                batch_y = y_shuffled[i : i + batch_size]
                optimizer.zero_grad(set_to_none=True)
                out = model(batch_x)
                loss = criterion(out, batch_y)
                loss.backward()
                optimizer.step()
                train_loss_acc += loss.detach() * len(batch_x)
            train_loss = (train_loss_acc / n_train).item()
        else:
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad(set_to_none=True)
                out = model(batch_x)
                loss = criterion(out, batch_y)
                loss.backward()
                optimizer.step()
                train_loss_acc += loss.detach() * len(batch_x)
            train_loss = (train_loss_acc / n_train).item()

        # Validation
        model.eval()
        val_loss_acc = torch.zeros(1, device=device)
        correct_acc = torch.zeros(1, device=device, dtype=torch.long)

        eval_bs = 32768 if device.type == "cuda" else 4096
        with torch.no_grad():
            if device.type == "cuda":
                for i in range(0, n_val, eval_bs):
                    bx = X_val_t[i : i + eval_bs]
                    by = y_val_t[i : i + eval_bs]
                    out = model(bx)
                    loss = criterion(out, by)
                    val_loss_acc += loss.detach() * len(bx)
                    correct_acc += (torch.argmax(out, dim=1) == by).sum()
                val_loss = (val_loss_acc / n_val).item()
                val_acc = (correct_acc.float() / n_val).item()
            else:
                for bx, by in val_loader:
                    out = model(bx)
                    loss = criterion(out, by)
                    val_loss_acc += loss.detach() * len(bx)
                    correct_acc += (torch.argmax(out, dim=1) == by).sum()
                val_loss = (val_loss_acc / n_val).item()
                val_acc = (correct_acc.float() / n_val).item()

        if val_loss < best_loss:
            best_loss = val_loss
            best_weights = copy.deepcopy(model.state_dict())

        ep_sec = time.perf_counter() - ep_start
        rate = n_train / max(0.001, ep_sec)
        print(f"    Epoch [{epoch+1:2d}/{epochs:2d}] ({ep_sec:.2f}s, {rate:,.0f} samples/s) - "
              f"Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f} | Val Acc: {val_acc*100:.2f}%")

    print(f"[+] Completed MLP training in {time.perf_counter() - total_start:.2f}s.")
    model.load_state_dict(best_weights)
    model.cpu()
    return model


# ==============================================================================
# 7. POST-TRAINING QUANTIZATION & C PARAM EXPORT (quantize.py & create_mlp_c_params.py)
# ==============================================================================

def quantize_mlp_model(model: MLP, X_val: np.ndarray, is_binary: bool) -> Dict[str, Any]:
    """
    Post-Training Quantization (PTQ) strictly replicating quantize.py.
    """
    out_dim = 2 if is_binary else 7
    scale_factor = 127
    state_dict = {}

    if PYTORCH_QUANT_AVAILABLE:
        print("[*] Running Post-Training Quantization (PTQ) via NVIDIA pytorch_quantization...")
        try:
            quant_nn.QuantLinear.set_default_quant_desc_input(QuantDescriptor(num_bits=8, calib_method="histogram"))
            q_model = QuantMLP(in_dim=12, hidden_sizes=[16, 16], out_dim=out_dim)
            q_model.load_state_dict(model.state_dict(), strict=False)
            q_model.eval()

            # Enable calibrators
            for name, module in q_model.named_modules():
                if isinstance(module, quant_nn.TensorQuantizer) and module._calibrator is not None:
                    module.disable_quant()
                    module.enable_calib()
                    if isinstance(module._calibrator, calib.HistogramCalibrator):
                        module._calibrator._num_bins = 128
                elif isinstance(module, quant_nn.TensorQuantizer):
                    module.disable()

            # Feed validation samples
            calib_bs = 4096
            with torch.no_grad():
                for i in range(0, len(X_val), calib_bs):
                    q_model(torch.FloatTensor(X_val[i : i + calib_bs]))

            # Disable calibrators and enable quant
            for _, module in q_model.named_modules():
                if isinstance(module, quant_nn.TensorQuantizer) and module._calibrator is not None:
                    module.enable_quant()
                    module.disable_calib()
                elif isinstance(module, quant_nn.TensorQuantizer):
                    module.enable()

            # Compute amax (percentile 99.99 preserves attack tails without over-clipping)
            for name, module in q_model.named_modules():
                if isinstance(module, quant_nn.TensorQuantizer) and module._calibrator is not None:
                    try:
                        module.load_calib_amax(method="percentile", percentile=99.99, strict=False)
                    except Exception:
                        try:
                            module.load_calib_amax(method="entropy", strict=False)
                        except Exception:
                            module.load_calib_amax(strict=False)

            # Extract weights and scale factors matching quantize_model_params() in quantize.py
            indices = [0, 2, 4]
            for layer_idx, idx in enumerate(indices, start=1):
                lin = q_model.net[idx]
                w = lin.weight.detach().cpu()
                if hasattr(lin, '_weight_quantizer') and lin._weight_quantizer._amax is not None:
                    s_w = lin._weight_quantizer._amax.detach().cpu().numpy()
                else:
                    s_w = torch.max(torch.abs(w), dim=1, keepdim=True)[0].numpy()

                if hasattr(lin, '_input_quantizer') and lin._input_quantizer._amax is not None:
                    s_x = lin._input_quantizer._amax.detach().cpu().numpy()
                else:
                    s_x = np.max(np.abs(X_val))

                s_w = np.maximum(np.array(s_w), 1e-6)
                s_x = np.maximum(float(np.squeeze(s_x)), 1e-6)

                w_np = w.numpy() if isinstance(w, torch.Tensor) else np.array(w)
                scale = w_np * (scale_factor / s_w)
                w_q = np.clip(np.round(scale), -scale_factor, scale_factor).astype(np.int8)
                state_dict[f"layer_{layer_idx}_weight"] = w_q.T
                state_dict[f"layer_{layer_idx}_s_x"] = float(scale_factor / s_x)
                state_dict[f"layer_{layer_idx}_s_x_inv"] = float(s_x / scale_factor)
                state_dict[f"layer_{layer_idx}_s_w_inv"] = (s_w / scale_factor).squeeze()

            return state_dict

        except Exception as e:
            print(f"[!] PyTorch quantization toolkit error: {e}. Using analytical fallback.")

    # Analytical min-max fallback
    print("[*] Performing analytical min-max quantization...")
    val_t = torch.FloatTensor(X_val)
    model.eval()
    with torch.no_grad():
        a0 = val_t
        amax_x1 = max(float(torch.max(torch.abs(a0)).item()), 1e-6)
        z1 = model.net[0](a0)
        a1 = model.net[1](z1)
        amax_x2 = max(float(torch.max(torch.abs(a1)).item()), 1e-6)
        z2 = model.net[2](a1)
        a2 = model.net[3](z2)
        amax_x3 = max(float(torch.max(torch.abs(a2)).item()), 1e-6)

    amaxes_x = [amax_x1, amax_x2, amax_x3]
    for layer_idx, lin in enumerate([model.net[0], model.net[2], model.net[4]], start=1):
        w = lin.weight.detach().cpu().numpy()
        s_w = np.maximum(np.max(np.abs(w), axis=1, keepdims=True), 1e-6)
        w_q = np.clip(np.round(w * (scale_factor / s_w)), -scale_factor, scale_factor).astype(np.int8)
        s_x = amaxes_x[layer_idx - 1]
        state_dict[f"layer_{layer_idx}_weight"] = w_q.T
        state_dict[f"layer_{layer_idx}_s_x"] = float(scale_factor / s_x)
        state_dict[f"layer_{layer_idx}_s_x_inv"] = float(s_x / scale_factor)
        state_dict[f"layer_{layer_idx}_s_w_inv"] = (s_w / scale_factor).flatten()

    return state_dict


def export_c_mlp_files(state_dict: Dict[str, Any], scaler: MinMaxScaler,
                       is_binary: bool, output_dir: Path):
    """
    Writes mlp_params.h and mlp_params.c strictly matching create_mlp_c_params.py.
    """
    h_file = output_dir / "mlp_params.h"
    c_file = output_dir / "mlp_params.c"
    out_dim = 2 if is_binary else 7

    data_min_fxp = (scaler.data_min_ * MULTIPLIER_FXP).round().astype(np.int64).tolist()
    data_scale_fxp = (scaler.scale_ * MULTIPLIER_FXP).round().astype(np.int64).tolist()

    with open(h_file, "w") as f:
        f.write("#ifndef MLP_PARAMS\n#define MLP_PARAMS\n#include <stdint.h>\n\n")
        f.write("#define INPUT_DIM 12\n#define H1 16\n#define H2 16\n")
        f.write(f"#define OUTPUT_DIM {out_dim}\n\n")
        f.write(f"extern const int64_t data_min[{len(data_min_fxp)}];\n")
        f.write(f"extern const int64_t data_scale[{len(data_scale_fxp)}];\n\n")
        for layer_idx in range(1, 4):
            f.write(f"extern const int layer_{layer_idx}_s_x;\n")
            f.write(f"extern const int layer_{layer_idx}_s_x_inv;\n")
            f.write(f"extern const int layer_{layer_idx}_s_w_inv[{len(state_dict[f'layer_{layer_idx}_s_w_inv'])}];\n")
        f.write("\n")
        for layer_idx in range(1, 4):
            f.write(f"extern const int8_t layer_{layer_idx}_weight[{len(state_dict[f'layer_{layer_idx}_weight'].flatten())}];\n")
        f.write("\n#endif\n")

    with open(c_file, "w") as f:
        f.write('#include "mlp_params.h"\n\n')
        f.write(f"const int64_t data_min[{len(data_min_fxp)}] = {{" + ", ".join(map(str, data_min_fxp)) + "};\n")
        f.write(f"const int64_t data_scale[{len(data_scale_fxp)}] = {{" + ", ".join(map(str, data_scale_fxp)) + "};\n\n")
        for layer_idx in range(1, 4):
            sx = int(round(state_dict[f'layer_{layer_idx}_s_x'] * MULTIPLIER_FXP))
            sx_inv = int(round(state_dict[f'layer_{layer_idx}_s_x_inv'] * MULTIPLIER_FXP))
            f.write(f"const int layer_{layer_idx}_s_x = {sx};\n")
            f.write(f"const int layer_{layer_idx}_s_x_inv = {sx_inv};\n")
            sw_inv = np.round(state_dict[f'layer_{layer_idx}_s_w_inv'] * MULTIPLIER_FXP).astype(int)
            f.write(f"const int layer_{layer_idx}_s_w_inv[{len(sw_inv)}] = {{" + ", ".join(map(str, sw_inv)) + "};\n\n")
        for layer_idx in range(1, 4):
            w_flat = state_dict[f'layer_{layer_idx}_weight'].flatten().astype(int)
            f.write(f"const int8_t layer_{layer_idx}_weight[{len(w_flat)}] = {{" + ", ".join(map(str, w_flat)) + "};\n\n")


def compile_mlp_c(c_source_dir: Path, work_dir: Path) -> CDLL:
    for fname in ["mlp.c", "nn_math.c", "nn.c", "mlp.h", "nn_math.h", "nn.h"]:
        src = c_source_dir / fname
        if not src.exists():
            src = c_source_dir / "include" / fname
        if src.exists() and not (work_dir / fname).exists():
            shutil.copy(src, work_dir / fname)

    include_dir = work_dir / "include"
    inc_arg = f"-I{include_dir}" if include_dir.exists() else f"-I{work_dir}"
    cmd = ["gcc", "-Wall", "-O3", "-fPIC", "-shared", inc_arg,
           str(work_dir / "mlp_params.c"), str(work_dir / "mlp.c"),
           str(work_dir / "nn_math.c"), str(work_dir / "nn.c"),
           "-o", str(work_dir / "mlp.so")]
    subprocess.run(cmd, check=True)
    return CDLL(str(work_dir / "mlp.so"))


def evaluate_mlp_c(cdll: CDLL, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Evaluates quantized C model matching test_mlp_c.py.
    """
    run_mlp_func = cdll.run_mlp
    c_int64_p = POINTER(c_longlong)
    c_uint_p = POINTER(c_uint)
    run_mlp_func.argtypes = [c_int64_p, c_uint, c_uint_p]
    run_mlp_func.restype = None

    X_fxp = np.ascontiguousarray(np.round(X_test * MULTIPLIER_FXP), dtype=np.int64)
    n_samples = len(X_fxp)
    n_bench = min(10000, n_samples)

    preds = np.zeros(n_samples, dtype=np.int64)
    latencies = []
    class_buf = np.zeros(1, dtype=np.uintc)
    class_p = class_buf.ctypes.data_as(c_uint_p)

    for i in range(n_bench):
        sample_p = X_fxp[i].ctypes.data_as(c_int64_p)
        start = time.perf_counter()
        run_mlp_func(sample_p, c_uint(1), class_p)
        end = time.perf_counter()
        latencies.append((end - start) * 1e6)
        preds[i] = int(class_buf[0])

    for i in range(n_bench, n_samples):
        sample_p = X_fxp[i].ctypes.data_as(c_int64_p)
        run_mlp_func(sample_p, c_uint(1), class_p)
        preds[i] = int(class_buf[0])

    return preds, np.array(latencies)


def train_and_eval_nn(X_train: np.ndarray, y_train: np.ndarray,
                      X_val: np.ndarray, y_val: np.ndarray,
                      X_test: np.ndarray, y_test: np.ndarray,
                      scaler: MinMaxScaler, is_binary: bool,
                      epochs: int, batch_size: int, lr: float,
                      c_source_dir: Path, work_dir: Path,
                      seed: int = 0,
                      pretrained_checkpoint: Optional[str] = None) -> Dict[str, Any]:
    out_dim = 2 if is_binary else 7
    print(f"\n{'='*30} NEURAL NETWORK (MLP 12->16->16->{out_dim}, Binary={is_binary}) {'='*30}")

    loaded_pretrained = False
    if pretrained_checkpoint:
        task_sub = "binary-classification" if is_binary else "multi-classification"
        ckpt_cand = None
        if pretrained_checkpoint.lower() == "auto":
            search_paths = [
                c_source_dir.parent / "saved_models/nn" / task_sub / "16x16" / "mlp_pktflw.th",
                c_source_dir.parent / "saved_models" / task_sub / "16x16" / "mlp_pktflw.th",
                Path("../saved_models/nn") / task_sub / "16x16" / "mlp_pktflw.th",
            ]
            for p in search_paths:
                if p.exists():
                    ckpt_cand = p
                    break
        else:
            p = Path(pretrained_checkpoint)
            if p.exists():
                ckpt_cand = p

        if ckpt_cand and ckpt_cand.exists():
            print(f"[*] Loading pre-trained checkpoint from: {ckpt_cand}")
            saved = torch.load(ckpt_cand, map_location="cpu", weights_only=False)
            model = MLP(in_dim=12, hidden_sizes=[16, 16], out_dim=out_dim)
            model.load_state_dict(saved.get("state_dict", saved))
            if "scaler" in saved and saved["scaler"] is not None:
                scaler = saved["scaler"]
            loaded_pretrained = True

    if not loaded_pretrained:
        ckpt_cand = None
        model = train_mlp_model(X_train, y_train, X_val, y_val, out_dim=out_dim,
                                hidden_sizes=[16, 16], epochs=epochs, batch_size=batch_size,
                                lr=lr, seed=seed)

    # 1. Evaluate Float MLP in Python
    model.eval()
    eval_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(eval_device)
    preds_list = []
    eval_bs = 32768 if eval_device.type == "cuda" else 4096
    with torch.no_grad():
        for i in range(0, len(X_test), eval_bs):
            bx = torch.FloatTensor(X_test[i : i + eval_bs]).to(eval_device)
            preds_list.append(torch.argmax(model(bx), dim=1).cpu().numpy())
    preds_py = np.concatenate(preds_list)

    # Benchmark single-sample host latency
    model_cpu = copy.deepcopy(model).cpu()
    model_cpu.eval()
    n_bench = min(10000, len(X_test))
    bench_samples = torch.FloatTensor(X_test[:n_bench])
    latencies_py = []
    with torch.no_grad():
        for i in range(n_bench):
            sample = bench_samples[i:i+1]
            start = time.perf_counter()
            _ = model_cpu(sample)
            end = time.perf_counter()
            latencies_py.append((end - start) * 1e6)

    acc_py = accuracy_score(y_test, preds_py)
    prec_py = precision_score(y_test, preds_py, average="weighted" if not is_binary else "binary", zero_division=0)
    rec_py = recall_score(y_test, preds_py, average="weighted" if not is_binary else "binary", zero_division=0)
    f1_py = f1_score(y_test, preds_py, average="weighted" if not is_binary else "binary", zero_division=0)
    rep_py = classification_report(y_test, preds_py, output_dict=True, zero_division=0)

    # 2. Post-Training Quantization
    q_state_dict = None
    if ckpt_cand:
        q_cand = ckpt_cand.parent / "mlp_pktflw_quant.th"
        if q_cand.exists():
            try:
                q_saved = torch.load(q_cand, map_location="cpu", weights_only=False)
                if "state_dict" in q_saved and "layer_1_weight" in q_saved["state_dict"]:
                    q_state_dict = q_saved["state_dict"]
            except Exception:
                pass

    if q_state_dict is None:
        q_state_dict = quantize_mlp_model(model, X_val, is_binary=is_binary)

    # 3. Compile mlp.so and evaluate C fixed-point model
    task_dir = work_dir / f"nn_{'binary' if is_binary else 'multi'}"
    task_dir.mkdir(parents=True, exist_ok=True)
    export_c_mlp_files(q_state_dict, scaler, is_binary, task_dir)
    cdll = compile_mlp_c(c_source_dir, task_dir)
    preds_c, latencies_c = evaluate_mlp_c(cdll, X_test)

    acc_c = accuracy_score(y_test, preds_c)
    prec_c = precision_score(y_test, preds_c, average="weighted" if not is_binary else "binary", zero_division=0)
    rec_c = recall_score(y_test, preds_c, average="weighted" if not is_binary else "binary", zero_division=0)
    f1_c = f1_score(y_test, preds_c, average="weighted" if not is_binary else "binary", zero_division=0)
    rep_c = classification_report(y_test, preds_c, output_dict=True, zero_division=0)

    print(f"[*] NN (float)  Accuracy: {acc_py:.4f} | Prec: {prec_py:.4f} | Rec: {rec_py:.4f} | F1: {f1_py:.4f} | Latency: {np.mean(latencies_py):.2f} μs")
    print(f"[*] NN (fixed)  Accuracy: {acc_c:.4f} | Prec: {prec_c:.4f} | Rec: {rec_c:.4f} | F1: {f1_c:.4f} | Latency: {np.mean(latencies_c):.2f} μs")

    return {
        "nn_float": {
            "accuracy": acc_py, "precision": prec_py, "recall": rec_py, "f1": f1_py,
            "latency_mean_us": float(np.mean(latencies_py)), "latency_std_us": float(np.std(latencies_py)),
            "report": rep_py
        },
        "nn_fixed": {
            "accuracy": acc_c, "precision": prec_c, "recall": rec_c, "f1": f1_c,
            "latency_mean_us": float(np.mean(latencies_c)), "latency_std_us": float(np.std(latencies_c)),
            "report": rep_c
        }
    }


# ==============================================================================
# 8. OUTPUT TABLES (Table 6 & Table 7 Replication)
# ==============================================================================

def print_table_6(results_binary: Dict[str, Any], feature_mode: str):
    title = f"Table 6: Binary Classification Performance ({feature_mode.upper()})"
    print(f"\n{title}")
    print("=" * len(title))
    header = f"{'Classifier':<16} | {'Data format':<11} | {'Accuracy':<8} | {'Precision':<9} | {'Recall':<8} | {'F1-score':<8} | {'Latency [μs]':<12}"
    print(header)
    print("-" * len(header))

    models = [
        ("DT (float)", "float32", results_binary.get("dt_float")),
        ("DT (fixed)", "int64",   results_binary.get("dt_fixed")),
        ("NN (float)", "float32", results_binary.get("nn_float")),
        ("NN (fixed)", "int8 PTQ",results_binary.get("nn_fixed")),
    ]

    for name, fmt, res in models:
        if res:
            acc = f"{res['accuracy']:.3f}"
            prec = f"{res['precision']:.3f}"
            rec = f"{res['recall']:.3f}"
            f1 = f"{res['f1']:.3f}"
            lat = f"{res['latency_mean_us']:.2f}"
            print(f"{name:<16} | {fmt:<11} | {acc:<8} | {prec:<9} | {rec:<8} | {f1:<8} | {lat:<12}")
    print("-" * len(header))


def print_table_7(results_multi: Dict[str, Any], feature_mode: str):
    title = f"Table 7: Multi-Class Detection Metrics (7 Categories, {feature_mode.upper()})"
    print(f"\n{title}")
    print("=" * len(title))
    header = f"{'Traffic Category':<16} | {'DT (float)':<22} | {'DT (fixed)':<22} | {'NN (float)':<22} | {'NN (fixed)':<22}"
    print(header)
    print(f"{'':<16} | {'Prec / Rec / F1':<22} | {'Prec / Rec / F1':<22} | {'Prec / Rec / F1':<22} | {'Prec / Rec / F1':<22}")
    print("-" * len(header))

    dt_f = results_multi.get("dt_float", {}).get("report", {})
    dt_c = results_multi.get("dt_fixed", {}).get("report", {})
    nn_f = results_multi.get("nn_float", {}).get("report", {})
    nn_c = results_multi.get("nn_fixed", {}).get("report", {})

    def fmt_cell(rep, cat_idx):
        str_idx = str(cat_idx)
        if str_idx in rep:
            p = rep[str_idx].get("precision", 0.0)
            r = rep[str_idx].get("recall", 0.0)
            f = rep[str_idx].get("f1-score", 0.0)
            return f"{p:.3f} / {r:.3f} / {f:.3f}"
        return "- / - / -"

    for cat_idx, cat_name in enumerate(LABEL_NAMES_7):
        c_dt_f = fmt_cell(dt_f, cat_idx)
        c_dt_c = fmt_cell(dt_c, cat_idx)
        c_nn_f = fmt_cell(nn_f, cat_idx)
        c_nn_c = fmt_cell(nn_c, cat_idx)
        print(f"{cat_name:<16} | {c_dt_f:<22} | {c_dt_c:<22} | {c_nn_f:<22} | {c_nn_c:<22}")

    print("-" * len(header))
    acc_dt_f = results_multi.get("dt_float", {}).get("accuracy", 0.0)
    acc_dt_c = results_multi.get("dt_fixed", {}).get("accuracy", 0.0)
    acc_nn_f = results_multi.get("nn_float", {}).get("accuracy", 0.0)
    acc_nn_c = results_multi.get("nn_fixed", {}).get("accuracy", 0.0)
    print(f"{'Accuracy':<16} | {acc_dt_f:<22.3f} | {acc_dt_c:<22.3f} | {acc_nn_f:<22.3f} | {acc_nn_c:<22.3f}")

    lat_dt_f = results_multi.get("dt_float", {}).get("latency_mean_us", 0.0)
    lat_dt_c = results_multi.get("dt_fixed", {}).get("latency_mean_us", 0.0)
    lat_nn_f = results_multi.get("nn_float", {}).get("latency_mean_us", 0.0)
    lat_nn_c = results_multi.get("nn_fixed", {}).get("latency_mean_us", 0.0)
    print(f"{'Latency [μs]':<16} | {lat_dt_f:<22.2f} | {lat_dt_c:<22.2f} | {lat_nn_f:<22.2f} | {lat_nn_c:<22.2f}")
    print("-" * len(header))


def export_results_to_files(all_results: Dict[str, Any], output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "summary_stage1.json"
    csv_path = output_dir / "summary_stage1.csv"

    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=4)
    print(f"[+] Saved JSON summary: {json_path}")

    csv_rows = ["task,feature_mode,model,format,accuracy,precision,recall,f1_score,latency_mean_us,latency_std_us"]
    for task in ["binary", "multi"]:
        if task in all_results:
            for f_mode, f_res in all_results[task].items():
                for m_key, m_val in f_res.items():
                    fmt = "float32" if "float" in m_key else ("int64" if "dt" in m_key else "int8 PTQ")
                    row = f"{task},{f_mode},{m_key},{fmt},{m_val['accuracy']:.5f},{m_val['precision']:.5f},{m_val['recall']:.5f},{m_val['f1']:.5f},{m_val['latency_mean_us']:.3f},{m_val['latency_std_us']:.3f}"
                    csv_rows.append(row)

    with open(csv_path, "w") as f:
        f.write("\n".join(csv_rows) + "\n")
    print(f"[+] Saved CSV summary:  {csv_path}")


# ==============================================================================
# 9. MAIN EXECUTION
# ==============================================================================

def detect_repo_paths(user_repo_dir: Optional[str] = None):
    candidate = Path(user_repo_dir).resolve() if user_repo_dir else Path.cwd()
    if candidate.name in ["src", "fixed-nn", "fixed-dt"]:
        candidate = candidate.parent
    if candidate.name in ["fixed-nn", "fixed-dt"]:
        candidate = candidate.parent

    if (candidate / "Kernel-Packet-Processing-Empowered-by-Lightweight-Neural-Network-and-Decision-Tree").exists():
        default_sub = candidate / "Kernel-Packet-Processing-Empowered-by-Lightweight-Neural-Network-and-Decision-Tree"
    elif (candidate / "Kernel-Packet-Processing-Empowered-by-Lightweight-Neural-Network-and-Decision-Tree-original").exists():
        default_sub = candidate / "Kernel-Packet-Processing-Empowered-by-Lightweight-Neural-Network-and-Decision-Tree-original"
    else:
        default_sub = candidate

    target_repo = default_sub if not user_repo_dir else candidate
    dt_dir = target_repo / "fixed-dt"
    nn_dir = target_repo / "fixed-nn" / "src"
    data_dir = target_repo / "dataset"
    default_mode = "mad" if "original" in target_repo.name.lower() else "scv"

    return target_repo, dt_dir, nn_dir, data_dir, default_mode


def main():
    parser = argparse.ArgumentParser(description="Stage I ML Evaluation (Hara & Sasabe Replication)")
    parser.add_argument("--repo-dir", type=str, default=None, help="Path to repository")
    parser.add_argument("--task", type=str, choices=["binary", "multi", "both"], default="both", help="Task")
    parser.add_argument("--model", type=str, choices=["dt", "nn", "both"], default="both", help="Model")
    parser.add_argument("--feature-mode", type=str, choices=["mad", "scv", "both", "auto"], default="auto", help="Feature mode")
    parser.add_argument("--data-dir", type=str, default=None, help="Directory containing flows.pickle")
    parser.add_argument("--output-dir", type=str, default="./results_stage1", help="Output directory")
    parser.add_argument("--max-length", type=int, default=100, help="Max packets per flow (default: 100)")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs for MLP (default: 10, matching train_mlp.py)")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size (default: 128, matching train_mlp.py)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: 1e-3, matching train_mlp.py)")
    parser.add_argument("--max-depth", type=int, default=5, help="Decision Tree max depth (default: 5)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--mock-data", action="store_true", help="Synthetic smoke-test without flows.pickle")
    parser.add_argument("--no-cache", action="store_true", help="Bypass .npz cache and re-extract from pickle")
    parser.add_argument("--pretrained-nn", type=str, default=None, help="Path to pre-trained checkpoint or 'auto'")

    args = parser.parse_args()

    if not SKLEARN_AVAILABLE:
        print("[!] Error: scikit-learn is required.")
        sys.exit(1)

    repo_dir, dt_dir, nn_dir, detected_data_dir, auto_mode = detect_repo_paths(args.repo_dir)
    data_dir = Path(args.data_dir).resolve() if args.data_dir else detected_data_dir
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    f_modes = [auto_mode] if args.feature_mode == "auto" else ([args.feature_mode] if args.feature_mode != "both" else ["mad", "scv"])
    tasks = ["binary", "multi"] if args.task == "both" else [args.task]

    print("=" * 80)
    print("STAGE I: USER-SPACE MACHINE LEARNING EVALUATION (Hara & Sasabe Replication)")
    print("=" * 80)
    print(f"Target Repository:  {repo_dir}")
    print(f"Dataset Directory:  {data_dir}")
    print(f"Output Directory:   {output_dir}")
    print(f"Tasks:              {tasks}")
    print(f"Models:             {args.model}")
    print(f"Feature Modes:      {f_modes}")
    print(f"Epochs:             {args.epochs}")
    print(f"Batch Size:         {args.batch_size}")
    print(f"Learning Rate:      {args.lr}")
    print(f"Pretrained NN:      {args.pretrained_nn}")
    print("=" * 80)

    all_results = {}

    for f_mode in f_modes:
        print(f"\n[*] Loading features for: {f_mode.upper()} ...")
        X_raw, y_bin, y_cat = load_dataset(
            data_dir=data_dir, output_dir=output_dir, feature_mode=f_mode,
            max_length=args.max_length, use_mock=args.mock_data,
            seed=args.seed, no_cache=args.no_cache
        )

        for task in tasks:
            is_binary = (task == "binary")
            if task not in all_results:
                all_results[task] = {}

            print(f"\n{'#'*35} TASK: {task.upper()} | FEATURES: {f_mode.upper()} {'#'*35}")
            task_work_dir = output_dir / task / f_mode
            task_work_dir.mkdir(parents=True, exist_ok=True)

            y = y_bin if is_binary else np.array([RAW_TO_7_MAPPING.get(int(c), 0) for c in y_cat], dtype=np.int64)

            (X_raw_train, y_train), (X_raw_val, y_val), (X_raw_test, y_test) = split_3fold(
                X_raw, y, fold=0, n_fold=3, train_val_split=0.8, seed=args.seed
            )

            # Fit MinMaxScaler for Neural Network (per train_mlp.py)
            scaler = MinMaxScaler()
            X_train = scaler.fit_transform(X_raw_train)
            X_val = scaler.transform(X_raw_val)
            X_test = scaler.transform(X_raw_test)

            task_results = {}

            # DT: Trained & evaluated on raw unscaled features (per learn_dt_float.py & learn_dt.py)
            if args.model in ["dt", "both"]:
                dt_source = dt_dir / "dt.c"
                dt_res = train_and_eval_dt(X_raw_train, y_train, X_raw_test, y_test,
                                           is_binary=is_binary, max_depth=args.max_depth,
                                           dt_source_path=dt_source, work_dir=task_work_dir, seed=args.seed)
                task_results.update(dt_res)

            # NN: Trained on scaled features with Adam lr=1e-3, batch_size=128 (per train_mlp.py)
            if args.model in ["nn", "both"]:
                nn_res = train_and_eval_nn(X_train, y_train, X_val, y_val, X_test, y_test,
                                           scaler=scaler, is_binary=is_binary,
                                           epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                                           c_source_dir=nn_dir, work_dir=task_work_dir, seed=args.seed,
                                           pretrained_checkpoint=args.pretrained_nn)
                task_results.update(nn_res)

            all_results[task][f_mode] = task_results

            if is_binary:
                print_table_6(task_results, feature_mode=f_mode)
            else:
                print_table_7(task_results, feature_mode=f_mode)

    export_results_to_files(all_results, output_dir)
    print("\n[+] Stage I evaluation pipeline completed successfully.")


if __name__ == "__main__":
    main()
