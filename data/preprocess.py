"""
Data preprocessing for IoTID20 and CICIDS2017 datasets.

Outputs:
  - train.npz / test.npz: feature matrices + labels
  - partitions.json: Dirichlet partition indices per device
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from pathlib import Path


# ============================================================
# IoTID20 preprocessing
# ============================================================
def preprocess_iotid20(raw_path: str, output_dir: str,
                       alpha: float = 0.5, N: int = 50,
                       test_ratio: float = 0.2, seed: int = 1):
    """
    IoTID20: 625,783 records, 83 columns = 80 features + 3 label fields.
    Target: 80-dim float32 features + binary label (0=normal, 1=attack).
    """
    print(f"[IoTID20] Loading from {raw_path} ...")
    df = pd.read_csv(raw_path)

    # --- Identify label columns ---
    # IoTID20 has 3 label columns: typically 'Label', 'Attack_Type', and one more
    # Keep 'Label' for binary classification; 'Attack_Type' for multi-class (optional)
    label_cols = [c for c in df.columns if c.lower() in
                  ['label', 'attack_type', 'class', 'cat', 'subcategory']]
    if len(label_cols) == 0:
        # Fallback: assume last 3 columns are labels
        label_cols = df.columns[-3:].tolist()
        print(f"[IoTID20] No explicit label columns found; "
              f"assuming last 3: {label_cols}")

    feature_cols = [c for c in df.columns if c not in label_cols]
    print(f"[IoTID20] Feature columns: {len(feature_cols)}, "
          f"Label columns: {len(label_cols)}")

    # --- Clean features ---
    X = df[feature_cols].copy()
    # Remove non-numeric columns
    non_numeric = X.select_dtypes(exclude=[np.number]).columns
    if len(non_numeric) > 0:
        print(f"[IoTID20] Dropping non-numeric columns: {list(non_numeric)}")
        X = X.drop(columns=non_numeric)

    # Fill NaN / Inf
    X = X.replace([np.inf, -np.inf], np.nan)
    nan_counts = X.isna().sum()
    cols_with_nan = nan_counts[nan_counts > 0]
    if len(cols_with_nan) > 0:
        print(f"[IoTID20] Columns with NaN: {cols_with_nan.to_dict()}")
        X = X.fillna(X.median())

    # Drop rows that still have NaN (should be 0 after median fill)
    X = X.dropna()
    print(f"[IoTID20] Cleaned feature dimension: {X.shape[1]}")

    # --- Binary label ---
    # Find the main label column
    label_col = [c for c in label_cols if c.lower() == 'label']
    if len(label_col) == 0:
        label_col = label_cols[:1]  # use first label column
    label_col = label_col[0]

    # Map: 'Normal' / 0 → 0, anything else → 1
    y_raw = df.loc[X.index, label_col]
    y = (y_raw != 'Normal').astype(int).values
    print(f"[IoTID20] Label distribution: normal={np.sum(y==0)}, "
          f"attack={np.sum(y==1)}")

    # --- Z-score standardization ---
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.values).astype(np.float32)

    # --- Final feature dimension check ---
    actual_dim = X_scaled.shape[1]
    print(f"[IoTID20] Final feature dim: {actual_dim}")
    if actual_dim != 80:
        print(f"[IoTID20] WARNING: Expected 80 features, got {actual_dim}. "
              f"Adjust config.input_dim_iotid20 accordingly.")

    # --- Train/test split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_ratio, random_state=seed, stratify=y
    )

    # --- Dirichlet partition ---
    partitions = dirichlet_partition(y_train, N, alpha, seed)

    # --- Save ---
    os.makedirs(output_dir, exist_ok=True)
    np.savez(os.path.join(output_dir, 'iotid20_train.npz'),
             X=X_train, y=y_train)
    np.savez(os.path.join(output_dir, 'iotid20_test.npz'),
             X=X_test, y=y_test)
    # --- 保存 per-α 分区文件（避免 α Bug）---
    partition_filename = f'iotid20_partitions_alpha{alpha}.json'
    with open(os.path.join(output_dir, partition_filename), 'w', encoding='utf-8') as f:
        json.dump({
            'N': N, 'alpha': alpha, 'seed': seed,
            'feature_dim': actual_dim,
            'num_train': len(y_train), 'num_test': len(y_test),
            'normal_train': int(np.sum(y_train == 0)),
            'attack_train': int(np.sum(y_train == 1)),
            'device_indices': partitions
        }, f, indent=2)
    # 同时保存一份通用文件（兼容旧版本）
    with open(os.path.join(output_dir, 'iotid20_partitions.json'), 'w', encoding='utf-8') as f:
        json.dump({
            'N': N, 'alpha': alpha, 'seed': seed,
            'feature_dim': actual_dim,
            'num_train': len(y_train), 'num_test': len(y_test),
            'normal_train': int(np.sum(y_train == 0)),
            'attack_train': int(np.sum(y_train == 1)),
            'device_indices': partitions
        }, f, indent=2)
    print(f"[IoTID20] Saved to {output_dir} (partition: {partition_filename})")


# ============================================================
# IoTID20 from preprocessed .csv.gz chunks (将军的本地数据)
# ============================================================
def preprocess_iotid20_from_chunks(chunk_dir: str, output_dir: str,
                                    alpha: float = 0.5, N: int = 20,
                                    test_ratio: float = 0.2, seed: int = 1,
                                    skip_scaler: bool = True):
    """
    Read IoTID20 from 7 preprocessed .csv.gz chunk files (processed_chunk_1~7).
    These are already cleaned (Schema OK) and Z-score standardized, so
    by default skips re-standardization (skip_scaler=True).

    Data format (verified from trunk samples):
      - 86 columns: 80 features — 4 non-numeric (Flow_ID/Src_IP/Dst_IP/Timestamp)
        + 3 label columns (label/Cat/Sub_Cat)
      - 'label' is binary int64: 0=attack(93.6%), 1=normal(6.4%)（注意：与常见约定相反）
      - Features are already Z-score standardized (skip_scaler=True safe)
      - Cat/Sub_Cat are string attack categories (not used for binary training)
    """
    print(f"[IoTID20] Loading chunks from {chunk_dir} ...")
    chunk_dir_path = Path(chunk_dir)
    chunk_files = sorted(chunk_dir_path.glob('processed_chunk_*.csv.gz'))
    if not chunk_files:
        raise FileNotFoundError(
            f"No 'processed_chunk_*.csv.gz' found in {chunk_dir}. "
            f"Please check the path.")
    print(f"[IoTID20] Found {len(chunk_files)} chunk files")

    # --- Read & concatenate all chunks ---
    dfs = []
    for f in chunk_files:
        df_chunk = pd.read_csv(f, compression='gzip', nrows=None)
        print(f"  {f.name}: {len(df_chunk)} rows, {df_chunk.shape[1]} cols")
        dfs.append(df_chunk)
    df = pd.concat(dfs, ignore_index=True)
    print(f"[IoTID20] Combined: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"[IoTID20] Column names: {list(df.columns[:8])} ... "
          f"{list(df.columns[-5:])}")

    # --- Identify label columns ---
    label_cols = [c for c in df.columns if c.lower() in
                  ['label', 'attack_type', 'class', 'cat', 'subcategory']]
    if len(label_cols) == 0:
        label_cols = df.columns[-3:].tolist()
        print(f"[IoTID20] No explicit label columns; assuming last 3: "
              f"{label_cols}")
    feature_cols = [c for c in df.columns if c not in label_cols]
    print(f"[IoTID20] Features: {len(feature_cols)} cols, Labels: "
          f"{len(label_cols)} cols: {label_cols}")

    # --- Extract features (drop non-numeric) ---
    X = df[feature_cols].copy()
    non_numeric = X.select_dtypes(exclude=[np.number]).columns
    n_dropped = len(non_numeric)
    if n_dropped > 0:
        print(f"[IoTID20] Dropping non-numeric features: {list(non_numeric)}")
        X = X.drop(columns=non_numeric)

    # Handle NaN/Inf (data is preprocessed, but just in case)
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True))
    X = X.dropna()
    actual_dim = X.shape[1]
    print(f"[IoTID20] Cleaned feature dim: {actual_dim} "
          f"(dropped {n_dropped} non-numeric)")

    # --- Z-score standardization (skip if already standardized) ---
    if skip_scaler:
        X_scaled = X.values.astype(np.float32)
        print(f"[IoTID20] Skipped re-standardization (data already Z-scored)")
    else:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X.values).astype(np.float32)
        print(f"[IoTID20] Applied Z-score standardization")

    # --- Check dimension ---
    print(f"[IoTID20] Final feature dim: {actual_dim}")
    if actual_dim != 79:
        print(f"[IoTID20] INFO: Feature dim={actual_dim}. "
              f"Update configs/base.yaml input_dim_iotid20 accordingly.")

    # --- Binary label ---
    label_col = [c for c in label_cols if c.lower() == 'label']
    if len(label_col) == 0:
        label_col = label_cols[:1]
    label_col = label_col[0]
    y_raw = df.loc[X.index, label_col]

    if y_raw.dtype in [np.int64, np.int32, np.float64, np.float32]:
        y = y_raw.astype(int).values
    else:
        y = (y_raw != 'Normal').astype(int).values
    print(f"[IoTID20] Label dist: normal={np.sum(y==0)}, "
          f"attack={np.sum(y==1)} ({np.sum(y==1)/len(y)*100:.1f}%)")

    # --- Train/test split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_ratio, random_state=seed, stratify=y
    )

    # --- Dirichlet partition ---
    partitions = dirichlet_partition(y_train, N, alpha, seed)

    # --- Save ---
    os.makedirs(output_dir, exist_ok=True)
    np.savez(os.path.join(output_dir, 'iotid20_train.npz'),
             X=X_train, y=y_train)
    np.savez(os.path.join(output_dir, 'iotid20_test.npz'),
             X=X_test, y=y_test)
    # --- 保存 per-α 分区文件（避免 α Bug）---
    partition_filename = f'iotid20_partitions_alpha{alpha}.json'
    with open(os.path.join(output_dir, partition_filename), 'w', encoding='utf-8') as f:
        json.dump({
            'N': N, 'alpha': alpha, 'seed': seed,
            'feature_dim': actual_dim,
            'num_train': len(y_train), 'num_test': len(y_test),
            'normal_train': int(np.sum(y_train == 0)),
            'attack_train': int(np.sum(y_train == 1)),
            'device_indices': partitions
        }, f, indent=2)
    # 同时保存一份通用文件（兼容旧版本）
    with open(os.path.join(output_dir, 'iotid20_partitions.json'), 'w', encoding='utf-8') as f:
        json.dump({
            'N': N, 'alpha': alpha, 'seed': seed,
            'feature_dim': actual_dim,
            'num_train': len(y_train), 'num_test': len(y_test),
            'normal_train': int(np.sum(y_train == 0)),
            'attack_train': int(np.sum(y_train == 1)),
            'device_indices': partitions
        }, f, indent=2)
    print(f"[IoTID20] Preprocessing complete! Saved to {output_dir} (partition: {partition_filename})")


# ============================================================
# CICIDS2017 preprocessing
# ============================================================
def preprocess_cicids2017(raw_path: str, output_dir: str,
                          alpha: float = 0.5, N: int = 50,
                          test_ratio: float = 0.2, seed: int = 1):
    """
    CICIDS2017: Tuesday + Wednesday subsets → ~566,934 samples, 76 features.
    Raw data is multiple CSV files per day.
    """
    print(f"[CICIDS2017] Loading from {raw_path} ...")
    raw_dir = Path(raw_path)

    # Load Tuesday and Wednesday CSV files
    dfs = []
    for day in ['Tuesday', 'Wednesday']:
        day_files = list(raw_dir.glob(f'*{day}*.csv'))
        if len(day_files) == 0:
            # Try alternate naming: 'Tuesday-WorkingHours' etc.
            day_files = list(raw_dir.glob(f'*{day.lower()}*.csv'))
        if len(day_files) == 0:
            print(f"[CICIDS2017] WARNING: No files found for {day} "
                  f"in {raw_dir}. Trying all CSV files...")
            day_files = list(raw_dir.glob('*.csv'))

        for f in day_files:
            print(f"[CICIDS2017]   Reading {f.name} ...")
            df_day = pd.read_csv(f)
            dfs.append(df_day)

    df = pd.concat(dfs, ignore_index=True)
    print(f"[CICIDS2017] Combined shape: {df.shape}")

    # --- Clean column names (CICIDS2017 has extra spaces) ---
    df.columns = df.columns.str.strip()

    # --- Identify label column ---
    label_col = 'Label'
    if label_col not in df.columns:
        label_col = [c for c in df.columns if 'label' in c.lower()][0]
    print(f"[CICIDS2017] Label column: '{label_col}'")

    # --- Binary label ---
    y_raw = df[label_col]
    y = (y_raw != 'BENIGN').astype(int).values

    # --- Features ---
    feature_cols = [c for c in df.columns if c != label_col]
    X = df[feature_cols].copy()

    # Remove non-numeric
    non_numeric = X.select_dtypes(exclude=[np.number]).columns
    if len(non_numeric) > 0:
        print(f"[CICIDS2017] Dropping non-numeric: {list(non_numeric)}")
        X = X.drop(columns=non_numeric)

    # Handle Inf / NaN
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())
    X = X.dropna()
    print(f"[CICIDS2017] Cleaned feature dim: {X.shape[1]}")

    # Align y with X (after dropna)
    y = y[X.index]

    # --- Select exactly 76 features if more ---
    if X.shape[1] > 76:
        # Drop low-variance or constant columns first
        variances = X.var()
        const_cols = variances[variances == 0].index
        if len(const_cols) > 0:
            print(f"[CICIDS2017] Dropping constant columns: {list(const_cols)}")
            X = X.drop(columns=const_cols)

        if X.shape[1] > 76:
            # Keep top 76 by variance
            variances = X.var().sort_values(ascending=False)
            keep_cols = variances.head(76).index
            X = X[keep_cols]
            print(f"[CICIDS2017] Truncated to top-76 features by variance")

    # --- Z-score ---
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.values).astype(np.float32)

    actual_dim = X_scaled.shape[1]
    print(f"[CICIDS2017] Final feature dim: {actual_dim}, "
          f"samples: {len(y)}")
    if actual_dim != 76:
        print(f"[CICIDS2017] WARNING: Expected 76, got {actual_dim}.")

    # --- Train/test split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_ratio, random_state=seed, stratify=y
    )

    # --- Dirichlet partition ---
    partitions = dirichlet_partition(y_train, N, alpha, seed)

    # --- Save ---
    os.makedirs(output_dir, exist_ok=True)
    np.savez(os.path.join(output_dir, 'cicids_train.npz'),
             X=X_train, y=y_train)
    np.savez(os.path.join(output_dir, 'cicids_test.npz'),
             X=X_test, y=y_test)
    with open(os.path.join(output_dir, 'cicids_partitions.json'), 'w', encoding='utf-8') as f:
        json.dump({
            'N': N, 'alpha': alpha, 'seed': seed,
            'feature_dim': actual_dim,
            'num_train': len(y_train), 'num_test': len(y_test),
            'normal_train': int(np.sum(y_train == 0)),
            'attack_train': int(np.sum(y_train == 1)),
            'device_indices': partitions
        }, f, indent=2)
    print(f"[CICIDS2017] Saved to {output_dir}")


# ============================================================
# Dirichlet partition (standard FL non-IID protocol)
# ============================================================
def dirichlet_partition(labels: np.ndarray, N: int, alpha: float,
                        seed: int) -> dict:
    """
    Partition data across N devices using Dirichlet(alpha) distribution.

    For each class c, draw N proportions from Dir(alpha), then assign
    class-c samples proportionally to each device.

    Returns: dict {device_id: list_of_indices}
    """
    np.random.seed(seed)
    num_classes = len(np.unique(labels))
    device_indices = {i: [] for i in range(N)}

    for c in range(num_classes):
        class_idx = np.where(labels == c)[0]
        n_class = len(class_idx)
        # Draw Dirichlet proportions
        proportions = np.random.dirichlet([alpha] * N)
        # Proportional allocation
        splits = (proportions * n_class).astype(int)
        # Adjust remainder
        remainder = n_class - splits.sum()
        for r in range(remainder):
            splits[r % N] += 1
        # Shuffle and split
        perm = np.random.permutation(class_idx)
        start = 0
        for i in range(N):
            end = start + splits[i]
            device_indices[i].extend(perm[start:end].tolist())
            start = end

    # Sort indices per device for reproducibility
    for i in range(N):
        device_indices[i] = sorted(device_indices[i])

    # Stats
    sizes = [len(device_indices[i]) for i in range(N)]
    print(f"[Dirichlet] N={N}, α={alpha}: min={min(sizes)}, "
          f"max={max(sizes)}, mean={np.mean(sizes):.0f}")
    return device_indices


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='Preprocess datasets')
    parser.add_argument('--dataset', choices=['iotid20', 'cicids2017', 'both'],
                        default='both')
    parser.add_argument('--raw_dir', default='data/raw')
    parser.add_argument('--chunk_dir', default=None,
                        help='Directory with processed_chunk_*.csv.gz files '
                             '(for preprocessed IoTID20 chunks on local machine)')
    parser.add_argument('--output_dir', default='data/processed')
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--N', type=int, default=20,
                        help='Number of FL devices (default: 20 for lightweight)')
    parser.add_argument('--test_ratio', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()

    if args.dataset in ['iotid20', 'both']:
        if args.chunk_dir:
            # Read from preprocessed chunked gzip files
            preprocess_iotid20_from_chunks(
                args.chunk_dir, args.output_dir,
                alpha=args.alpha, N=args.N,
                test_ratio=args.test_ratio, seed=args.seed)
        else:
            # Read from single raw CSV (original mode)
            iotid20_path = os.path.join(args.raw_dir, 'IoTID20.csv')
            preprocess_iotid20(iotid20_path, args.output_dir,
                               alpha=args.alpha, N=args.N,
                               test_ratio=args.test_ratio, seed=args.seed)

    if args.dataset in ['cicids2017', 'both']:
        cicids_path = os.path.join(args.raw_dir, 'CICIDS2017')
        preprocess_cicids2017(cicids_path, args.output_dir,
                              alpha=args.alpha, N=args.N,
                              test_ratio=args.test_ratio, seed=args.seed)

    print("[Done] Preprocessing complete.")


if __name__ == '__main__':
    main()
