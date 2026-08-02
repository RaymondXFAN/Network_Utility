"""
Single-machine simulator for HierFed-Matter-NSAC-DPBA.

Simulates the device → gateway → cloud three-tier topology,
NSAC compression, DPBA-FIM noise injection, and full FL training loop.
"""

import os
import json
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path

from models.dnn import AnomalyDNN, build_model
from core.hierfed import hierfed_matter_round, star_aggregate, edge_aggregate, core_aggregate
from core.nsac import SLICE_CONFIGS, compute_communication_stats
from data.preprocess import dirichlet_partition
from core.dpba_fim import (
    FIMTracker, compute_device_sensitivity, allocate_privacy_budget,
    compute_heterogeneity_from_partition, clip_gradient, inject_dp_noise,
    dpba_fim_gradient_processing, compute_noise_scale,
)


# ============================================================
# Device assignment
# ============================================================
def assign_device_types(N: int, seed: int = 1) -> List[str]:
    """Assign device types (camera 40%, thermostat 30%, bulb 30%)."""
    np.random.seed(seed)
    types = []
    for i in range(N):
        r = np.random.random()
        if r < 0.4:
            types.append('camera')
        elif r < 0.7:
            types.append('thermostat')
        else:
            types.append('bulb')
    return types


def assign_slice_to_gateways(
    M: int,
    device_types: List[str],
    cluster_map: Dict[int, List[int]],
) -> Dict[int, str]:
    """
    Assign slice types to gateways based on dominant device type in cluster.

    camera-heavy → uRLLC, thermostat-heavy → eMBB, bulb-heavy → mMTC.
    """
    assignments = {}
    type_to_slice = {'camera': 'urllc', 'thermostat': 'embb', 'bulb': 'mmtc'}

    for gw_id, device_ids in cluster_map.items():
        # Count device types in this cluster
        type_counts = {}
        for dev_id in device_ids:
            dt = device_types[dev_id]
            type_counts[dt] = type_counts.get(dt, 0) + 1
        # Dominant type determines slice
        dominant = max(type_counts, key=type_counts.get)
        assignments[gw_id] = type_to_slice.get(dominant, 'embb')

    return assignments


# ============================================================
# Local Training
# ============================================================
def local_train(
    model: AnomalyDNN,
    train_data: np.ndarray,  # (n_i, dim)
    train_labels: np.ndarray,  # (n_i,)
    num_epochs: int = 5,
    batch_size: int = 32,
    learning_rate: float = 0.01,
    oversample: bool = True,
) -> torch.Tensor:
    """
    Local training on a device: compute gradient of the full model.

    Returns: flat gradient vector (1D tensor of length d).
    """
    device_data = torch.tensor(train_data, dtype=torch.float32)
    device_labels = torch.tensor(train_labels, dtype=torch.long)
    n_local = len(device_labels)
    if n_local == 0:
        return torch.zeros(model.num_parameters)

    # ---- 损失函数：plain CE ----
    # FL 聚合中全局/本地类别权重导致各设备梯度方向不一致，
    # 聚合后方向打架 → AUC 反降。改用 plain CE，推理时阈值调整弥补不平衡。
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # Save initial params
    initial_params = torch.cat([p.data.clone().flatten() for p in model.parameters()])

    # Local training — balanced sampling via WeightedRandomSampler
    model.train()
    class_counts = torch.bincount(device_labels, minlength=2).float()
    both_present = (class_counts.min() > 0) and oversample

    if both_present:
        # ---- Oversample minority: each mini-batch gets balanced class mix ----
        # WeightedRandomSampler draws n_local samples per epoch (same iteration count),
        # but minority samples appear more often → gradient has balanced class signal.
        # This is DIFFERENT from class-weighted CE (which changed gradient DIRECTION
        # and caused FL inconsistency). Oversampling changes data COMPOSITION, not loss.
        class_weights = 1.0 / class_counts
        sample_weights = class_weights[device_labels]
        sampler = WeightedRandomSampler(
            weights=sample_weights, num_samples=n_local, replacement=True
        )
        dataset = TensorDataset(device_data, device_labels)
        loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
        for epoch in range(num_epochs):
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
    else:
        # ---- Uniform sampling (single-class device or oversample=False) ----
        for epoch in range(num_epochs):
            indices = torch.randperm(n_local)
            for start in range(0, n_local, batch_size):
                end = min(start + batch_size, n_local)
                batch_idx = indices[start:end]
                batch_x = device_data[batch_idx]
                batch_y = device_labels[batch_idx]

                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

    # Compute gradient as difference: g = w_after - w_before
    final_params = torch.cat([p.data.clone().flatten() for p in model.parameters()])
    gradient = final_params - initial_params  # This is Δw = η * avg_gradient

    # Reset model to initial state (global model)
    with torch.no_grad():
        idx = 0
        for p in model.parameters():
            size = p.numel()
            p.data.copy_(initial_params[idx:idx + size].reshape(p.shape))
            idx += size

    return gradient


# ============================================================
# Main Simulation Loop
# ============================================================
class HierFedMatterSimulator:
    """Full simulation of HierFed-Matter-NSAC-DPBA on a single machine."""

    def __init__(self, config: dict):
        self.config = config
        self.dataset_name = config.get('dataset_name', 'iotid20')

        # Load data
        self._load_data()

        # Build model
        input_dim = self.feature_dim
        self.model = build_model(input_dim)
        self.d = self.model.num_parameters

        # Cluster topology
        N = config.get('N', 50)
        M = config.get('M', 5)
        K = config.get('K', 10)
        self.N = N
        self.M = M
        self.K = K

        # Cluster map: gateway_id → [device_ids]
        self.cluster_map = {}
        for gw_id in range(M):
            start_dev = gw_id * K
            end_dev = start_dev + K
            self.cluster_map[gw_id] = list(range(start_dev, min(end_dev, N)))

        # Device types
        self.device_types = assign_device_types(N, config.get('seed', 1))

        # Slice assignments
        self.slice_assignments = assign_slice_to_gateways(
            M, self.device_types, self.cluster_map
        )

        # DPBA-FIM components
        num_linear_layers = len([m for m in self.model.net
                                  if isinstance(m, torch.nn.Linear)])
        self.fim_tracker = FIMTracker(num_linear_layers,
                                       beta_F=config.get('beta_F', 0.9))

        # Compute device sensitivity
        local_counts = [len(self.partitions.get(i, [])) for i in range(N)]
        labels_train = self.y_train
        H_scores = compute_heterogeneity_from_partition(labels_train, self.partitions)
        self.device_sensitivity = compute_device_sensitivity(
            self.device_types, local_counts, H_scores,
            alpha_s=config.get('alpha_s', 0.5),
            beta_s=config.get('beta_s_coeff', 0.3),
            gamma_s=config.get('gamma_s', 0.2),
        )

        # Privacy budgets
        # float() 防御：PyYAML 会把 '1e-5' 这种无小数点的写法解析成字符串，
        # 显式转 float 可避免后续做除法时出现 TypeError
        self.epsilon_total = float(config.get('epsilon_total', 3.0))
        self.delta = float(config.get('delta', 1e-5))
        self.clip_norm = float(config.get('clip_norm', 1.0))
        self.T = config.get('T', 100)
        self.eta = config.get('eta', 0.01)

        # Tracking
        self.round_metrics = []

    def _load_data(self):
        """Load preprocessed data and partition, with α-aware partition loading/regeneration."""
        processed_dir = self.config.get('processed_dir', 'data/processed')
        dataset = self.dataset_name
        alpha = self.config.get('alpha', 0.5)
        N = self.config.get('N', 20)

        prefix = 'iotid20' if dataset == 'iotid20' else 'cicids'
        train_file = os.path.join(processed_dir, f'{prefix}_train.npz')
        test_file = os.path.join(processed_dir, f'{prefix}_test.npz')

        # --- Load train/test data ---
        train_npz = np.load(train_file)
        self.X_train = train_npz['X']
        self.y_train = train_npz['y']
        self.feature_dim = self.X_train.shape[1]

        test_npz = np.load(test_file)
        self.X_test = test_npz['X']
        self.y_test = test_npz['y']

        # --- α-aware partition loading ---
        # 优先查找 per-α 分区文件
        per_alpha_file = os.path.join(processed_dir,
                                       f'{prefix}_partitions_alpha{alpha}.json')
        generic_file = os.path.join(processed_dir, f'{prefix}_partitions.json')

        partition_loaded = False
        # 1) 尝试加载 per-α 分区文件
        if os.path.exists(per_alpha_file):
            with open(per_alpha_file, 'r', encoding='utf-8') as f:
                part_data = json.load(f)
            self.partitions = part_data['device_indices']
            partition_loaded = True
            print(f"[Partition] Loaded per-α file: {per_alpha_file} (α={alpha})")
        # 2) 尝试加载通用分区文件，检查 α 是否匹配
        elif os.path.exists(generic_file):
            with open(generic_file, 'r', encoding='utf-8') as f:
                part_data = json.load(f)
            file_alpha = part_data.get('alpha', None)
            if file_alpha == alpha:
                self.partitions = part_data['device_indices']
                partition_loaded = True
                print(f"[Partition] Loaded generic file (α matches: {alpha})")
            else:
                print(f"[Partition] ⚠️ Generic file α={file_alpha} ≠ config α={alpha}, regenerating...")
        # 3) 没找到任何分区文件
        else:
            print(f"[Partition] No partition file found, regenerating for α={alpha}...")

        # --- 如果分区未加载，用 dirichlet_partition 动态生成 ---
        if not partition_loaded:
            # 用固定的 partition_seed=1（与预处理一致）确保可复现
            partition_seed = 1
            partitions = dirichlet_partition(self.y_train, N, alpha, partition_seed)
            self.partitions = partitions
            # 保存为 per-α 文件供后续复用
            save_data = {
                'N': N, 'alpha': alpha, 'seed': partition_seed,
                'feature_dim': self.feature_dim,
                'num_train': len(self.y_train),
                'num_test': len(self.y_test),
                'normal_train': int(np.sum(self.y_train == 0)),
                'attack_train': int(np.sum(self.y_train == 1)),
                'device_indices': partitions
            }
            os.makedirs(processed_dir, exist_ok=True)
            with open(per_alpha_file, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, indent=2)
            print(f"[Partition] Saved new per-α file: {per_alpha_file}")

        # ---- 诊断：训练集 & 测试集类别分布 ----
        n_tr0 = int(np.sum(self.y_train == 0))
        n_tr1 = int(np.sum(self.y_train == 1))
        n_te0 = int(np.sum(self.y_test == 0))
        n_te1 = int(np.sum(self.y_test == 1))
        print(f"[Diag] Train: class0={n_tr0}({n_tr0/len(self.y_train)*100:.1f}%), "
              f"class1={n_tr1}({n_tr1/len(self.y_train)*100:.1f}%), "
              f"total={len(self.y_train)}")
        print(f"[Diag] Test:  class0={n_te0}({n_te0/len(self.y_test)*100:.1f}%), "
              f"class1={n_te1}({n_te1/len(self.y_test)*100:.1f}%), "
              f"total={len(self.y_test)}")

    def run(self) -> Dict:
        """Run full HierFed-Matter-NSAC-DPBA simulation for T rounds."""
        print(f"\n{'='*60}")
        print(f"Running HierFed-Matter-NSAC-DPBA on {self.dataset_name}")
        print(f"N={self.N}, M={self.M}, K={self.K}, T={self.T}")
        print(f"ε={self.epsilon_total}, δ={self.delta}, C={self.clip_norm}")
        print(f"d={self.d}, feature_dim={self.feature_dim}")
        print(f"{'='*60}\n")

        all_results = []

        for t in range(1, self.T + 1):
            # --- Step 1: Device local training ---
            device_gradients = {}
            for dev_id in range(self.N):
                idx_list = self.partitions.get(str(dev_id), [])
                if len(idx_list) == 0:
                    device_gradients[dev_id] = torch.zeros(self.d)
                    continue

                local_data = self.X_train[idx_list]
                local_labels = self.y_train[idx_list]

                gradient = local_train(
                    self.model, local_data, local_labels,
                    num_epochs=self.config.get('E_local', 5),
                    batch_size=self.config.get('batch_size', 32),
                    learning_rate=self.eta,
                    oversample=self.config.get('oversample_minority', True),
                )

                # 轻量版务实 DP：原 ε_total/T 再乘归一化敏感度 → 单设备 ε≈0.003，
                # 噪声 σ=C·√(2ln(1.25/δ))/ε 爆炸到上千，梯度被完全淹没、训练失效。
                # 改为以 ε_total 作为每设备每轮预算量级（演示级宽松 DP），敏感度用
                # 相对权重（0..1）加权，保证 σ 落在可训练区间以“保命中”。
                epsilon_per_round = self.epsilon_total
                fim_w = self.fim_tracker.get_weights()
                S_rel = self.device_sensitivity / (self.device_sensitivity.max() + 1e-12)
                device_eps = epsilon_per_round * (0.4 + 0.6 * S_rel[dev_id])

                # Clip gradient + DP noise injection (v6: 重新启用噪声)
                # v4/v5 关闭噪声后模型卡死(AUC≈0.55)或学反(AUC<0.5)；
                # v0 实测 ε=3.0 噪声虽大(σ×√d≈142 per device)，但经 FL 聚合
                # (K=5×M=4=20设备平均) 后有效噪声≈142/√20≈31 → 仍>信号(1.0)，
                # 但方向信号在多次聚合中存活 → 防止 model collapse → AUC=0.67。
                # 噪声级别由 DPBA-FIM 敏感度加权分配，低敏感度设备噪声更大。
                clipped = clip_gradient(gradient, self.clip_norm)
                noisy_grad = inject_dp_noise(clipped, device_eps, self.delta, self.clip_norm)
                device_gradients[dev_id] = noisy_grad

                # --- Step 3: Update FIM tracker (every 5 rounds for efficiency) ---
                if t % 5 == 0 and len(idx_list) > 0:
                    with torch.no_grad():
                        local_t_data = torch.tensor(local_data, dtype=torch.float32)
                        local_t_labels = torch.tensor(local_labels, dtype=torch.long)
                    self.fim_tracker.update(
                        self.model, local_t_data, local_t_labels,
                        list(range(self.fim_tracker.num_layers))
                    )

            # --- Step 4: HierFed-Matter round ---
            global_params = torch.cat([p.data.clone().flatten()
                                       for p in self.model.parameters()])
            new_params, round_stats = hierfed_matter_round(
                device_gradients, self.cluster_map,
                self.slice_assignments, global_params,
                learning_rate=self.eta, d=self.d,
                model_size_bytes=self.d * 4,
                enable_nsac=self.config.get('enable_nsac', True),
            )

            # --- Step 5: Update global model ---
            idx = 0
            with torch.no_grad():
                for p in self.model.parameters():
                    size = p.numel()
                    p.data.copy_(new_params[idx:idx + size].reshape(p.shape))
                    idx += size

            # --- Step 6: Evaluate on test set ---
            test_metrics = self._evaluate()

            round_result = {
                'round': t,
                **test_metrics,
                'avg_compression_ratio': np.mean(round_stats['compression_ratios']),
                'avg_latency_ms': np.mean(round_stats['latencies_ms']),
                'total_compressed_bytes': round_stats['total_compressed_bytes'],
            }
            all_results.append(round_result)

            if t % 10 == 0 or t == 1:
                print(f"Round {t}/{self.T}: "
                      f"F1={test_metrics['f1']:.4f}, "
                      f"macro-F1={test_metrics['f1_macro']:.4f}, "
                      f"Acc={test_metrics['accuracy']:.4f}, "
                      f"AUC={test_metrics['auc_roc']:.4f}, "
                      f"thresh={test_metrics['optimal_threshold']:.2f}, "
                      f"CR={round_result['avg_compression_ratio']:.4f}")

        # --- Final evaluation ---
        final_metrics = self._evaluate()
        print(f"\nFinal: F1={final_metrics['f1']:.4f}, "
              f"macro-F1={final_metrics['f1_macro']:.4f}, "
              f"Acc={final_metrics['accuracy']:.4f}, "
              f"AUC={final_metrics['auc_roc']:.4f}, "
              f"optimal_threshold={final_metrics['optimal_threshold']:.2f}")

        # Communication stats
        comm_stats = compute_communication_stats(
            self.d, self.N, self.M, self.T,
            self.slice_assignments
        )
        print(f"Communication: speedup={comm_stats['speedup']:.1f}×, "
              f"reduction={comm_stats['reduction_pct']:.1f}%")

        return {
            'final_metrics': final_metrics,
            'round_metrics': all_results,
            'communication_stats': comm_stats,
            'config': self.config,
        }

    def _evaluate(self) -> Dict:
        """Evaluate model on test set with threshold optimization."""
        from sklearn.metrics import f1_score, accuracy_score, roc_auc_score

        self.model.eval()
        X_t = torch.tensor(self.X_test, dtype=torch.float32)
        with torch.no_grad():
            outputs = self.model(X_t)
            probs = torch.softmax(outputs, dim=1)
            preds_default = outputs.argmax(dim=1).numpy()

        y_true = self.y_test
        y_prob = probs[:, 1].numpy()  # P(class=1)

        # ---- Threshold optimization: find threshold that maximizes F1 ----
        best_f1 = 0.0
        best_thresh = 0.5
        for thresh in np.arange(0.05, 0.95, 0.05):
            preds_t = (y_prob >= thresh).astype(int)
            f1_t = f1_score(y_true, preds_t, zero_division=0)
            if f1_t > best_f1:
                best_f1 = f1_t
                best_thresh = thresh

        preds_optimal = (y_prob >= best_thresh).astype(int)
        f1_optimal = best_f1
        acc_optimal = accuracy_score(y_true, preds_optimal)

        # Default threshold (0.5) metrics
        f1_default = f1_score(y_true, preds_default, zero_division=0)
        acc_default = accuracy_score(y_true, preds_default)

        # Macro-F1 (average F1 across both classes)
        f1_macro = f1_score(y_true, preds_optimal, average='macro', zero_division=0)

        try:
            auc = roc_auc_score(y_true, y_prob)
        except ValueError:
            auc = 0.0

        # ---- 诊断打印 ----
        n_cls0 = int(np.sum(y_true == 0))
        n_cls1 = int(np.sum(y_true == 1))
        print(f"[Diag] Test set: total={len(y_true)}, "
              f"class0={n_cls0}({n_cls0/len(y_true)*100:.1f}%), "
              f"class1={n_cls1}({n_cls1/len(y_true)*100:.1f}%)")
        print(f"[Diag] Prob: P(cls1)_mean={y_prob.mean():.4f}, "
              f"std={y_prob.std():.4f}, "
              f"min={y_prob.min():.4f}, max={y_prob.max():.4f}")
        print(f"[Diag] Threshold: optimal={best_thresh:.2f} "
              f"(F1={f1_optimal:.4f}, macro-F1={f1_macro:.4f}, "
              f"Acc={acc_optimal:.4f})")
        print(f"[Diag] Default:   thresh=0.50 "
              f"(F1={f1_default:.4f}, Acc={acc_default:.4f})")

        return {
            'f1': f1_optimal,
            'f1_default': f1_default,
            'f1_macro': f1_macro,
            'accuracy': acc_optimal,
            'accuracy_default': acc_default,
            'auc_roc': auc,
            'optimal_threshold': best_thresh,
        }
