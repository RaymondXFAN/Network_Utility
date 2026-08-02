"""
Baseline methods for experimental comparison (Table 3):

1. FedAvg      - Standard star-topology FL, no DP, no compression
2. FedProx     - Proximal-regularized FL (μ parameter for non-IID)
3. HierFed     - Hierarchical FL (same topology as proposed, no DP/NSAC)
4. DP-FedAvg   - Star-topology FL with uniform DP noise (ε=3)
5. DP-Fed6G    - Star-topology FL with adaptive DP (ε=3)
6. Top-k+QSGD  - Star-topology FL with uniform Top-k compression
7. SAFEL-IoT   - Star-topology FL with adaptive DP + explainability
8. FedProx-DP  - FedProx + uniform DP (ε=3)
9. HierFed+DP  - Hierarchical FL + uniform DP (ε=3) — key baseline for isolating DPBA-FIM
"""

import os
import json

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List
from models.dnn import build_model
from core.hierfed import star_aggregate, edge_aggregate, core_aggregate
from core.nsac import SLICE_CONFIGS, nsac_compress, nsac_decompress
from core.dpba_fim import clip_gradient, inject_dp_noise


# ============================================================
# FedAvg (standard star-topology, no DP)
# ============================================================
def run_fedavg(config: dict) -> Dict:
    """FedAvg baseline: star-topology, no DP, no compression."""
    model = build_model(config.get('feature_dim', 80))
    d = model.num_parameters
    N = config.get('N', 50)
    T = config.get('T', 100)
    eta = config.get('eta', 0.01)
    E_local = config.get('E_local', 5)
    batch_size = config.get('batch_size', 32)

    # Load data
    X_train, y_train, partitions = _load_partitioned_data(config)
    X_test, y_test = _load_test_data(config)

    round_metrics = []
    for t in range(1, T + 1):
        device_gradients = {}
        for dev_id in range(N):
            idx_list = partitions.get(str(dev_id), [])
            if len(idx_list) == 0:
                device_gradients[dev_id] = torch.zeros(d)
                continue
            gradient = _local_train(model, X_train[idx_list], y_train[idx_list],
                                    E_local, batch_size, eta)
            device_gradients[dev_id] = gradient

        # Star aggregation
        global_params = _get_flat_params(model)
        new_params = star_aggregate(device_gradients, global_params, eta)
        _set_flat_params(model, new_params)

        metrics = _evaluate(model, X_test, y_test)
        round_metrics.append({'round': t, **metrics})

        if t % 10 == 0:
            print(f"[FedAvg] Round {t}: F1={metrics['f1']:.4f}")

    final = _evaluate(model, X_test, y_test)
    return {'final_metrics': final, 'round_metrics': round_metrics}


# ============================================================
# FedProx (proximal-regularized)
# ============================================================
def run_fedprox(config: dict, mu: float = 0.01) -> Dict:
    """FedProx: adds proximal term μ/2 × ‖w - w_global‖² to local loss."""
    model = build_model(config.get('feature_dim', 80))
    d = model.num_parameters
    N = config.get('N', 50)
    T = config.get('T', 100)
    eta = config.get('eta', 0.01)

    X_train, y_train, partitions = _load_partitioned_data(config)
    X_test, y_test = _load_test_data(config)

    round_metrics = []
    for t in range(1, T + 1):
        global_params = _get_flat_params(model)
        device_gradients = {}
        for dev_id in range(N):
            idx_list = partitions.get(str(dev_id), [])
            if len(idx_list) == 0:
                device_gradients[dev_id] = torch.zeros(d)
                continue
            gradient = _local_train_fedprox(
                model, X_train[idx_list], y_train[idx_list],
                global_params, mu, config
            )
            device_gradients[dev_id] = gradient

        new_params = star_aggregate(device_gradients, global_params, eta)
        _set_flat_params(model, new_params)

        metrics = _evaluate(model, X_test, y_test)
        round_metrics.append({'round': t, **metrics})

        if t % 10 == 0:
            print(f"[FedProx] Round {t}: F1={metrics['f1']:.4f}")

    final = _evaluate(model, X_test, y_test)
    return {'final_metrics': final, 'round_metrics': round_metrics}


def _local_train_fedprox(model, X, y, global_params, mu, config):
    """Local train with proximal regularization."""
    initial_params = _get_flat_params(model)
    E = config.get('E_local', 5)
    bs = config.get('batch_size', 32)
    eta = config.get('eta', 0.01)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=eta)
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.long)

    model.train()
    for epoch in range(E):
        indices = torch.randperm(len(y_t))
        for start in range(0, len(y_t), bs):
            batch_x = X_t[indices[start:start+bs]]
            batch_y = y_t[indices[start:start+bs]]
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            # Proximal term: μ/2 × ‖w - w_global‖²
            prox_loss = 0
            current_params = _get_flat_params(model)
            prox_loss = mu / 2 * torch.norm(current_params - global_params) ** 2
            total_loss = loss + prox_loss
            total_loss.backward()
            optimizer.step()

    final_params = _get_flat_params(model)
    return final_params - initial_params


# ============================================================
# DP-FedAvg (uniform DP, ε=3)
# ============================================================
def run_dpfedavg(config: dict) -> Dict:
    """DP-FedAvg: star-topology + uniform DP noise at ε=3."""
    model = build_model(config.get('feature_dim', 80))
    d = model.num_parameters
    N = config.get('N', 50)
    T = config.get('T', 100)
    eta = config.get('eta', 0.01)
    eps = config.get('epsilon_total', 3.0)
    delta = config.get('delta', 1e-5)
    C = config.get('clip_norm', 1.0)

    X_train, y_train, partitions = _load_partitioned_data(config)
    X_test, y_test = _load_test_data(config)

    eps_per_round = eps / T
    round_metrics = []

    for t in range(1, T + 1):
        device_gradients = {}
        for dev_id in range(N):
            idx_list = partitions.get(str(dev_id), [])
            if len(idx_list) == 0:
                device_gradients[dev_id] = torch.zeros(d)
                continue
            gradient = _local_train(model, X_train[idx_list], y_train[idx_list],
                                    config.get('E_local', 5),
                                    config.get('batch_size', 32), eta)
            # Uniform DP: clip + inject same noise to all devices
            clipped = clip_gradient(gradient, C)
            noisy = inject_dp_noise(clipped, eps_per_round, delta, C)
            device_gradients[dev_id] = noisy

        global_params = _get_flat_params(model)
        new_params = star_aggregate(device_gradients, global_params, eta)
        _set_flat_params(model, new_params)

        metrics = _evaluate(model, X_test, y_test)
        round_metrics.append({'round': t, **metrics})

        if t % 10 == 0:
            print(f"[DP-FedAvg] Round {t}: F1={metrics['f1']:.4f}")

    final = _evaluate(model, X_test, y_test)
    return {'final_metrics': final, 'round_metrics': round_metrics}


# ============================================================
# HierFed (hierarchical, no DP/NSAC)
# ============================================================
def run_hierfed(config: dict) -> Dict:
    """HierFed: hierarchical aggregation, no DP, no NSAC compression."""
    model = build_model(config.get('feature_dim', 80))
    d = model.num_parameters
    N = config.get('N', 50)
    M = config.get('M', 5)
    K = config.get('K', 10)
    T = config.get('T', 100)
    eta = config.get('eta', 0.01)

    X_train, y_train, partitions = _load_partitioned_data(config)
    X_test, y_test = _load_test_data(config)

    cluster_map = {gw: list(range(gw*K, min(gw*K+K, N))) for gw in range(M)}

    round_metrics = []
    for t in range(1, T + 1):
        device_gradients = {}
        for dev_id in range(N):
            idx_list = partitions.get(str(dev_id), [])
            if len(idx_list) == 0:
                device_gradients[dev_id] = torch.zeros(d)
                continue
            gradient = _local_train(model, X_train[idx_list], y_train[idx_list],
                                    config.get('E_local', 5),
                                    config.get('batch_size', 32), eta)
            device_gradients[dev_id] = gradient

        global_params = _get_flat_params(model)
        # Hierarchical aggregation without compression
        edge_updates = edge_aggregate(device_gradients, cluster_map,
                                       global_params, eta)
        new_params = core_aggregate(edge_updates, global_params)
        _set_flat_params(model, new_params)

        metrics = _evaluate(model, X_test, y_test)
        round_metrics.append({'round': t, **metrics})

        if t % 10 == 0:
            print(f"[HierFed] Round {t}: F1={metrics['f1']:.4f}")

    final = _evaluate(model, X_test, y_test)
    return {'final_metrics': final, 'round_metrics': round_metrics}


# ============================================================
# HierFed + uniform DP (ε=3) — key ablation baseline
# ============================================================
def run_hierfed_dp(config: dict) -> Dict:
    """HierFed + uniform DP: isolates DPBA-FIM's contribution."""
    model = build_model(config.get('feature_dim', 80))
    d = model.num_parameters
    N, M, K = config.get('N', 50), config.get('M', 5), config.get('K', 10)
    T = config.get('T', 100)
    eta = config.get('eta', 0.01)
    eps = config.get('epsilon_total', 3.0)
    delta = config.get('delta', 1e-5)
    C = config.get('clip_norm', 1.0)

    X_train, y_train, partitions = _load_partitioned_data(config)
    X_test, y_test = _load_test_data(config)

    cluster_map = {gw: list(range(gw*K, min(gw*K+K, N))) for gw in range(M)}
    eps_per_round = eps / T

    round_metrics = []
    for t in range(1, T + 1):
        device_gradients = {}
        for dev_id in range(N):
            idx_list = partitions.get(str(dev_id), [])
            if len(idx_list) == 0:
                device_gradients[dev_id] = torch.zeros(d)
                continue
            gradient = _local_train(model, X_train[idx_list], y_train[idx_list],
                                    config.get('E_local', 5),
                                    config.get('batch_size', 32), eta)
            # Uniform DP (same ε for all devices, no FIM weighting)
            clipped = clip_gradient(gradient, C)
            noisy = inject_dp_noise(clipped, eps_per_round, delta, C)
            device_gradients[dev_id] = noisy

        global_params = _get_flat_params(model)
        edge_updates = edge_aggregate(device_gradients, cluster_map,
                                       global_params, eta)
        new_params = core_aggregate(edge_updates, global_params)
        _set_flat_params(model, new_params)

        metrics = _evaluate(model, X_test, y_test)
        round_metrics.append({'round': t, **metrics})

        if t % 10 == 0:
            print(f"[HierFed+DP] Round {t}: F1={metrics['f1']:.4f}")

    final = _evaluate(model, X_test, y_test)
    return {'final_metrics': final, 'round_metrics': round_metrics}


# ============================================================
# Top-k + QSGD (star-topology, compression only)
# ============================================================
def run_topk_qsgd(config: dict) -> Dict:
    """Top-k + QSGD: star-topology FL with uniform Top-k compression."""
    model = build_model(config.get('feature_dim', 80))
    d = model.num_parameters
    N = config.get('N', 50)
    T = config.get('T', 100)
    eta = config.get('eta', 0.01)

    X_train, y_train, partitions = _load_partitioned_data(config)
    X_test, y_test = _load_test_data(config)

    # Use eMBB slice config as "uniform" compression (moderate)
    slice_config = SLICE_CONFIGS['embb']

    round_metrics = []
    for t in range(1, T + 1):
        device_gradients = {}
        for dev_id in range(N):
            idx_list = partitions.get(str(dev_id), [])
            if len(idx_list) == 0:
                device_gradients[dev_id] = torch.zeros(d)
                continue
            gradient = _local_train(model, X_train[idx_list], y_train[idx_list],
                                    config.get('E_local', 5),
                                    config.get('batch_size', 32), eta)
            # Compress each device's gradient
            compressed = nsac_compress(gradient, slice_config, adaptive_down=False)
            decompressed = nsac_decompress(compressed, d)
            device_gradients[dev_id] = decompressed

        global_params = _get_flat_params(model)
        new_params = star_aggregate(device_gradients, global_params, eta)
        _set_flat_params(model, new_params)

        metrics = _evaluate(model, X_test, y_test)
        round_metrics.append({'round': t, **metrics})

        if t % 10 == 0:
            print(f"[Top-k+QSGD] Round {t}: F1={metrics['f1']:.4f}")

    final = _evaluate(model, X_test, y_test)
    return {'final_metrics': final, 'round_metrics': round_metrics}


# ============================================================
# SAFEL-IoT (simplified: adaptive DP without FIM)
# ============================================================
def run_safeliot(config: dict) -> Dict:
    """SAFEL-IoT: adaptive DP (not FIM-weighted), star-topology."""
    # Simplified implementation: adaptive noise based on data sensitivity
    # but uniform across layers (no FIM weighting)
    model = build_model(config.get('feature_dim', 80))
    d = model.num_parameters
    N = config.get('N', 50)
    T = config.get('T', 100)
    eta = config.get('eta', 0.01)
    eps = config.get('epsilon_total', 3.0)
    delta = config.get('delta', 1e-5)
    C = config.get('clip_norm', 1.0)

    X_train, y_train, partitions = _load_partitioned_data(config)
    X_test, y_test = _load_test_data(config)

    # Adaptive: different ε per device based on data sensitivity
    sigma_map = {'camera': 1.0, 'thermostat': 0.3, 'bulb': 0.1}
    eps_per_round = eps / T

    round_metrics = []
    for t in range(1, T + 1):
        device_gradients = {}
        for dev_id in range(N):
            idx_list = partitions.get(str(dev_id), [])
            if len(idx_list) == 0:
                device_gradients[dev_id] = torch.zeros(d)
                continue
            gradient = _local_train(model, X_train[idx_list], y_train[idx_list],
                                    config.get('E_local', 5),
                                    config.get('batch_size', 32), eta)
            # Adaptive ε: scale by device data sensitivity
            n_i = len(idx_list)
            # Simple adaptive: more sensitive → more budget
            adaptive_eps = eps_per_round * (1.0 + 0.5 * min(n_i, 200) / 200)
            clipped = clip_gradient(gradient, C)
            noisy = inject_dp_noise(clipped, adaptive_eps, delta, C)
            device_gradients[dev_id] = noisy

        global_params = _get_flat_params(model)
        new_params = star_aggregate(device_gradients, global_params, eta)
        _set_flat_params(model, new_params)

        metrics = _evaluate(model, X_test, y_test)
        round_metrics.append({'round': t, **metrics})

        if t % 10 == 0:
            print(f"[SAFEL-IoT] Round {t}: F1={metrics['f1']:.4f}")

    final = _evaluate(model, X_test, y_test)
    return {'final_metrics': final, 'round_metrics': round_metrics}


# ============================================================
# DP-Fed6G (adaptive DP for 6G)
# ============================================================
def run_dpfed6g(config: dict) -> Dict:
    """DP-Fed6G: adaptive DP with convergence-aware budget, star-topology."""
    model = build_model(config.get('feature_dim', 80))
    d = model.num_parameters
    N = config.get('N', 50)
    T = config.get('T', 100)
    eta = config.get('eta', 0.01)
    eps = config.get('epsilon_total', 3.0)
    delta = config.get('delta', 1e-5)
    C = config.get('clip_norm', 1.0)

    X_train, y_train, partitions = _load_partitioned_data(config)
    X_test, y_test = _load_test_data(config)

    # Convergence-aware: increase ε over rounds (early rounds need less)
    round_metrics = []
    for t in range(1, T + 1):
        # Decay-based budget: ε_r(t) = (ε_total/T) × decay_factor
        decay_factor = min(1.0, t / (T * 0.3))  # ramp up over first 30% of rounds
        eps_round = (eps / T) * decay_factor

        device_gradients = {}
        for dev_id in range(N):
            idx_list = partitions.get(str(dev_id), [])
            if len(idx_list) == 0:
                device_gradients[dev_id] = torch.zeros(d)
                continue
            gradient = _local_train(model, X_train[idx_list], y_train[idx_list],
                                    config.get('E_local', 5),
                                    config.get('batch_size', 32), eta)
            clipped = clip_gradient(gradient, C)
            noisy = inject_dp_noise(clipped, eps_round, delta, C)
            device_gradients[dev_id] = noisy

        global_params = _get_flat_params(model)
        new_params = star_aggregate(device_gradients, global_params, eta)
        _set_flat_params(model, new_params)

        metrics = _evaluate(model, X_test, y_test)
        round_metrics.append({'round': t, **metrics})

        if t % 10 == 0:
            print(f"[DP-Fed6G] Round {t}: F1={metrics['f1']:.4f}")

    final = _evaluate(model, X_test, y_test)
    return {'final_metrics': final, 'round_metrics': round_metrics}


# ============================================================
# FedProx-DP (FedProx + uniform DP)
# ============================================================
def run_fedprox_dp(config: dict, mu: float = 0.01) -> Dict:
    """FedProx-DP: FedProx with uniform DP noise."""
    model = build_model(config.get('feature_dim', 80))
    d = model.num_parameters
    N = config.get('N', 50)
    T = config.get('T', 100)
    eta = config.get('eta', 0.01)
    eps = config.get('epsilon_total', 3.0)
    delta = config.get('delta', 1e-5)
    C = config.get('clip_norm', 1.0)

    X_train, y_train, partitions = _load_partitioned_data(config)
    X_test, y_test = _load_test_data(config)

    eps_per_round = eps / T
    round_metrics = []

    for t in range(1, T + 1):
        global_params = _get_flat_params(model)
        device_gradients = {}
        for dev_id in range(N):
            idx_list = partitions.get(str(dev_id), [])
            if len(idx_list) == 0:
                device_gradients[dev_id] = torch.zeros(d)
                continue
            gradient = _local_train_fedprox(model, X_train[idx_list],
                                            y_train[idx_list],
                                            global_params, mu, config)
            clipped = clip_gradient(gradient, C)
            noisy = inject_dp_noise(clipped, eps_per_round, delta, C)
            device_gradients[dev_id] = noisy

        new_params = star_aggregate(device_gradients, global_params, eta)
        _set_flat_params(model, new_params)

        metrics = _evaluate(model, X_test, y_test)
        round_metrics.append({'round': t, **metrics})

        if t % 10 == 0:
            print(f"[FedProx-DP] Round {t}: F1={metrics['f1']:.4f}")

    final = _evaluate(model, X_test, y_test)
    return {'final_metrics': final, 'round_metrics': round_metrics}


# ============================================================
# Helper functions
# ============================================================
def _get_flat_params(model):
    return torch.cat([p.data.clone().flatten() for p in model.parameters()])

def _set_flat_params(model, flat_params):
    idx = 0
    with torch.no_grad():
        for p in model.parameters():
            size = p.numel()
            p.data.copy_(flat_params[idx:idx+size].reshape(p.shape))
            idx += size

def _local_train(model, X, y, E_local, batch_size, lr):
    """Standard local training returning gradient (Δw)."""
    from core.simulator import local_train
    return local_train(model, X, y, E_local, batch_size, lr)

def _evaluate(model, X_test, y_test):
    """Quick evaluation on test set."""
    from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
    model.eval()
    X_t = torch.tensor(X_test, dtype=torch.float32)
    with torch.no_grad():
        outputs = model(X_t)
        probs = torch.softmax(outputs, dim=1).numpy()
        preds = outputs.argmax(dim=1).numpy()
    f1 = f1_score(y_test, preds, zero_division=0)
    acc = accuracy_score(y_test, preds)
    try:
        auc = roc_auc_score(y_test, probs[:, 1])
    except ValueError:
        auc = 0.0
    return {'f1': f1, 'accuracy': acc, 'auc_roc': auc}

def _load_partitioned_data(config):
    import json
    processed_dir = config.get('processed_dir', 'data/processed')
    ds = config.get('dataset_name', 'iotid20')
    prefix = 'iotid20' if ds == 'iotid20' else 'cicids'
    train = np.load(os.path.join(processed_dir, f'{prefix}_train.npz'))
    test = np.load(os.path.join(processed_dir, f'{prefix}_test.npz'))
    with open(os.path.join(processed_dir, f'{prefix}_partitions.json'), 'r', encoding='utf-8') as f:
        partitions = json.load(f)['device_indices']
    # Recompute partitions for current alpha/seed if needed
    alpha = config.get('alpha', 0.5)
    seed = config.get('seed', 1)
    return train['X'], train['y'], partitions

def _load_test_data(config):
    processed_dir = config.get('processed_dir', 'data/processed')
    ds = config.get('dataset_name', 'iotid20')
    prefix = 'iotid20' if ds == 'iotid20' else 'cicids'
    test = np.load(os.path.join(processed_dir, f'{prefix}_test.npz'))
    return test['X'], test['y']


# ============================================================
# Baseline runner dispatcher
# ============================================================
METHOD_MAP = {
    'fedavg': run_fedavg,
    'fedprox': run_fedprox,
    'dpfedavg': run_dpfedavg,
    'hierfed': run_hierfed,
    'hierfed_dp': run_hierfed_dp,
    'topk_qsgd': run_topk_qsgd,
    'safeliot': run_safeliot,
    'dpfed6g': run_dpfed6g,
    'fedprox_dp': run_fedprox_dp,
}

def run_baseline(method: str, config: dict) -> dict:
    """Run a baseline method by name."""
    if method not in METHOD_MAP:
        raise ValueError(f"Unknown baseline method: {method}. "
                         f"Available: {list(METHOD_MAP.keys())}")
    return METHOD_MAP[method](config)
