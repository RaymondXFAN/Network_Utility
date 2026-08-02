"""
HierFed-Matter: Hierarchical Aggregation via Matter Topology.

Two-phase aggregation protocol (§4.1):
  Edge Phase:   w_m^t = w^t - η · (1/K_m) Σ_{i∈D_m} g_i^t   ... Eq.(1)
  Core Phase:   w^{t+1} = w^t + Σ_m(K_m·Δw_m^t) / Σ_m K_m   ... Eq.(2)

Communication complexity:
  Star: C_star = N × |w| × T                                ... Eq.(3)
  HierFed core: C_core = M × |w_e| × T                      ... Eq.(5)
  Reduction ratio: M/N                                       ... Eq.(6)
"""

import torch
import numpy as np
from typing import Dict, List, Tuple
from core.nsac import nsac_compress, nsac_decompress, SliceConfig, SLICE_CONFIGS


# ============================================================
# Hierarchical Aggregation
# ============================================================
def edge_aggregate(
    device_gradients: Dict[int, torch.Tensor],  # device_id → gradient
    cluster_map: Dict[int, List[int]],           # gateway_id → [device_ids]
    global_model_params: torch.Tensor,
    learning_rate: float = 0.01,
) -> Dict[int, Tuple[torch.Tensor, int]]:
    """
    Edge Phase: Aggregate gradients within each gateway's cluster.

    For gateway m with devices D_m = cluster_map[m]:
      w_m^t = w^t - η · (1/K_m) Σ_{i∈D_m} g_i^t

    Returns:
        Dict {gateway_id: (edge_update Δw_m, cluster_size K_m)}
    """
    edge_updates = {}
    for gw_id, device_ids in cluster_map.items():
        K_m = len(device_ids)
        if K_m == 0:
            continue
        # Sum device gradients
        avg_grad = torch.zeros_like(global_model_params)
        for dev_id in device_ids:
            if dev_id in device_gradients:
                avg_grad += device_gradients[dev_id]
        avg_grad /= K_m

        # Edge update: local_train 返回的 gradient 已是 Δw = -η·avg_grad（含 η），
        # 故这里直接取均值作为 Δw_m，不要再乘 learning_rate（否则方向反且变成 η²）。
        delta_w_m = avg_grad
        edge_updates[gw_id] = (delta_w_m, K_m)

    return edge_updates


def core_aggregate(
    edge_updates: Dict[int, Tuple[torch.Tensor, int]],
    global_model_params: torch.Tensor,
) -> torch.Tensor:
    """
    Core Phase: Global aggregation across all gateways.

    w^{t+1} = w^t + Σ_m(K_m · Δw_m^t) / Σ_m K_m   ... Eq.(2)

    Args:
        edge_updates: Dict {gateway_id: (Δw_m, K_m)}
        global_model_params: Current global model w^t (flat 1D tensor).

    Returns:
        Updated global model w^{t+1} (flat 1D tensor).
    """
    total_K = 0
    weighted_sum = torch.zeros_like(global_model_params)
    for gw_id, (delta_w_m, K_m) in edge_updates.items():
        weighted_sum += K_m * delta_w_m
        total_K += K_m

    if total_K == 0:
        return global_model_params

    w_next = global_model_params + weighted_sum / total_K
    return w_next


# ============================================================
# HierFed-Matter Full Round
# ============================================================
def hierfed_matter_round(
    device_gradients: Dict[int, torch.Tensor],
    cluster_map: Dict[int, List[int]],
    slice_assignments: Dict[int, str],
    global_model_params: torch.Tensor,
    learning_rate: float = 0.01,
    d: int = 7826,
    enable_nsac: bool = True,
    model_size_bytes: int = 31304,  # |w| ≈ 30.6 KB (d=7826 × 4 bytes, lightweight)
) -> Tuple[torch.Tensor, Dict]:
    """
    One full HierFed-Matter round: Edge aggregate → NSAC compress → Core aggregate.

    Returns:
        (updated_global_params, round_stats_dict)
    """
    # --- Edge Phase ---
    edge_updates = edge_aggregate(
        device_gradients, cluster_map, global_model_params, learning_rate
    )

    # --- NSAC Compression (per gateway) ---
    compressed_updates = {}
    round_stats = {
        'compression_ratios': [],
        'latencies_ms': [],
        'total_compressed_bytes': 0,
    }

    for gw_id, (delta_w_m, K_m) in edge_updates.items():
        if enable_nsac:
            slice_name = slice_assignments.get(gw_id, 'embb')
            slice_config = SLICE_CONFIGS[slice_name]
            compressed = nsac_compress(delta_w_m, slice_config, model_size_bytes)
            # Decompress at cloud side
            decompressed = nsac_decompress(compressed, d)
            compressed_updates[gw_id] = (decompressed, K_m)
            round_stats['compression_ratios'].append(compressed['CR_actual'])
            round_stats['latencies_ms'].append(compressed['latency_ms'])
            round_stats['total_compressed_bytes'] += compressed['compressed_bytes']
        else:
            # No compression: send full gradient
            compressed_updates[gw_id] = (delta_w_m, K_m)
            round_stats['compression_ratios'].append(1.0)  # no compression
            round_stats['latencies_ms'].append(0)
            round_stats['total_compressed_bytes'] += delta_w_m.numel() * 4

    # --- Core Phase ---
    w_next = core_aggregate(compressed_updates, global_model_params)

    return w_next, round_stats


# ============================================================
# Star-topology aggregation (FedAvg baseline)
# ============================================================
def star_aggregate(
    device_gradients: Dict[int, torch.Tensor],
    global_model_params: torch.Tensor,
    learning_rate: float = 0.01,
) -> torch.Tensor:
    """
    Standard FedAvg star-topology aggregation:
      w^{t+1} = w^t - η · (1/N) Σ_i g_i^t
    """
    N = len(device_gradients)
    if N == 0:
        return global_model_params
    avg_grad = torch.stack(list(device_gradients.values())).mean(dim=0)
    # device_gradients 已是 Δw（含 η）；直接加权求和得到全局更新方向。
    w_next = global_model_params + avg_grad
    return w_next
