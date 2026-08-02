"""
NSAC: Network-Slice-Aware Compression (Top-k + QSGD-style quantization).

Per-slice configurations (aligned with §4.2 + Figure 4):
  - uRLLC: k_ratio=0.1, b_bits=8  → 40× compression
  - eMBB:  k_ratio=0.3, b_bits=16 → ~7× compression
  - mMTC:  k_ratio=0.05, b_bits=8 → 80× compression

CR_s = (k_ratio * d) / d * (b_bits / 32) = k_ratio * b_bits / 32
"""

import torch
import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class SliceConfig:
    """Per-slice compression configuration."""
    name: str
    k_ratio: float       # fraction of parameters to keep (Top-k)
    b_bits: int          # quantization bit-width
    BW_mbps: float       # allocated bandwidth (Mbps)
    L_max_ms: float      # maximum latency constraint (ms)
    proc_ms: float       # processing delay (ms)

    @property
    def CR(self) -> float:
        """Compression ratio (compressed size / full size)."""
        return self.k_ratio * self.b_bits / 32.0

    @property
    def compression_factor(self) -> float:
        """Compression factor = 1 / CR."""
        return 1.0 / self.CR if self.CR > 0 else float('inf')


# Default slice configs (§4.2)
SLICE_CONFIGS = {
    'urllc': SliceConfig('uRLLC', k_ratio=0.1, b_bits=8,
                          BW_mbps=10, L_max_ms=200, proc_ms=12),
    'embb': SliceConfig('eMBB', k_ratio=0.3, b_bits=16,
                          BW_mbps=100, L_max_ms=500, proc_ms=8),
    'mmtc': SliceConfig('mMTC', k_ratio=0.05, b_bits=8,
                          BW_mbps=1, L_max_ms=1000, proc_ms=5),
}


def nsac_compress(
    gradient: torch.Tensor,
    slice_config: SliceConfig,
    model_size_bytes: int = 31304,  # |w| ≈ 30.6 KB (d=7826 × 4 bytes, lightweight)
    adaptive_down: bool = True,
    min_k_ratio: float = 0.01,
    k_step: float = 0.01,
) -> Dict:
    """
    NSAC compression: Top-k sparsification + QSGD-style quantization.

    Args:
        gradient: Full gradient vector (1D tensor of length d).
        slice_config: Per-slice configuration.
        model_size_bytes: Full model size in bytes (for latency estimation).
        adaptive_down: If True, reduce k when latency exceeds L_max.

    Returns:
        Dict with: indices, q_values, b_bits, k_ratio_actual, CR_actual,
                   latency_ms, compressed_bytes.
    """
    d = gradient.numel()
    k = int(slice_config.k_ratio * d)
    if k < 1:
        k = 1
    b = slice_config.b_bits
    levels = 2 ** (b - 1) - 1  # quantization levels

    # --- Adaptive k adjustment for latency ---
    k_ratio_actual = slice_config.k_ratio
    while adaptive_down and k_ratio_actual > min_k_ratio:
        CR_actual = k_ratio_actual * b / 32.0
        # Model size in bytes: d × 4 (float32)
        payload_bytes = int(CR_actual * d * 4)  # compressed payload
        BW_bytes_per_ms = slice_config.BW_mbps * 1e6 / 8 / 1000  # bytes/ms
        latency_est = payload_bytes / BW_bytes_per_ms + slice_config.proc_ms
        if latency_est <= slice_config.L_max_ms:
            break
        k_ratio_actual -= k_step
        k = int(k_ratio_actual * d)
        if k < 1:
            k = 1

    # --- Top-k sparsification ---
    abs_grad = gradient.abs()
    topk_values, topk_indices = torch.topk(abs_grad, k)
    sparse_grad = gradient[topk_indices]

    # --- QSGD-style quantization ---
    # q_val = sign(val) × floor(|val| × levels / ‖val‖) / levels
    # Normalize by the norm of selected values for unbiasedness
    norm = sparse_grad.norm()
    if norm < 1e-10:
        norm = torch.tensor(1.0)

    # Quantize: scale each value to levels, then snap to level boundaries
    scaled = sparse_grad * levels / norm
    quantized = torch.sign(scaled) * torch.floor(scaled.abs()) / levels
    # Un-normalize to original scale
    q_values = quantized * norm / levels

    # --- Pack result ---
    CR_actual = k_ratio_actual * b / 32.0
    # Actual compressed bytes: indices (4 bytes each) + values (b/8 bytes each)
    compressed_bytes = k * 4 + k * (b // 8)
    # More accurate latency estimation
    BW_bytes_per_ms = slice_config.BW_mbps * 1e6 / 8 / 1000
    latency_ms = compressed_bytes / BW_bytes_per_ms + slice_config.proc_ms

    return {
        'indices': topk_indices.cpu().numpy(),
        'q_values': q_values.cpu().numpy(),
        'b_bits': b,
        'k_ratio_actual': k_ratio_actual,
        'CR_actual': CR_actual,
        'compression_factor': 1.0 / CR_actual if CR_actual > 0 else float('inf'),
        'latency_ms': latency_ms,
        'compressed_bytes': compressed_bytes,
        'norm': norm.item(),  # for decompression
    }


def nsac_decompress(
    compressed: Dict,
    d: int,
) -> torch.Tensor:
    """
    Decompress NSAC-compressed gradient back to full d-dimensional vector.

    Reconstruction: Δw[idx] = q_val × ‖g‖ / sqrt(k), others = 0.
    (Unbiased reconstruction using norm preservation)
    """
    indices = compressed['indices']
    q_values = compressed['q_values']
    norm = compressed['norm']
    k = len(indices)

    # Reconstruct: place quantized values at selected indices, 0 elsewhere
    reconstructed = torch.zeros(d, dtype=torch.float32)
    # Scale factor for unbiasedness: norm / sqrt(k)
    scale = norm / np.sqrt(k) if k > 0 else 1.0
    reconstructed[indices] = torch.tensor(q_values) * scale

    return reconstructed


def compute_communication_stats(
    d: int,
    N: int,
    M: int,
    T: int,
    slice_assignments: Dict[int, str],
    model_size_bytes: int = 31304,  # |w| ≈ 30.6 KB (d=7826 × 4 bytes, lightweight)
) -> Dict:
    """
    Compute total communication statistics for one experiment run.

    Args:
        d: Model dimensionality.
        N: Number of devices.
        M: Number of gateways.
        T: Number of rounds.
        slice_assignments: gateway_id → slice_name mapping.
        model_size_bytes: Full model size.

    Returns:
        Dict with total_traffic_star, total_traffic_hierfed, speedup, etc.
    """
    star_traffic = N * model_size_bytes * T  # Eq.(3)

    # HierFed-Matter: M gateways, each sends compressed update per round
    total_compressed_bytes_per_round = 0
    for gw_id in range(M):
        slice_name = slice_assignments.get(gw_id, 'embb')  # default eMBB
        sc = SLICE_CONFIGS[slice_name]
        # Compressed payload per gateway per round
        CR = sc.k_ratio * sc.b_bits / 32.0
        payload = int(CR * d * 4)  # compressed gradient bytes
        total_compressed_bytes_per_round += payload

    core_traffic = total_compressed_bytes_per_round * T  # Eq.(5) with compression
    speedup = star_traffic / core_traffic if core_traffic > 0 else float('inf')

    # Edge traffic (local, doesn't traverse core)
    edge_traffic = N * model_size_bytes * T  # Eq.(4)

    return {
        'star_traffic_MB': star_traffic / 1e6,
        'core_traffic_MB': core_traffic / 1e6,
        'edge_traffic_MB': edge_traffic / 1e6,
        'speedup': speedup,
        'reduction_pct': (1 - core_traffic / star_traffic) * 100,
    }
