"""
DPBA-FIM: Dynamic Privacy Budget Allocation via Fisher Information Matrix.

Core equations (§4.3):
  - FIM scalar: I_l(t) = β_F · I_l(t-1) + (1-β_F) · tr(FIM_l(t))   ... Eq.(11)
  - Device sensitivity: S_i = α_s·σ_i + β_s·(1/n_i) + γ_s·H_i        ... Eq.(12)
  - Budget allocation: ε_{i,l}(t) = (ε_total/T) · (I_l/ΣI_l') · (S_i/ΣS_j) ... Eq.(13)
  - Gradient clip: g̃ = g / max(1, ‖g‖/C)                              ... Eq.(14)
  - Noise inject: ĝ = g̃ + N(0, σ_noise² · I)                         ... Eq.(15)
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from models.dnn import AnomalyDNN


# ============================================================
# Fisher Information Computation
# ============================================================
class FIMTracker:
    """Tracks per-layer Fisher information scores with EMA smoothing."""

    def __init__(self, num_layers: int, beta_F: float = 0.9):
        self.num_layers = num_layers
        self.beta_F = beta_F
        # EMA-smoothed scores, initialized to 1.0 (uniform)
        self.I_scores = np.ones(num_layers, dtype=np.float64)

    def update(self, model: AnomalyDNN, local_data: torch.Tensor,
               local_labels: torch.Tensor, layer_param_ids: List[int]):
        """
        Compute per-layer FIM trace from local data and update EMA scores.

        Args:
            model: Current global model.
            local_data: Local dataset features.
            local_labels: Local dataset labels.
            layer_param_ids: Indices mapping each FIM score to model parameters.
                             Each Linear layer has 2 param groups (weight, bias),
                             so num_layers = num_linear_layers * 2 or
                             num_linear_layers (if tracking per-layer, not per-param).
        """
        model.eval()
        n = local_data.shape[0]
        if n == 0:
            return

        # Compute per-sample gradients for each layer
        # For efficiency, we use the loss gradient and compute trace approximation
        criterion = torch.nn.CrossEntropyLoss(reduction='none')

        # Zero gradients
        model.zero_grad()

        # Forward pass with per-sample loss
        outputs = model(local_data)
        losses = criterion(outputs, local_labels)

        # Compute per-layer gradient traces using batch gradient
        # Full per-sample gradient is expensive; use diagonal approximation
        # trace(FIM_l) ≈ Σ_i ‖∂L_i/∂w_l‖² / n_i
        layer_traces = []

        # Get layer modules
        linear_layers = [m for m in model.net if isinstance(m, torch.nn.Linear)]

        for layer in linear_layers:
            # Compute gradient for this layer's weight
            model.zero_grad()
            # Per-sample gradient approximation: use the mean gradient scaled
            mean_loss = losses.mean()
            mean_loss.backward(retain_graph=True)

            # Compute trace from weight gradient: trace ≈ ‖∇_w‖²
            if layer.weight.grad is not None:
                weight_trace = (layer.weight.grad ** 2).sum().item()
            else:
                weight_trace = 0.0

            # Also include bias gradient
            if layer.bias is not None and layer.bias.grad is not None:
                bias_trace = (layer.bias.grad ** 2).sum().item()
            else:
                bias_trace = 0.0

            layer_traces.append(weight_trace + bias_trace)

        model.zero_grad()

        # EMA update: I_l(t) = β_F · I_l(t-1) + (1-β_F) · trace_l(t)
        traces = np.array(layer_traces, dtype=np.float64)
        # Normalize traces by sample count for FIM definition
        traces = traces / n
        self.I_scores = self.beta_F * self.I_scores + (1 - self.beta_F) * traces

    def get_weights(self) -> np.ndarray:
        """Return normalized FIM weights: I_l / ΣI_l."""
        total = self.I_scores.sum()
        if total < 1e-10:
            return np.ones(self.num_layers) / self.num_layers
        return self.I_scores / total


# ============================================================
# Device Sensitivity Score
# ============================================================
def compute_device_sensitivity(
    device_types: List[str],
    local_sample_counts: List[int],
    heterogeneity_scores: List[float],
    alpha_s: float = 0.5,
    beta_s: float = 0.3,
    gamma_s: float = 0.2,
) -> np.ndarray:
    """
    Compute device sensitivity scores S_i (Eq.12).

    S_i = α_s · σ_i + β_s · (1/n_i) + γ_s · H_i

    Args:
        device_types: List of device type names ('camera', 'thermostat', 'bulb').
        local_sample_counts: n_i for each device.
        heterogeneity_scores: H_i for each device (from Dirichlet partition).
    """
    sigma_map = {'camera': 1.0, 'thermostat': 0.3, 'bulb': 0.1}

    S = np.zeros(len(device_types))
    for i, (dtype, n_i, H_i) in enumerate(
        zip(device_types, local_sample_counts, heterogeneity_scores)
    ):
        sigma_i = sigma_map.get(dtype, 0.1)  # default = bulb
        S[i] = alpha_s * sigma_i + beta_s * (1.0 / max(n_i, 1)) + gamma_s * H_i

    return S


def compute_heterogeneity_from_partition(
    labels: np.ndarray,
    device_indices: Dict[int, List[int]],
) -> List[float]:
    """
    Compute per-device heterogeneity H_i based on local label entropy.
    """
    H = []
    for dev_id, idx_list in device_indices.items():
        if len(idx_list) == 0:
            H.append(0.0)
            continue
        local_labels = labels[idx_list]
        # Class proportions
        classes = np.unique(local_labels)
        props = []
        for c in classes:
            p = np.sum(local_labels == c) / len(local_labels)
            props.append(p)
        # Shannon entropy as heterogeneity measure
        entropy = -sum(p * np.log2(p + 1e-10) for p in props)
        H.append(entropy)
    return H


# ============================================================
# Privacy Budget Allocation
# ============================================================
def allocate_privacy_budget(
    epsilon_total: float,
    T: int,
    fim_weights: np.ndarray,
    device_sensitivity: np.ndarray,
    num_devices: int,
    num_layers: int,
) -> np.ndarray:
    """
    Per-device, per-layer, per-round privacy budget (Eq.13).

    ε_{i,l}(t) = (ε_total / T) × (I_l / ΣI_l') × (S_i / ΣS_j)

    Returns:
        2D array [device, layer] of privacy budgets.
    """
    epsilon_per_round = epsilon_total / T  # fixed per-round budget

    # FIM layer weights (already normalized: sum = 1)
    fim_w = fim_weights  # shape: [num_layers]

    # Device sensitivity weights (normalize: sum = 1)
    S_w = device_sensitivity / device_sensitivity.sum()

    # Allocate: ε_{i,l} = ε_r × I_l × S_i
    budgets = np.outer(S_w, fim_w) * epsilon_per_round

    return budgets  # shape: [num_devices, num_layers]


# ============================================================
# DP Noise Injection
# ============================================================
def clip_gradient(gradient: torch.Tensor, C: float = 1.0) -> torch.Tensor:
    """Gradient clipping to norm C (Eq.14)."""
    norm = gradient.norm()
    if norm > C:
        gradient = gradient * (C / norm)
    return gradient


def compute_noise_scale(
    epsilon: float,
    delta: float,
    C: float = 1.0,
) -> float:
    """
    Compute Gaussian noise standard deviation for (ε, δ)-DP.

    σ_noise = C × sqrt(2 × ln(1.25/δ)) / ε
    """
    return C * np.sqrt(2 * np.log(1.25 / delta)) / epsilon


def inject_dp_noise(
    gradient: torch.Tensor,
    epsilon: float,
    delta: float = 1e-5,
    C: float = 1.0,
) -> torch.Tensor:
    """
    Apply DP noise injection to clipped gradient (Eq.15).

    ĝ = g̃ + N(0, σ_noise² × I)
    """
    sigma = compute_noise_scale(epsilon, delta, C)
    noise = torch.randn_like(gradient) * sigma
    return gradient + noise


def dpba_fim_gradient_processing(
    gradient: torch.Tensor,
    epsilon: float,
    delta: float = 1e-5,
    C: float = 1.0,
) -> torch.Tensor:
    """
    Full DPBA-FIM processing pipeline for a device's gradient:
    clip → inject noise with per-device, per-layer budget.

    For simplicity, this applies uniform noise scale per device
    (using the device-level ε budget). Per-layer variation can be
    added by splitting the gradient into per-layer segments.
    """
    clipped = clip_gradient(gradient, C)
    noisy = inject_dp_noise(clipped, epsilon, delta, C)
    return noisy
