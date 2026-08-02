"""
4-Layer DNN anomaly detection model for HierFed-Matter-NSAC-DPBA.

Lightweight architecture (d=7,826 with bias for Input=79):
  Input(dim) → FC(64, ReLU) → FC(32, ReLU) → FC(16, ReLU) → Output(2)
"""

import torch
import torch.nn as nn
from typing import List


class AnomalyDNN(nn.Module):
    """4-layer DNN for binary anomaly detection (normal vs. attack).
    
    Lightweight architecture (d=7,826 with bias for Input=79):
      Input(dim) → FC(64, ReLU) → FC(32, ReLU) → FC(16, ReLU) → Output(2)
    """

    def __init__(
        self,
        input_dim: int = 79,
        hidden_dims: List[int] = [64, 32, 16],
        output_dim: int = 2,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim

        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            prev_dim = h
        layers.append(nn.Linear(prev_dim, output_dim))
        # Softmax is applied in the loss (CrossEntropy includes log-softmax)
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    @property
    def num_parameters(self) -> int:
        """Total trainable parameter count (d)."""
        return sum(p.numel() for p in self.parameters())

    def get_layer_parameters(self) -> List[nn.Parameter]:
        """Return parameters grouped by layer (for per-layer FIM and DP)."""
        # Each Linear layer has weight + bias → 2 parameter tensors
        params = []
        for module in self.net:
            if isinstance(module, nn.Linear):
                for p in module.parameters():
                    params.append(p)
        return params

    def get_layer_names(self) -> List[str]:
        """Return descriptive names for each parameter layer."""
        names = []
        for name, module in self.net.named_modules():
            if isinstance(module, nn.Linear):
                names.append(f"{name}.weight")
                names.append(f"{name}.bias")
        return names


def build_model(input_dim: int = 79, hidden_dims: List[int] = [64, 32, 16],
                output_dim: int = 2) -> AnomalyDNN:
    """Factory function to build model with given input dimension."""
    model = AnomalyDNN(input_dim=input_dim, hidden_dims=hidden_dims,
                       output_dim=output_dim)
    print(f"[Model] input_dim={input_dim}, hidden_dims={hidden_dims}, "
          f"d={model.num_parameters}")
    return model


if __name__ == "__main__":
    # Quick sanity check
    m80 = build_model(80)
    m76 = build_model(76)
    x = torch.randn(4, 80)
    print(f"Output shape: {m80(x).shape}")  # (4, 2)
    print(f"Layer names: {m80.get_layer_names()}")
