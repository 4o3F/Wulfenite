"""Penalty for suppressing target speech too aggressively."""

from __future__ import annotations

import torch
from torch import nn


def _as_complex(spec: torch.Tensor) -> torch.Tensor:
    if spec.is_complex():
        return spec
    if spec.dim() >= 1 and spec.size(-1) == 2:
        return torch.view_as_complex(spec.contiguous())
    raise ValueError(
        "spec must be a complex tensor or a real-valued tensor with "
        f"final dimension 2, got {tuple(spec.shape)}"
    )


class OverSuppressionLoss(nn.Module):
    """Penalize estimates whose magnitude falls below the clean target."""

    def __init__(self, gamma: float = 0.6) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, estimate: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        y = _as_complex(estimate)
        s = _as_complex(target)
        y_mag = y.abs().clamp_min(1e-8).pow(self.gamma)
        s_mag = s.abs().clamp_min(1e-8).pow(self.gamma)
        penalty = torch.relu(s_mag - y_mag)
        return penalty.pow(2).mean()


__all__ = ["OverSuppressionLoss"]
