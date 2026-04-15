"""Lightweight multi-resolution magnitude loss."""

from __future__ import annotations

from typing import cast

import torch
import torch.nn.functional as F
from torch import nn


class MultiResolutionLoss(nn.Module):
    """Average magnitude-domain MSE over several STFT resolutions."""

    def __init__(
        self,
        windows: tuple[int, ...] = (80, 160, 320, 640),
        gamma: float = 0.6,
    ) -> None:
        super().__init__()
        self.windows = windows
        self.gamma = gamma
        for window in windows:
            self.register_buffer(
                f"window_{window}",
                torch.hann_window(window),
                persistent=False,
            )

    def _window(self, length: int) -> torch.Tensor:
        return cast(torch.Tensor, getattr(self, f"window_{length}"))

    def forward(self, estimate: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if estimate.shape != target.shape:
            raise ValueError(
                f"estimate {tuple(estimate.shape)} vs target {tuple(target.shape)}"
            )
        if estimate.dim() == 1:
            estimate = estimate.unsqueeze(0)
            target = target.unsqueeze(0)
        if estimate.dim() != 2:
            raise ValueError(
                f"estimate/target must be [B, T], got {tuple(estimate.shape)}"
            )

        total = torch.zeros((), device=estimate.device, dtype=estimate.dtype)
        for window_length in self.windows:
            window = self._window(window_length).to(device=estimate.device, dtype=estimate.dtype)
            est_spec = torch.stft(
                estimate,
                n_fft=window_length,
                hop_length=window_length // 2,
                win_length=window_length,
                window=window,
                center=True,
                return_complex=True,
            )
            tgt_spec = torch.stft(
                target,
                n_fft=window_length,
                hop_length=window_length // 2,
                win_length=window_length,
                window=window,
                center=True,
                return_complex=True,
            )
            est_mag = est_spec.abs().clamp_min(1e-8).pow(self.gamma)
            tgt_mag = tgt_spec.abs().clamp_min(1e-8).pow(self.gamma)
            total = total + F.mse_loss(est_mag, tgt_mag)
        return total / len(self.windows)


__all__ = ["MultiResolutionLoss"]
