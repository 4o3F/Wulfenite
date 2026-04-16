"""Spectral-domain losses for pDFNet2 training."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from .multi_res import MultiResolutionLoss
from .over_suppression import OverSuppressionLoss


def _as_complex(spec: torch.Tensor) -> torch.Tensor:
    if spec.is_complex():
        return spec
    if spec.dim() >= 1 and spec.size(-1) == 2:
        return torch.view_as_complex(spec.contiguous())
    raise ValueError(
        "spec must be a complex tensor or a real-valued tensor with "
        f"final dimension 2, got {tuple(spec.shape)}"
    )


class SpectralLoss(nn.Module):
    """Magnitude + complex spectral loss used by pDFNet2."""

    def __init__(
        self,
        gamma: float = 1.0,
        under_suppression_weight: float = 1.0,
        complex_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.under_suppression_weight = under_suppression_weight
        self.complex_weight = complex_weight

    def forward(self, estimate: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        y = _as_complex(estimate)
        s = _as_complex(target)
        y_mag = y.abs().clamp_min(1e-8).pow(self.gamma)
        s_mag = s.abs().clamp_min(1e-8).pow(self.gamma)
        mag_err = (y_mag - s_mag).pow(2)
        mag_err = mag_err * torch.where(
            y_mag < s_mag,
            self.under_suppression_weight,
            1.0,
        )
        mag_loss = mag_err.mean()
        complex_loss = (
            F.mse_loss(y.real, s.real) + F.mse_loss(y.imag, s.imag)
        ) * self.complex_weight
        return mag_loss + complex_loss


class PDfNet2Loss(nn.Module):
    """Combined pDFNet2 training loss."""

    def __init__(
        self,
        *,
        lambda_spec: float = 1e3,
        lambda_mr: float = 5e2,
        lambda_os: float = 5e2,
        gamma: float = 1.0,
        fft_size: int = 320,
        hop_size: int = 160,
        win_size: int = 320,
    ) -> None:
        super().__init__()
        self.lambda_spec = lambda_spec
        self.lambda_mr = lambda_mr
        self.lambda_os = lambda_os
        self.spectral = SpectralLoss(gamma=gamma)
        self.multi_res = MultiResolutionLoss(gamma=gamma)
        self.over_suppression = OverSuppressionLoss(gamma=gamma)
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.win_size = win_size
        self.window: torch.Tensor
        self.register_buffer("window", torch.hann_window(win_size), persistent=False)

    def _stft(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        if waveform.dim() != 2:
            raise ValueError(
                f"waveform must be [T] or [B, T], got {tuple(waveform.shape)}"
            )
        length = waveform.size(-1)
        if length <= self.fft_size:
            padded_length = self.fft_size
        else:
            steps = math.ceil((length - self.fft_size) / self.hop_size)
            padded_length = self.fft_size + max(0, steps) * self.hop_size
        padded = F.pad(waveform, (0, padded_length - length))
        spec = torch.stft(
            padded,
            n_fft=self.fft_size,
            hop_length=self.hop_size,
            win_length=self.win_size,
            window=self.window.to(device=waveform.device, dtype=waveform.dtype),
            center=False,
            return_complex=True,
        ).transpose(1, 2)
        return spec

    def forward(
        self,
        estimate_waveform: torch.Tensor,
        target_waveform: torch.Tensor,
        *,
        estimate_spec: torch.Tensor | None = None,
        target_spec: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if estimate_waveform.shape != target_waveform.shape:
            raise ValueError(
                f"estimate_waveform {tuple(estimate_waveform.shape)} vs "
                f"target_waveform {tuple(target_waveform.shape)}"
            )
        if estimate_spec is None:
            estimate_spec = self._stft(estimate_waveform)
        if target_spec is None:
            target_spec = self._stft(target_waveform)
        spec_loss = self.spectral(estimate_spec, target_spec)
        mr_loss = self.multi_res(estimate_waveform, target_waveform)
        os_loss = self.over_suppression(estimate_spec, target_spec)
        total = (
            self.lambda_spec * spec_loss
            + self.lambda_mr * mr_loss
            + self.lambda_os * os_loss
        )
        return total, {
            "spectral": spec_loss,
            "multi_res": mr_loss,
            "over_suppression": os_loss,
        }


__all__ = ["SpectralLoss", "PDfNet2Loss"]
