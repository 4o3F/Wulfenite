"""Equivalent rectangular bandwidth utilities."""

from __future__ import annotations

import math
from typing import overload

import torch


@overload
def freq2erb(freq_hz: float) -> float: ...


@overload
def freq2erb(freq_hz: torch.Tensor) -> torch.Tensor: ...


def freq2erb(freq_hz: float | torch.Tensor) -> float | torch.Tensor:
    """Convert frequency in Hz to the Glasberg-Moore ERB number."""
    scale = 24.7 * 9.265
    if isinstance(freq_hz, torch.Tensor):
        return 9.265 * torch.log1p(freq_hz / scale)
    return 9.265 * math.log1p(freq_hz / scale)


@overload
def erb2freq(n_erb: float) -> float: ...


@overload
def erb2freq(n_erb: torch.Tensor) -> torch.Tensor: ...


def erb2freq(n_erb: float | torch.Tensor) -> float | torch.Tensor:
    """Convert an ERB number back to frequency in Hz."""
    scale = 24.7 * 9.265
    if isinstance(n_erb, torch.Tensor):
        return scale * torch.expm1(n_erb / 9.265)
    return scale * math.expm1(n_erb / 9.265)


def erb_fb(
    n_freqs: int = 161,
    nb_bands: int = 24,
    sample_rate: int = 16000,
    min_nb_freqs: int = 2,
) -> torch.Tensor:
    """Build a triangular ERB filterbank matrix of shape ``[B, F]``."""
    if n_freqs <= 0:
        raise ValueError(f"n_freqs must be positive, got {n_freqs}")
    if nb_bands <= 0:
        raise ValueError(f"nb_bands must be positive, got {nb_bands}")
    if sample_rate <= 0:
        raise ValueError(f"sample_rate must be positive, got {sample_rate}")
    if min_nb_freqs <= 0:
        raise ValueError(f"min_nb_freqs must be positive, got {min_nb_freqs}")

    nyquist = sample_rate / 2.0
    erb_points = torch.linspace(
        float(freq2erb(0.0)),
        float(freq2erb(nyquist)),
        nb_bands + 2,
        dtype=torch.float32,
    )
    hz_points = erb2freq(erb_points)
    bin_positions = hz_points / nyquist * float(n_freqs - 1)
    freq_positions = torch.arange(n_freqs, dtype=torch.float32)
    fb = torch.zeros(nb_bands, n_freqs, dtype=torch.float32)

    for band in range(nb_bands):
        left = float(bin_positions[band].item())
        center = float(bin_positions[band + 1].item())
        right = float(bin_positions[band + 2].item())

        width = right - left
        if width < (min_nb_freqs - 1):
            scale = (min_nb_freqs - 1) / max(width, 1e-6)
            left = max(0.0, center - (center - left) * scale)
            right = min(float(n_freqs - 1), center + (right - center) * scale)

        if center <= left:
            center = min(float(n_freqs - 1), left + 0.5)
        if right <= center:
            right = min(float(n_freqs - 1), center + 0.5)

        rising = (freq_positions - left) / max(center - left, 1e-6)
        falling = (right - freq_positions) / max(right - center, 1e-6)
        tri = torch.minimum(rising, falling).clamp_min(0.0)
        if int((tri > 0).sum().item()) < min_nb_freqs:
            center_idx = int(round(center))
            start = max(0, center_idx - min_nb_freqs // 2)
            end = min(n_freqs, start + min_nb_freqs)
            start = max(0, end - min_nb_freqs)
            tri[start:end] = torch.linspace(0.5, 1.0, end - start, dtype=torch.float32)
        fb[band] = tri

    return fb / fb.sum(dim=1, keepdim=True).clamp_min(1e-8)


def erb_fb_inverse(filterbank: torch.Tensor) -> torch.Tensor:
    """Return a pseudo-inverse expansion matrix of shape ``[B, F]``."""
    if filterbank.dim() != 2:
        raise ValueError(
            f"filterbank must be 2-D [B, F], got {tuple(filterbank.shape)}"
        )
    return torch.linalg.pinv(filterbank).transpose(0, 1)


__all__ = ["freq2erb", "erb2freq", "erb_fb", "erb_fb_inverse"]
