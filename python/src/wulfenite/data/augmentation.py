"""Acoustic augmentation helpers for the on-the-fly mixer.

Two building blocks:

- **Synthetic room impulse response (RIR)** — a cheap
  exponentially-decaying tail plus a handful of sparse early
  reflections, good enough to bridge the clean-studio → small-room
  domain gap without requiring a real RIR dataset.
- **Additive Gaussian noise at a target SNR** — for mild room-tone
  simulation on top of the pair-mixed speech.

The RIR convolution uses FFT-based convolution (``O(N log N)``)
instead of direct 1-D conv. For typical kernel sizes ~5-10 k samples
this is 50-100× faster on CPU and was the dominant per-sample cost
in v1's data loader before we fixed it (commit ``392c731`` on v1).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import torch


# ---------------------------------------------------------------------------
# RIR synthesis
# ---------------------------------------------------------------------------


@dataclass
class ReverbConfig:
    """Parameters for :func:`synth_room_rir`.

    Defaults target small rooms (RT60 0.08 – 0.25 s), which matches
    typical streamer / gamer setups. Larger rooms can be simulated by
    widening ``rt60_range``.
    """

    sample_rate: int = 16000
    rt60_range: tuple[float, float] = (0.08, 0.25)
    num_early_reflections_range: tuple[int, int] = (3, 6)
    early_delay_range_ms: tuple[float, float] = (5.0, 35.0)
    early_amplitude_range: tuple[float, float] = (0.25, 0.70)
    diffuse_tail_scale: float = 0.10


def synth_room_rir(
    cfg: ReverbConfig,
    rng: random.Random,
) -> torch.Tensor:
    """Synthesize one small-room RIR.

    Structure:
        direct_path (1.0 at t=0)
      + 3-6 signed early reflections, random delay in 5-35 ms
      + exponentially-decaying diffuse noise tail over 2·RT60 seconds

    Peak-normalized so convolution does not change overall gain
    beyond the direct path.
    """
    rt60 = rng.uniform(*cfg.rt60_range)
    length = int(max(0.05, rt60 * 2.0) * cfg.sample_rate)
    rir = torch.zeros(length)

    # Direct path
    rir[0] = 1.0

    # Early reflections
    n_early = rng.randint(*cfg.num_early_reflections_range)
    for _ in range(n_early):
        delay_ms = rng.uniform(*cfg.early_delay_range_ms)
        sample_idx = int(delay_ms * cfg.sample_rate / 1000.0)
        if sample_idx >= length:
            continue
        amp = rng.uniform(*cfg.early_amplitude_range) * rng.choice([-1.0, 1.0])
        rir[sample_idx] += amp

    # Diffuse exponential-decay tail.
    # tau chosen so the tail drops by ~60 dB over rt60 seconds:
    #   exp(-rt60 / tau) = 10^-3   =>   tau = rt60 / (3 · ln10)
    tau_samples = max(1.0, rt60 * cfg.sample_rate / (3.0 * math.log(10.0)))
    t = torch.arange(length, dtype=torch.float32)
    envelope = torch.exp(-t / tau_samples)
    rir = rir + torch.randn(length) * envelope * cfg.diffuse_tail_scale

    peak = float(rir.abs().max())
    if peak > 0:
        rir = rir / peak
    return rir


# ---------------------------------------------------------------------------
# Convolution
# ---------------------------------------------------------------------------


def apply_rir(signal: torch.Tensor, rir: torch.Tensor) -> torch.Tensor:
    """Convolve a 1D signal with a 1D RIR, return the same length.

    FFT-based (O(N log N)) rather than direct 1D conv, which for
    typical RIR kernels in the 5-10 k sample range is roughly
    50-100× faster on CPU than ``torch.nn.functional.conv1d``.

    The output is head-aligned with the input (length unchanged) —
    the reverb tail that falls off the end is dropped.
    """
    n_signal = signal.shape[-1]
    n_kernel = rir.shape[-1]
    n_full = n_signal + n_kernel - 1
    n_fft = 1 << (n_full - 1).bit_length()  # next power of 2

    S = torch.fft.rfft(signal, n=n_fft)
    K = torch.fft.rfft(rir, n=n_fft)
    y = torch.fft.irfft(S * K, n=n_fft)
    return y[:n_signal]


# ---------------------------------------------------------------------------
# Additive noise
# ---------------------------------------------------------------------------


def add_noise_at_snr(
    signal: torch.Tensor,
    noise: torch.Tensor,
    snr_db: float,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Add pre-sampled noise to ``signal`` at the specified SNR.

    The noise is scaled so that:

    .. math::

        \\mathrm{SNR} = 10 \\log_{10}
            \\frac{\\text{signal power}}{\\text{noise power}}

    matches the requested ``snr_db``. If ``noise`` is shorter than
    ``signal`` it is looped; if longer, a random window is taken.
    """
    n = signal.shape[-1]
    if noise.shape[-1] < n:
        reps = (n + noise.shape[-1] - 1) // noise.shape[-1]
        noise = noise.repeat(reps)[:n]
    elif noise.shape[-1] > n:
        start = torch.randint(0, noise.shape[-1] - n + 1, (1,)).item()
        noise = noise[start:start + n]

    sig_rms = float(torch.sqrt((signal * signal).mean() + eps))
    noise_rms = float(torch.sqrt((noise * noise).mean() + eps))
    if sig_rms < eps or noise_rms < eps:
        return signal

    target_noise_rms = sig_rms / (10.0 ** (snr_db / 20.0))
    noise = noise * (target_noise_rms / noise_rms)
    return signal + noise


def add_gaussian_noise(
    signal: torch.Tensor,
    snr_db: float,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Add fresh white noise at the given SNR.

    Convenience wrapper around :func:`add_noise_at_snr` for the case
    where no external noise dataset is available.
    """
    noise = torch.randn(
        signal.shape, generator=generator, device=signal.device, dtype=signal.dtype,
    )
    return add_noise_at_snr(signal, noise, snr_db)
