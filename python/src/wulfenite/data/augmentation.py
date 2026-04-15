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
from typing import Literal

import torch
import torch.nn.functional as F


RoomPreset = Literal["small", "medium", "large", "mixed"]


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
    rt60_range: tuple[float, float] = (0.08, 0.80)
    num_early_reflections_range: tuple[int, int] = (3, 12)
    early_delay_range_ms: tuple[float, float] = (5.0, 80.0)
    early_amplitude_range: tuple[float, float] = (0.25, 0.70)
    diffuse_tail_scale: float = 0.10

    @classmethod
    def from_preset(
        cls,
        preset: RoomPreset,
        sample_rate: int = 16000,
    ) -> "ReverbConfig":
        if preset == "small":
            return cls(
                sample_rate=sample_rate,
                rt60_range=(0.08, 0.20),
                num_early_reflections_range=(3, 6),
                early_delay_range_ms=(5.0, 35.0),
            )
        if preset == "medium":
            return cls(
                sample_rate=sample_rate,
                rt60_range=(0.20, 0.40),
                num_early_reflections_range=(5, 9),
                early_delay_range_ms=(10.0, 55.0),
            )
        if preset == "large":
            return cls(
                sample_rate=sample_rate,
                rt60_range=(0.40, 0.65),
                num_early_reflections_range=(8, 14),
                early_delay_range_ms=(15.0, 80.0),
            )
        if preset == "mixed":
            return cls(sample_rate=sample_rate)
        raise ValueError(f"Unsupported room preset: {preset}")


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
    tail_gen = torch.Generator()
    tail_gen.manual_seed(rng.getrandbits(63))
    rir = rir + torch.randn(length, generator=tail_gen) * envelope * cfg.diffuse_tail_scale

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


def _fit_noise_length(
    noise: torch.Tensor,
    length: int,
    rng: random.Random | None,
) -> torch.Tensor:
    if noise.dim() != 1:
        raise ValueError(f"noise must be 1-D, got {tuple(noise.shape)}")
    if length < 0:
        raise ValueError(f"length must be >= 0, got {length}")
    if noise.shape[-1] == length:
        return noise
    if noise.shape[-1] == 0:
        return torch.zeros(length, device=noise.device, dtype=noise.dtype)
    if noise.shape[-1] < length:
        reps = (length + noise.shape[-1] - 1) // noise.shape[-1]
        return noise.repeat(reps)[:length]
    if rng is not None:
        start = rng.randint(0, noise.shape[-1] - length)
    else:
        start = torch.randint(0, noise.shape[-1] - length + 1, (1,)).item()
    return noise[start:start + length]


def _estimate_signal_rms(
    signal: torch.Tensor,
    *,
    mode: Literal["full", "active"] = "full",
    frame_samples: int = 512,
    threshold_db: float = -40.0,
    eps: float = 1e-12,
) -> float:
    """Estimate RMS from either the full signal or active frames only."""
    if signal.dim() != 1:
        raise ValueError(f"signal must be 1-D, got {tuple(signal.shape)}")
    if frame_samples <= 0:
        raise ValueError(f"frame_samples must be positive, got {frame_samples}")
    if mode == "full":
        return float(torch.sqrt((signal * signal).mean() + eps))
    if mode != "active":
        raise ValueError(f"Unsupported RMS mode: {mode}")

    if signal.numel() == 0:
        return 0.0

    if signal.numel() < frame_samples:
        frames = signal.unsqueeze(0)
    else:
        hop_samples = frame_samples
        frames = signal.unfold(0, frame_samples, hop_samples)
    frame_rms = torch.sqrt((frames * frames).mean(dim=-1) + eps)
    frame_db = 20.0 * torch.log10(frame_rms + eps)
    active_mask = frame_db >= threshold_db
    if not bool(active_mask.any()):
        return float(torch.sqrt((signal * signal).mean() + eps))
    active_power = (frames[active_mask] * frames[active_mask]).mean()
    return float(torch.sqrt(active_power + eps))


def scale_noise_to_snr(
    reference: torch.Tensor,
    noise: torch.Tensor,
    snr_db: float,
    *,
    rng: random.Random | None = None,
    rms_mode: Literal["full", "active"] = "full",
    activity_frame_samples: int = 512,
    activity_threshold_db: float = -40.0,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Scale ``noise`` to achieve the requested SNR relative to ``reference``."""
    fitted_noise = _fit_noise_length(noise, reference.shape[-1], rng)
    if float(reference.abs().max()) < eps or float(fitted_noise.abs().max()) < eps:
        return torch.zeros_like(reference)
    ref_rms = _estimate_signal_rms(
        reference,
        mode=rms_mode,
        frame_samples=activity_frame_samples,
        threshold_db=activity_threshold_db,
        eps=eps,
    )
    noise_rms = _estimate_signal_rms(
        fitted_noise,
        mode=rms_mode,
        frame_samples=activity_frame_samples,
        threshold_db=activity_threshold_db,
        eps=eps,
    )
    if ref_rms < eps or noise_rms < eps:
        return torch.zeros_like(reference)
    target_noise_rms = ref_rms / (10.0 ** (snr_db / 20.0))
    return fitted_noise * (target_noise_rms / noise_rms)


def add_noise_at_snr(
    signal: torch.Tensor,
    noise: torch.Tensor,
    snr_db: float,
    eps: float = 1e-12,
    *,
    rms_mode: Literal["full", "active"] = "full",
    activity_frame_samples: int = 512,
    activity_threshold_db: float = -40.0,
) -> torch.Tensor:
    """Add pre-sampled noise to ``signal`` at the specified SNR.

    The noise is scaled so that:

    .. math::

        \\mathrm{SNR} = 10 \\log_{10}
            \\frac{\\text{signal power}}{\\text{noise power}}

    matches the requested ``snr_db``. If ``noise`` is shorter than
    ``signal`` it is looped; if longer, a random window is taken.
    """
    scaled_noise = scale_noise_to_snr(
        signal,
        noise,
        snr_db,
        rms_mode=rms_mode,
        activity_frame_samples=activity_frame_samples,
        activity_threshold_db=activity_threshold_db,
        eps=eps,
    )
    return signal + scaled_noise


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


def apply_random_gain(
    signal: torch.Tensor,
    gain_range_db: tuple[float, float] = (-6.0, 6.0),
    rng: random.Random | None = None,
) -> torch.Tensor:
    """Apply a random gain to ``signal`` in decibels."""
    if gain_range_db[0] > gain_range_db[1]:
        raise ValueError(
            f"gain_range_db must be ordered as (min, max), got {gain_range_db}"
        )
    sampler = rng if rng is not None else random
    gain_db = sampler.uniform(*gain_range_db)
    gain = 10.0 ** (gain_db / 20.0)
    return signal * gain


def apply_bandwidth_limit(
    signal: torch.Tensor,
    sample_rate: int = 16000,
    cutoff_range_hz: tuple[float, float] = (4000.0, 7000.0),
    rng: random.Random | None = None,
    order: int = 101,
) -> torch.Tensor:
    """Apply a random low-pass FIR filter to simulate channel bandwidth limits."""
    if signal.dim() != 1:
        raise ValueError(f"signal must be 1-D, got {tuple(signal.shape)}")
    if sample_rate <= 0:
        raise ValueError(f"sample_rate must be positive, got {sample_rate}")
    if order <= 0 or order % 2 == 0:
        raise ValueError(f"order must be a positive odd integer, got {order}")

    min_cutoff, max_cutoff = cutoff_range_hz
    nyquist = sample_rate / 2.0
    if not 0.0 < min_cutoff <= max_cutoff < nyquist:
        raise ValueError(
            "cutoff_range_hz must satisfy 0 < min <= max < Nyquist, got "
            f"{cutoff_range_hz} for sample_rate={sample_rate}"
        )

    sampler = rng if rng is not None else random
    cutoff_hz = sampler.uniform(min_cutoff, max_cutoff)
    fc_normalized = cutoff_hz / nyquist

    half_order = order // 2
    n = torch.arange(
        -half_order,
        half_order + 1,
        device=signal.device,
        dtype=signal.dtype,
    )
    kernel = fc_normalized * torch.sinc(fc_normalized * n)
    kernel = kernel * torch.hamming_window(
        order,
        periodic=False,
        device=signal.device,
        dtype=signal.dtype,
    )
    kernel = kernel / kernel.sum().clamp_min(torch.finfo(kernel.dtype).eps)

    filtered = F.conv1d(
        signal.view(1, 1, -1),
        kernel.view(1, 1, -1),
        padding=half_order,
    )
    return filtered.view_as(signal)


__all__ = [
    "RoomPreset",
    "ReverbConfig",
    "synth_room_rir",
    "apply_rir",
    "_fit_noise_length",
    "_estimate_signal_rms",
    "scale_noise_to_snr",
    "add_noise_at_snr",
    "add_gaussian_noise",
    "apply_random_gain",
    "apply_bandwidth_limit",
]
