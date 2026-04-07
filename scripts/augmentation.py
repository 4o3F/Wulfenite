"""Acoustic augmentation for Phase 0b training.

Bridges the gap between AISHELL-1 (clean studio recordings) and real-world
small-room streaming audio. Provides:

1. Synthetic room impulse responses (RIRs) with a simple early-reflection
   + exponentially-decaying diffuse tail model. Good enough to simulate
   "target and interferer standing in different spots of a small room"
   without requiring a real RIR dataset download.

2. Additive broadband noise at a controllable SNR. Simulates room tone,
   mic self-noise, HVAC, etc.

These are deliberately cheap to run on-the-fly inside the Dataset
__getitem__ call — no torchaudio dependency, pure torch + random.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import torch


@dataclass
class ReverbConfig:
    """Parameters for synth_room_rir."""

    sample_rate: int = 16000
    rt60_range: tuple[float, float] = (0.08, 0.35)  # seconds; small-room range
    num_early_reflections_range: tuple[int, int] = (3, 6)
    early_delay_range_ms: tuple[float, float] = (5.0, 35.0)
    early_amplitude_range: tuple[float, float] = (0.25, 0.70)
    diffuse_tail_scale: float = 0.10


def synth_room_rir(cfg: ReverbConfig, rng: random.Random) -> torch.Tensor:
    """Generate one synthetic small-room RIR.

    The RIR is:
        direct_path (always 1.0 at t=0)
      + a handful of specular early reflections, each at a random delay
        with a random signed amplitude
      + a diffuse exponentially-decaying noise tail

    The result is normalized to peak=1 so convolution does not change
    overall gain beyond the direct path.
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

    # Diffuse exponential-decay tail
    # tau chosen so that amplitude falls by ~60 dB over rt60 seconds:
    #   exp(-rt60 / tau) = 10**(-3)  =>  tau = rt60 / (3 * ln10)
    tau_samples = max(1.0, rt60 * cfg.sample_rate / (3.0 * math.log(10.0)))
    t = torch.arange(length, dtype=torch.float32)
    envelope = torch.exp(-t / tau_samples)
    rir = rir + torch.randn(length) * envelope * cfg.diffuse_tail_scale

    peak = float(rir.abs().max())
    if peak > 0:
        rir = rir / peak
    return rir


def apply_rir(signal: torch.Tensor, rir: torch.Tensor) -> torch.Tensor:
    """Convolve a 1D signal with a 1D RIR, return the same length.

    Uses FFT-based convolution rather than direct ``conv1d``. For RIR
    kernels in the ~5k-15k sample range that we use here, FFT convolution
    is 50-100× faster on CPU than direct convolution, which removes the
    main bottleneck in the on-the-fly augmentation pipeline.

    Cost: O((N+K) log(N+K)) instead of O(N·K).

    The result is head-aligned with the input (length == ``signal.shape[0]``).
    """
    n_signal = signal.shape[-1]
    n_kernel = rir.shape[-1]
    n_full = n_signal + n_kernel - 1
    # Round up to a power of two for the FFT (much faster transform).
    n_fft = 1 << (n_full - 1).bit_length()

    S = torch.fft.rfft(signal, n=n_fft)
    K = torch.fft.rfft(rir, n=n_fft)
    y = torch.fft.irfft(S * K, n=n_fft)
    return y[:n_signal]


def add_gaussian_noise(signal: torch.Tensor, snr_db: float,
                       rng: random.Random | None = None) -> torch.Tensor:
    """Add Gaussian white noise to ``signal`` at the specified SNR."""
    sig_rms = float(torch.sqrt((signal * signal).mean() + 1e-12))
    if sig_rms < 1e-9:
        return signal
    noise = torch.randn_like(signal)
    noise_rms = float(torch.sqrt((noise * noise).mean() + 1e-12))
    target_noise_rms = sig_rms / (10.0 ** (snr_db / 20.0))
    noise = noise * (target_noise_rms / (noise_rms + 1e-12))
    return signal + noise


# ---------------------------------------------------------------------------
# Convenience: full augmentation for one (target, interferer, enrollment) triplet
# ---------------------------------------------------------------------------


@dataclass
class AugmentationConfig:
    """High-level augmentation knobs for the AISHELL mixer."""

    reverb: ReverbConfig = None  # type: ignore[assignment]
    noise_snr_range_db: tuple[float, float] = (15.0, 30.0)
    p_reverb: float = 0.85   # probability of applying reverb
    p_noise: float = 0.80    # probability of adding noise
    # If True, the enrollment is also passed through a random (different) RIR
    # so the speaker embedding sees the same kind of coloration it would in
    # a deployed streaming setup where the enrollment was recorded in-room.
    reverb_enrollment: bool = True

    # Target silence augmentation (for false-extraction prevention).
    # Periodically zero out random regions of the target signal so the
    # model learns "when target is silent, output silence" — even when
    # the interferer is loud. Without this, the model has only ever seen
    # full-speech targets and will hallucinate an extraction whenever the
    # interferer is present, regardless of whether the target is talking.
    sample_rate: int = 16000
    target_silence_prob: float = 0.5     # apply silence to this fraction of samples
    target_silence_max_frac: float = 0.80  # never zero more than this fraction (keep loss numerically stable)
    target_silence_min_region_ms: float = 200.0
    target_silence_max_regions: int = 3

    def __post_init__(self) -> None:
        if self.reverb is None:
            self.reverb = ReverbConfig()


def augment_triplet(
    target: torch.Tensor,
    interferer: torch.Tensor,
    enrollment: torch.Tensor,
    cfg: AugmentationConfig,
    rng: random.Random,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply per-source reverb + enrollment reverb, return the augmented
    ``(target, interferer, enrollment)`` ready for mixing downstream.

    The target and the interferer get DIFFERENT RIRs (simulating different
    spatial positions in the same room). The enrollment gets its own
    separate RIR (recorded earlier from the same room).
    """
    if rng.random() < cfg.p_reverb:
        rir_tgt = synth_room_rir(cfg.reverb, rng)
        target = apply_rir(target, rir_tgt)

        rir_int = synth_room_rir(cfg.reverb, rng)
        interferer = apply_rir(interferer, rir_int)

        if cfg.reverb_enrollment:
            rir_enr = synth_room_rir(cfg.reverb, rng)
            enrollment = apply_rir(enrollment, rir_enr)

    return target, interferer, enrollment


def add_mixture_noise(mixture: torch.Tensor,
                      cfg: AugmentationConfig,
                      rng: random.Random) -> torch.Tensor:
    """Optionally add broadband noise to the final mixture."""
    if rng.random() < cfg.p_noise:
        snr = rng.uniform(*cfg.noise_snr_range_db)
        mixture = add_gaussian_noise(mixture, snr, rng)
    return mixture


def insert_target_silence(target: torch.Tensor,
                          cfg: AugmentationConfig,
                          rng: random.Random) -> torch.Tensor:
    """Zero out one or more contiguous regions of the target waveform.

    Returns a NEW tensor (the input is not mutated). Caps the total
    silenced fraction at ``cfg.target_silence_max_frac`` so the loss
    always has some non-zero target energy to compute SDR against.

    Used to teach the model that "when the target speaker is silent,
    output silence even if the interferer is loud" — a behavior that
    full-speech-only training never exposes the model to.
    """
    if rng.random() >= cfg.target_silence_prob:
        return target

    n = target.shape[-1]
    if n <= 0:
        return target

    max_total_silence = int(cfg.target_silence_max_frac * n)
    min_region = int(cfg.target_silence_min_region_ms * cfg.sample_rate / 1000.0)
    if max_total_silence < min_region:
        return target

    out = target.clone()
    silenced_so_far = 0
    n_regions = rng.randint(1, cfg.target_silence_max_regions)
    for _ in range(n_regions):
        remaining = max_total_silence - silenced_so_far
        if remaining < min_region:
            break
        region_len = rng.randint(min_region, remaining)
        max_start = n - region_len
        if max_start <= 0:
            break
        start = rng.randint(0, max_start)
        out[start:start + region_len] = 0.0
        silenced_so_far += region_len
    return out
