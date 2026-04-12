"""Anti-suppression recall loss.

Penalizes frames where the estimate energy falls below a fraction of
the target energy. This directly attacks the dropped-word symptom
by ensuring the model maintains minimum output energy on active
target frames.

The loss is asymmetric: only under-estimation is penalized, never
over-estimation (that is SDR's job).
"""

from __future__ import annotations

import torch


def target_recall_loss(
    estimate: torch.Tensor,
    target: torch.Tensor,
    *,
    frame_size: int = 320,
    floor: float = 0.3,
    energy_threshold: float = 1e-6,
    eps: float = 1e-8,
    reduction: str = "mean",
) -> torch.Tensor:
    """Penalize frames where estimate energy falls below floor * target energy.

    Only active target frames (energy > threshold) contribute.

    Args:
        estimate: [B, T] estimated waveform.
        target: [B, T] target waveform.
        frame_size: window size for per-frame energy computation.
        floor: minimum ratio estimate_energy / target_energy.
        energy_threshold: minimum target frame energy to count as active.
        eps: stabilizer.
        reduction: "mean", "sum", or "none".

    Returns:
        Scalar loss or [B] per-sample.
    """
    B, T = estimate.shape
    n_frames = T // frame_size
    usable = n_frames * frame_size
    est = estimate[:, :usable].reshape(B, n_frames, frame_size)
    tgt = target[:, :usable].reshape(B, n_frames, frame_size)

    est_energy = (est * est).sum(dim=-1)
    tgt_energy = (tgt * tgt).sum(dim=-1)

    active = tgt_energy > energy_threshold
    ratio = est_energy / (tgt_energy + eps)
    penalty = torch.clamp(floor - ratio, min=0.0)
    penalty = penalty * active.float()

    n_active = active.float().sum(dim=-1).clamp(min=1.0)
    per_sample = penalty.sum(dim=-1) / n_active

    if reduction == "mean":
        return per_sample.mean()
    if reduction == "sum":
        return per_sample.sum()
    if reduction == "none":
        return per_sample
    raise ValueError(f"unknown reduction {reduction!r}")
