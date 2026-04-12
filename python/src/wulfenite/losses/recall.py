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
    active_frames: torch.Tensor | None = None,
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
        active_frames: optional ``[B, F]`` ground-truth activity mask.
            When provided it replaces the target-energy thresholding.
            If ``F`` does not match the derived frame count for
            ``frame_size``, the mask is downsampled by logical OR.
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

    if active_frames is None:
        active = tgt_energy > energy_threshold
    else:
        if active_frames.dim() != 2 or active_frames.shape[0] != B:
            raise ValueError(
                f"active_frames must be [B, F]; got {tuple(active_frames.shape)}"
            )
        mask = active_frames[:, :active_frames.shape[1]].to(est_energy.device).bool()
        if mask.shape[1] != n_frames:
            if mask.shape[1] % n_frames != 0:
                raise ValueError(
                    f"active_frames length {mask.shape[1]} cannot align to {n_frames} "
                    f"frames for frame_size={frame_size}"
                )
            factor = mask.shape[1] // n_frames
            mask = mask.reshape(B, n_frames, factor).any(dim=-1)
        active = mask
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
