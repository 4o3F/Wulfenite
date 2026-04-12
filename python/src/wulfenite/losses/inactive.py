"""Framewise penalty on output energy during target-inactive regions."""

from __future__ import annotations

import torch


def target_inactive_loss(
    estimate: torch.Tensor,
    mixture: torch.Tensor,
    *,
    inactive_frames: torch.Tensor,
    frame_size: int = 160,
    eps: float = 1e-8,
    reduction: str = "mean",
) -> torch.Tensor:
    """Penalize output energy on frames where the target is silent.

    The penalty is the estimate-to-mixture energy ratio on the frames
    marked inactive by the mixer:

    .. math::

        \\frac{\\lVert \\hat{s}_f \\rVert^2}{\\lVert m_f \\rVert^2 + \\epsilon}

    Args:
        estimate: ``[B, T]`` estimated waveform.
        mixture: ``[B, T]`` input mixture waveform.
        inactive_frames: ``[B, F]`` boolean or float mask. If ``F``
            does not match the derived frame count for ``frame_size``,
            it is downsampled by logical OR.
        frame_size: samples per energy frame.
        eps: stabilizer.
        reduction: ``"mean"``, ``"sum"``, or ``"none"``.

    Returns:
        Scalar loss (``"mean"`` / ``"sum"``) or ``[B]`` (``"none"``).
    """
    if estimate.shape != mixture.shape:
        raise ValueError(
            f"estimate {tuple(estimate.shape)} vs mixture {tuple(mixture.shape)}"
        )
    if inactive_frames.dim() != 2 or inactive_frames.shape[0] != estimate.shape[0]:
        raise ValueError(
            "inactive_frames must be [B, F]; got "
            f"{tuple(inactive_frames.shape)}"
        )

    B, T = estimate.shape
    n_frames = T // frame_size
    usable = n_frames * frame_size
    est = estimate[:, :usable].reshape(B, n_frames, frame_size)
    mix = mixture[:, :usable].reshape(B, n_frames, frame_size)

    mask = inactive_frames.to(est.device).bool()
    if mask.shape[1] != n_frames:
        if mask.shape[1] % n_frames != 0:
            raise ValueError(
                f"inactive_frames length {mask.shape[1]} cannot align to {n_frames} "
                f"frames for frame_size={frame_size}"
            )
        factor = mask.shape[1] // n_frames
        mask = mask.reshape(B, n_frames, factor).any(dim=-1)

    est_energy = (est * est).sum(dim=-1)
    mix_energy = (mix * mix).sum(dim=-1)
    # Exclude near-silent mixture frames to avoid division instability
    # on BACKGROUND_ONLY events where both energies approach zero.
    audible = mix_energy > eps
    masked = mask.float() * audible.float()
    denom = masked.sum(dim=-1).clamp(min=1.0)
    per_sample = ((est_energy / (mix_energy + eps)) * masked).sum(dim=-1) / denom

    if reduction == "mean":
        return per_sample.mean()
    if reduction == "sum":
        return per_sample.sum()
    if reduction == "none":
        return per_sample
    raise ValueError(f"unknown reduction {reduction!r}")
