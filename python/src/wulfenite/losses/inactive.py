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
    threshold: float = 0.05,
    topk_fraction: float = 0.25,
    reduction: str = "mean",
) -> torch.Tensor:
    """Penalize output energy on frames where the target is silent.

    The penalty is the estimate-to-mixture energy ratio on the frames
    marked inactive by the mixer. Near-silent mixture frames
    (``mix_energy <= 1e-4``) are excluded. A soft threshold keeps tiny
    residuals cheap while top-k pooling focuses the loss on the worst
    leakage frames:

    .. math::

        \\max\\left(0, \\frac{\\lVert \\hat{s}_f \\rVert^2}
        {\\lVert m_f \\rVert^2 + \\epsilon} - \\tau \\right)^2

    Args:
        estimate: ``[B, T]`` estimated waveform.
        mixture: ``[B, T]`` input mixture waveform.
        inactive_frames: ``[B, F]`` boolean or float mask. If ``F``
            does not match the derived frame count for ``frame_size``,
            it is downsampled by logical OR.
        frame_size: samples per energy frame.
        eps: stabilizer for the ratio computation.
        threshold: leakage ratio threshold ``tau``.
        topk_fraction: fraction of the worst masked frames to average
            per sample. Set to ``1.0`` to use all masked frames.
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
    if not 0.0 < topk_fraction <= 1.0:
        raise ValueError(
            f"topk_fraction must be in (0, 1]; got {topk_fraction}"
        )

    # Exclude near-silent mixture frames and focus on the strongest
    # leakage frames rather than averaging over every inactive frame.
    audible = mix_energy > 1e-4
    masked = mask.float() * audible.float()
    ratio = est_energy / (mix_energy + eps)
    penalty = torch.relu(ratio - threshold).square()

    per_sample_terms: list[torch.Tensor] = []
    for i in range(B):
        active = masked[i].bool()
        if not bool(active.any().item()):
            per_sample_terms.append(torch.zeros((), device=estimate.device))
            continue
        values = penalty[i][active]
        k = max(1, int(torch.ceil(torch.tensor(
            values.numel() * topk_fraction, device=estimate.device,
        )).item()))
        topk_vals, _ = torch.topk(values, k=min(k, values.numel()))
        per_sample_terms.append(topk_vals.mean())
    per_sample = torch.stack(per_sample_terms, dim=0)

    if reduction == "mean":
        return per_sample.mean()
    if reduction == "sum":
        return per_sample.sum()
    if reduction == "none":
        return per_sample
    raise ValueError(f"unknown reduction {reduction!r}")
