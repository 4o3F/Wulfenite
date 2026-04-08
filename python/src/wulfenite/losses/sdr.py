"""Direct (non-scale-invariant) SDR loss.

The signal-to-distortion ratio between an estimate and a target, with
NO scale invariance. Defined as:

.. math::

    \\mathrm{SDR} = 10 \\log_{10}
        \\frac{\\lVert \\mathbf{s} \\rVert^2}
             {\\lVert \\hat{\\mathbf{s}} - \\mathbf{s} \\rVert^2}

and the corresponding loss is ``-SDR`` (higher SDR → more negative
loss). Optional zero-mean normalization removes DC bias before the
error is measured.

**Why non-scale-invariant.** Wulfenite v1 (BSRNN) trained with
SI-SDR and repeatedly fell into degenerate solutions — Phase 0a
produced tiny-amplitude outputs (shape-correct, scale → 0), Phase 0b
with SI-SDR + log-mag penalty slid into pass-through
(``output ≈ mixture``). Both look fine by the scale-invariant metric
but are useless on real audio. Direct SDR has neither escape hatch:
pass-through gives SDR ≈ input SNR (≈ 0 dB at 0 dB training SNR),
tiny-output gives SDR ≈ 0 dB, only real separation pushes SDR
strongly positive. See ``docs/architecture.md`` section 5 for the
full rationale.
"""

from __future__ import annotations

import torch


def sdr_loss(
    estimate: torch.Tensor,
    target: torch.Tensor,
    *,
    eps: float = 1e-8,
    zero_mean: bool = True,
    reduction: str = "mean",
) -> torch.Tensor:
    """Negative direct SDR, non-scale-invariant.

    Args:
        estimate: ``[B, T]`` time-domain estimate.
        target: ``[B, T]`` time-domain reference. Must have nonzero
            energy per batch element — silent targets should be
            routed through :func:`wulfenite.losses.silence.target_absent_loss`
            instead.
        eps: small constant added to the denominator and inside the
            log to keep the loss finite on near-perfect estimates.
        zero_mean: subtract per-signal mean before computing the
            error, removing DC bias. Matches the state-of-the-art
            Conv-TasNet-era convention.
        reduction: ``"mean"``, ``"sum"``, or ``"none"``. ``"none"``
            returns a per-sample loss of shape ``[B]``.

    Returns:
        Scalar loss (``"mean"`` / ``"sum"``) or ``[B]`` (``"none"``).
    """
    if estimate.shape != target.shape:
        raise ValueError(
            f"estimate shape {tuple(estimate.shape)} must equal "
            f"target shape {tuple(target.shape)}"
        )

    if zero_mean:
        estimate = estimate - estimate.mean(dim=-1, keepdim=True)
        target = target - target.mean(dim=-1, keepdim=True)

    error = estimate - target
    target_energy = (target * target).sum(dim=-1)       # [B]
    error_energy = (error * error).sum(dim=-1) + eps    # [B]

    ratio = target_energy / error_energy
    sdr_per_sample = 10.0 * torch.log10(ratio + eps)    # [B], positive = good

    loss_per_sample = -sdr_per_sample                   # [B]

    if reduction == "mean":
        return loss_per_sample.mean()
    if reduction == "sum":
        return loss_per_sample.sum()
    if reduction == "none":
        return loss_per_sample
    raise ValueError(f"unknown reduction {reduction!r}")
