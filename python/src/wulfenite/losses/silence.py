"""Target-absent energy penalty.

Supervisory signal for the "target speaker is not talking" branch of
training. When the mixture contains only interferers and the target
is silent, the correct estimate is silence. The loss is the
energy of the estimate, normalized by the mixture's energy so that
the scale is invariant to the loudness of the interferers:

.. math::

    L_{\\text{absent}} = \\frac{\\lVert \\hat{\\mathbf{s}} \\rVert^2}
                             {\\lVert \\mathbf{m} \\rVert^2 + \\epsilon}

- ``estimate = 0`` → loss = 0 (perfect).
- ``estimate = mixture`` (pass-through) → loss = 1.
- Louder estimates → strictly larger loss.

This is the simple half of the mixture-aware silence training
described in ``docs/architecture.md`` section 5. The other half is
the binary cross-entropy on the presence head in
:mod:`wulfenite.losses.presence`.
"""

from __future__ import annotations

import torch


def target_absent_loss(
    estimate: torch.Tensor,
    mixture: torch.Tensor,
    *,
    eps: float = 1e-8,
    reduction: str = "mean",
) -> torch.Tensor:
    """Energy ratio penalty for target-absent training samples.

    Args:
        estimate: ``[B, T]`` model output on samples where target is
            silent (only interferers are present).
        mixture: ``[B, T]`` input mixture. Used only to normalize the
            estimate's energy so the loss is scale-invariant w.r.t.
            interferer loudness.
        eps: stabilizer for empty mixtures.
        reduction: ``"mean"``, ``"sum"``, or ``"none"``.

    Returns:
        Scalar loss (``"mean"`` / ``"sum"``) or ``[B]`` (``"none"``).
    """
    if estimate.shape != mixture.shape:
        raise ValueError(
            f"estimate {tuple(estimate.shape)} vs mixture {tuple(mixture.shape)}"
        )

    est_energy = (estimate * estimate).sum(dim=-1)      # [B]
    mix_energy = (mixture * mixture).sum(dim=-1) + eps  # [B]

    loss_per_sample = est_energy / mix_energy

    if reduction == "mean":
        return loss_per_sample.mean()
    if reduction == "sum":
        return loss_per_sample.sum()
    if reduction == "none":
        return loss_per_sample
    raise ValueError(f"unknown reduction {reduction!r}")
