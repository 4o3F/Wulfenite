"""Binary cross-entropy on the target-presence head.

SpeakerBeam-SS has an auxiliary "is the target speaker talking in
this chunk" head that outputs a single logit per batch element. This
loss is the standard BCE-with-logits against the ground-truth
presence label drawn from the mixer (1 for present, 0 for absent).

Training this head in parallel with the separator has two benefits:

- It gives the mixture-aware silence branch a crisp gradient signal
  even when the main SDR branch can't be computed (silent target).
- At inference time the head's output can drive a VAD-style gate on
  the separator output, suppressing any residual false extraction
  the model might emit during long target-absent stretches. This is
  an optional, runtime-side feature — the ONNX contract exports the
  logit so the Rust caller can decide whether to use it.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def presence_loss(
    presence_logit: torch.Tensor,
    target_present: torch.Tensor,
    *,
    reduction: str = "mean",
) -> torch.Tensor:
    """BCE-with-logits on the presence head.

    Args:
        presence_logit: ``[B]`` pre-sigmoid logit from the model's
            presence head.
        target_present: ``[B]`` ground-truth label, ``1.0`` if target
            is speaking in the sample and ``0.0`` otherwise. Can be
            float or long; coerced to float internally.
        reduction: passed through to
            ``F.binary_cross_entropy_with_logits``.

    Returns:
        Scalar loss (``"mean"`` / ``"sum"``) or ``[B]`` (``"none"``).
    """
    if presence_logit.shape != target_present.shape:
        raise ValueError(
            f"presence_logit {tuple(presence_logit.shape)} vs "
            f"target_present {tuple(target_present.shape)}"
        )
    return F.binary_cross_entropy_with_logits(
        presence_logit,
        target_present.to(presence_logit.dtype),
        reduction=reduction,
    )
