"""Mask-aware combined loss for wulfenite training.

Combines the four per-component losses defined elsewhere in this
package into a single scalar suitable for ``loss.backward()``:

- :func:`wulfenite.losses.sdr.sdr_loss`: direct SDR, per-sample,
  only applied to target-present samples (silent targets have zero
  energy and would blow up the SDR denominator).
- :func:`wulfenite.losses.mr_stft.MultiResolutionSTFTLoss`: frequency
  supervision, also only on target-present samples for consistency
  with the SDR branch.
- :func:`wulfenite.losses.silence.target_absent_loss`: energy penalty,
  only on target-absent samples.
- :func:`wulfenite.losses.presence.presence_loss`: BCE on the presence
  head, always applied to all samples.

The split is based on a per-sample binary ``target_present`` label
coming from the mixer (``1`` when the mixture contains target speech,
``0`` when it is interferer-only). Empty-subset branches are safely
skipped — each loss contribution is materialized as a zero-tensor if
no samples fall into its group, so the backward pass is always
well-defined regardless of batch composition.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .mr_stft import MultiResolutionSTFTLoss
from .presence import presence_loss
from .sdr import sdr_loss
from .silence import target_absent_loss


@dataclass
class LossWeights:
    """Per-component weights for :class:`WulfeniteLoss`.

    Defaults follow the design doc (``docs/architecture.md`` section
    5). ``presence`` is deliberately small because the presence head
    is auxiliary — its purpose is to feed a clean gradient to the
    silent branch, not to dominate optimization.
    """

    sdr: float = 1.0
    mr_stft: float = 1.0
    absent: float = 1.0
    presence: float = 0.1
    speaker_cls: float = 0.2


@dataclass
class LossParts:
    """Per-component scalar breakdown for logging.

    All values are plain Python floats, safe to log to tensorboard /
    csv without keeping the autograd graph alive.
    """

    total: float
    sdr: float
    mr_stft: float
    absent: float
    presence: float
    speaker_cls: float
    n_present: int
    n_absent: int


class WulfeniteLoss(nn.Module):
    """End-to-end loss module used during training.

    Built as an ``nn.Module`` so its ``MultiResolutionSTFTLoss``
    sub-module's window buffers move with ``.to(device)`` and the
    trainer can treat the loss the same as any other model component.
    """

    def __init__(
        self,
        weights: LossWeights | None = None,
        mr_stft_loss: MultiResolutionSTFTLoss | None = None,
    ) -> None:
        super().__init__()
        self.weights = weights or LossWeights()
        self.mr_stft = mr_stft_loss or MultiResolutionSTFTLoss()

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        clean: torch.Tensor,
        target: torch.Tensor,
        mixture: torch.Tensor,
        target_present: torch.Tensor,
        presence_logit: torch.Tensor | None = None,
        speaker_logits: torch.Tensor | None = None,
        target_speaker_idx: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, LossParts]:
        """Compute the full training loss.

        Args:
            clean: ``[B, T]`` model estimate.
            target: ``[B, T]`` reference waveform. For samples with
                ``target_present == 0`` the content is ignored and
                may be any convenient placeholder (zeros are the
                obvious choice from the mixer).
            mixture: ``[B, T]`` input mixture. Used by the
                target-absent branch as the energy reference.
            target_present: ``[B]`` binary label, ``1.0`` if the
                sample's target is audible and ``0.0`` otherwise.
                Can be float or long; coerced to float.
            presence_logit: ``[B]`` optional presence-head output.
                If ``None``, the presence term is skipped (useful for
                ablations or when running a model without the head).
            speaker_logits: optional ``[B, num_speakers]`` logits from
                the learnable encoder's auxiliary classifier.
            target_speaker_idx: optional ``[B]`` long tensor with the
                claimed target speaker id for each enrollment.

        Returns:
            Tuple of ``(total_loss, parts)`` where ``parts`` is a
            :class:`LossParts` dataclass with individual component
            values as plain floats.
        """
        if clean.shape != target.shape or clean.shape != mixture.shape:
            raise ValueError(
                "clean / target / mixture must all share shape "
                f"[B, T]; got {tuple(clean.shape)}, {tuple(target.shape)}, "
                f"{tuple(mixture.shape)}"
            )
        if target_present.shape != clean.shape[:1]:
            raise ValueError(
                f"target_present must be shape [B]; got "
                f"{tuple(target_present.shape)}"
            )
        if speaker_logits is not None and speaker_logits.size(0) != clean.shape[0]:
            raise ValueError(
                "speaker_logits must have batch dimension [B, ...]; got "
                f"{tuple(speaker_logits.shape)}"
            )
        if (
            target_speaker_idx is not None
            and target_speaker_idx.shape != clean.shape[:1]
        ):
            raise ValueError(
                "target_speaker_idx must be shape [B]; got "
                f"{tuple(target_speaker_idx.shape)}"
            )

        device = clean.device
        dtype = clean.dtype
        zero = torch.zeros((), device=device, dtype=dtype)

        present_mask = target_present.to(device).bool()
        absent_mask = ~present_mask
        n_present = int(present_mask.sum().item())
        n_absent = int(absent_mask.sum().item())

        # --- Present branch: SDR + MR-STFT ----------------------------
        l_sdr = zero
        l_stft = zero
        if n_present > 0:
            l_sdr = sdr_loss(
                clean[present_mask], target[present_mask]
            )
            l_stft = self.mr_stft(
                clean[present_mask], target[present_mask]
            )

        # --- Absent branch: energy penalty ----------------------------
        l_absent = zero
        if n_absent > 0:
            l_absent = target_absent_loss(
                clean[absent_mask], mixture[absent_mask]
            )

        # --- Presence head: BCE on all samples ------------------------
        l_presence = zero
        if presence_logit is not None:
            l_presence = presence_loss(
                presence_logit, target_present.to(device),
            )

        # --- Auxiliary speaker classification on all samples ----------
        l_speaker_cls = zero
        if speaker_logits is not None and target_speaker_idx is not None:
            l_speaker_cls = F.cross_entropy(
                speaker_logits,
                target_speaker_idx.to(device=device, dtype=torch.long),
            )

        w = self.weights
        total = (
            w.sdr * l_sdr
            + w.mr_stft * l_stft
            + w.absent * l_absent
            + w.presence * l_presence
            + w.speaker_cls * l_speaker_cls
        )

        parts = LossParts(
            total=float(total.detach()),
            sdr=float(l_sdr.detach()),
            mr_stft=float(l_stft.detach()),
            absent=float(l_absent.detach()),
            presence=float(l_presence.detach()),
            speaker_cls=float(l_speaker_cls.detach()),
            n_present=n_present,
            n_absent=n_absent,
        )
        return total, parts
