"""Mask-aware combined loss for Wulfenite training."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .mr_stft import MultiResolutionSTFTLoss
from .presence import presence_loss
from .recall import target_recall_loss
from .sdr import sdr_loss
from .silence import target_absent_loss


@dataclass
class LossWeights:
    """Per-component weights for :class:`WulfeniteLoss`."""

    sdr: float = 1.0
    mr_stft: float = 1.0
    absent: float = 0.5
    presence: float = 0.1
    recall: float = 0.5


@dataclass
class LossParts:
    """Per-component scalar breakdown for logging."""

    total: float
    sdr: float
    mr_stft: float
    recall: float
    absent: float
    presence: float
    n_present: int
    n_absent: int


class WulfeniteLoss(nn.Module):
    """End-to-end loss module used during training."""

    def __init__(
        self,
        weights: LossWeights | None = None,
        mr_stft_loss: MultiResolutionSTFTLoss | None = None,
        recall_frame_size: int = 320,
        recall_floor: float = 0.3,
    ) -> None:
        super().__init__()
        self.weights = weights or LossWeights()
        self.mr_stft = mr_stft_loss or MultiResolutionSTFTLoss()
        self.recall_frame_size = recall_frame_size
        self.recall_floor = recall_floor

    def forward(
        self,
        clean: torch.Tensor,
        target: torch.Tensor,
        mixture: torch.Tensor,
        target_present: torch.Tensor,
        presence_logit: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, LossParts]:
        """Compute the full training loss."""
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

        device = clean.device
        dtype = clean.dtype
        zero = torch.zeros((), device=device, dtype=dtype)

        present_mask = target_present.to(device).bool()
        absent_mask = ~present_mask
        n_present = int(present_mask.sum().item())
        n_absent = int(absent_mask.sum().item())

        l_sdr = zero
        l_stft = zero
        l_recall = zero
        if n_present > 0:
            l_sdr = sdr_loss(
                clean[present_mask], target[present_mask]
            )
            l_stft = self.mr_stft(
                clean[present_mask], target[present_mask]
            )
            if self.weights.recall > 0.0:
                l_recall = target_recall_loss(
                    clean[present_mask],
                    target[present_mask],
                    frame_size=self.recall_frame_size,
                    floor=self.recall_floor,
                )

        l_absent = zero
        if n_absent > 0:
            l_absent = target_absent_loss(
                clean[absent_mask], mixture[absent_mask]
            )

        l_presence = zero
        if presence_logit is not None:
            l_presence = presence_loss(
                presence_logit, target_present.to(device),
            )

        w = self.weights
        total = (
            w.sdr * l_sdr
            + w.mr_stft * l_stft
            + w.recall * l_recall
            + w.absent * l_absent
            + w.presence * l_presence
        )

        parts = LossParts(
            total=float(total.detach()),
            sdr=float(l_sdr.detach()),
            mr_stft=float(l_stft.detach()),
            recall=float(l_recall.detach()),
            absent=float(l_absent.detach()),
            presence=float(l_presence.detach()),
            n_present=n_present,
            n_absent=n_absent,
        )
        return total, parts
