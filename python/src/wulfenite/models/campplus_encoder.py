"""Thin CAM++ speaker encoder wrapper for Wulfenite."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import nn

from ..audio_features import compute_fbank_batch
from .campplus import CAMPPlus

if TYPE_CHECKING:
    from ..training.config import TrainingConfig


class CampPlusSpeakerEncoder(nn.Module):
    """Fine-tunable CAM++ backbone returning native 192-d embeddings."""

    def __init__(self, backbone: CAMPPlus) -> None:
        super().__init__()
        self.backbone = backbone
        for module in self.backbone.modules():
            if hasattr(module, "memory_efficient"):
                module.memory_efficient = False

    def forward(
        self,
        waveform: torch.Tensor | None = None,
        fbank: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode waveform or FBank into ``(raw, unit_l2)`` embeddings."""
        if fbank is None:
            if waveform is None:
                raise ValueError("Either waveform or fbank must be provided.")
            fbank = compute_fbank_batch(waveform)
        elif fbank.dim() == 2:
            fbank = fbank.unsqueeze(0)

        raw = self.backbone(fbank)
        norm = F.normalize(raw, p=2, dim=-1)
        return raw, norm

    @torch.no_grad()
    def encode_enrollment(self, waveform: torch.Tensor) -> torch.Tensor:
        """Inference helper mirroring the TSE enrollment API."""
        was_training = self.training
        self.eval()
        try:
            _, norm = self.forward(waveform)
        finally:
            if was_training:
                self.train()
        return norm

    def optimizer_groups(
        self,
        cfg: "TrainingConfig",
        base_lr: float | None = None,
    ) -> list[dict]:
        """Return the CAM++ backbone optimizer group."""
        lr = base_lr if base_lr is not None else cfg.encoder_lr
        params = [p for p in self.backbone.parameters() if p.requires_grad]
        return [{"name": "encoder_backbone", "params": params, "lr": lr}]
