"""CAM++ speaker-encoder adapter for Wulfenite.

Wraps the pretrained/native 192-dim CAM++ backbone so the separator
always receives an L2-normalized embedding in bottleneck space.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import nn

from .campplus import CAMPPlus, EMBEDDING_SIZE
from .dvector import compute_fbank_batch

if TYPE_CHECKING:
    from ..training.config import TrainingConfig


@dataclass
class SpeakerEncoderOutput:
    """Normalized speaker-encoder outputs consumed by the TSE wrapper."""

    separator_embedding: torch.Tensor
    native_embedding: torch.Tensor
    speaker_logits: torch.Tensor | None


class CampPlusSpeakerEncoder(nn.Module):
    """Adapter that projects CAM++ embeddings into separator space."""

    supports_classifier = False
    supports_pretrain = False

    def __init__(
        self,
        backbone: CAMPPlus,
        bottleneck_dim: int,
        freeze_backbone: bool,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.freeze_backbone = freeze_backbone
        self.to_separator = nn.Linear(EMBEDDING_SIZE, bottleneck_dim)

        if self.freeze_backbone:
            self.backbone.eval()
            for p in self.backbone.parameters():
                p.requires_grad_(False)
        else:
            # CAM++ uses activation checkpointing internally. The
            # enrollment FBank input is non-differentiable, so no input
            # tensor requires grad; leaving checkpointing enabled would
            # suppress parameter gradients during fine-tuning.
            for module in self.backbone.modules():
                if hasattr(module, "memory_efficient"):
                    module.memory_efficient = False

    def train(self, mode: bool = True) -> "CampPlusSpeakerEncoder":
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def _forward_backbone(self, fbank: torch.Tensor) -> torch.Tensor:
        if self.freeze_backbone:
            with torch.no_grad():
                return self.backbone(fbank)
        return self.backbone(fbank)

    def forward(self, waveform: torch.Tensor) -> SpeakerEncoderOutput:
        """Encode raw waveform into separator-space speaker embeddings."""
        fbank = compute_fbank_batch(waveform)
        raw = self._forward_backbone(fbank)
        norm = F.normalize(raw, p=2, dim=-1)
        proj = self.to_separator(norm)
        sep_emb = F.normalize(proj, p=2, dim=-1)
        return SpeakerEncoderOutput(
            separator_embedding=sep_emb,
            native_embedding=norm.detach(),
            speaker_logits=None,
        )

    @torch.no_grad()
    def encode_enrollment(self, waveform: torch.Tensor) -> torch.Tensor:
        """Inference helper mirroring the learnable encoder API."""
        was_training = self.training
        self.eval()
        output = self.forward(waveform)
        if was_training:
            self.train()
        return output.separator_embedding

    def optimizer_groups(self, cfg: "TrainingConfig") -> list[dict]:
        """Return optimizer groups for frozen or fine-tuned CAM++."""
        groups = [
            {
                "name": "encoder_projection",
                "params": [p for p in self.to_separator.parameters() if p.requires_grad],
                "lr": cfg.learning_rate,
            }
        ]
        if not self.freeze_backbone:
            backbone_params = [
                p for p in self.backbone.parameters() if p.requires_grad
            ]
            if backbone_params:
                groups.insert(
                    0,
                    {
                        "name": "encoder_backbone",
                        "params": backbone_params,
                        "lr": cfg.learning_rate * cfg.encoder_lr_scale,
                    },
                )
        return groups
