"""CAM++ speaker-encoder adapter for Wulfenite.

Wraps the pretrained/native 192-dim CAM++ backbone so the separator
always receives an L2-normalized embedding in bottleneck space.
"""

from __future__ import annotations

import math
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


class ArcFaceClassifier(nn.Module):
    """ArcFace/AAM-Softmax classifier operating on unit speaker embeddings."""

    def __init__(
        self,
        emb_dim: int,
        num_classes: int,
        scale: float = 30.0,
        margin: float = 0.2,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_classes, emb_dim))
        nn.init.xavier_uniform_(self.weight)
        self.scale = float(scale)
        self.margin = float(margin)

    @property
    def out_features(self) -> int:
        return self.weight.size(0)

    @property
    def in_features(self) -> int:
        return self.weight.size(1)

    def _load_from_state_dict(
        self,
        state_dict: dict,
        prefix: str,
        local_metadata: dict,
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        # Older checkpoints stored an nn.Linear classifier with a bias term.
        state_dict.pop(prefix + "bias", None)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(
        self,
        embedding: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        emb = F.normalize(embedding, p=2, dim=-1)
        weight = F.normalize(self.weight, p=2, dim=-1)
        cosine = F.linear(emb, weight).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        if labels is None:
            return cosine * self.scale

        labels = labels.to(device=cosine.device, dtype=torch.long).reshape(-1)
        if labels.shape[0] != cosine.shape[0]:
            raise ValueError(
                "labels must have shape [B]; got "
                f"{tuple(labels.shape)} for cosine {tuple(cosine.shape)}"
            )

        cos_m = math.cos(self.margin)
        sin_m = math.sin(self.margin)
        sine = torch.sqrt((1.0 - cosine * cosine).clamp_min(0.0))
        phi = cosine * cos_m - sine * sin_m
        one_hot = F.one_hot(labels, num_classes=self.out_features).to(cosine.dtype)
        return ((one_hot * phi) + ((1.0 - one_hot) * cosine)) * self.scale


class CampPlusSpeakerEncoder(nn.Module):
    """Adapter that projects CAM++ embeddings into separator space."""

    supports_classifier = False
    supports_pretrain = False

    def __init__(
        self,
        backbone: CAMPPlus,
        bottleneck_dim: int,
        freeze_backbone: bool,
        num_speakers: int | None = None,
        projection_type: str = "mlp",
        projection_hidden_dim: int = 384,
        arcface_scale: float = 30.0,
        arcface_margin: float = 0.2,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.freeze_backbone = freeze_backbone
        if projection_type == "linear":
            self.to_separator = nn.Linear(EMBEDDING_SIZE, bottleneck_dim)
        elif projection_type == "mlp":
            if projection_hidden_dim <= 0:
                raise ValueError(
                    "projection_hidden_dim must be positive for MLP projection."
                )
            self.to_separator = nn.Sequential(
                nn.Linear(EMBEDDING_SIZE, projection_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(projection_hidden_dim, bottleneck_dim),
            )
        else:
            raise ValueError(f"Unsupported projection_type: {projection_type}")
        self.classifier = (
            ArcFaceClassifier(
                bottleneck_dim,
                num_speakers,
                scale=arcface_scale,
                margin=arcface_margin,
            )
            if num_speakers is not None else None
        )
        self.supports_classifier = self.classifier is not None
        self.supports_pretrain = self.classifier is not None

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

    def forward(
        self,
        waveform: torch.Tensor | None = None,
        fbank: torch.Tensor | None = None,
        speaker_labels: torch.Tensor | None = None,
    ) -> SpeakerEncoderOutput:
        """Encode waveform or pre-computed FBank into speaker embeddings."""
        if fbank is None:
            if waveform is None:
                raise ValueError("waveform must be provided when fbank is None.")
            fbank = compute_fbank_batch(waveform)
        elif fbank.dim() == 2:
            fbank = fbank.unsqueeze(0)
        raw = self._forward_backbone(fbank)
        norm = F.normalize(raw, p=2, dim=-1)
        projected = self.to_separator(norm)
        sep_emb = F.normalize(projected, p=2, dim=-1)
        logits = (
            self.classifier(sep_emb, speaker_labels)
            if self.classifier is not None else None
        )
        return SpeakerEncoderOutput(
            separator_embedding=sep_emb,
            native_embedding=norm.detach(),
            speaker_logits=logits,
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

    def optimizer_groups(
        self,
        cfg: "TrainingConfig",
        base_lr: float | None = None,
    ) -> list[dict]:
        """Return optimizer groups for frozen or fine-tuned CAM++."""
        head_lr = cfg.learning_rate if base_lr is None else base_lr
        groups = [
            {
                "name": "encoder_projection",
                "params": [p for p in self.to_separator.parameters() if p.requires_grad],
                "lr": head_lr,
            }
        ]
        if self.classifier is not None:
            groups.append(
                {
                    "name": "encoder_classifier",
                    "params": [
                        p for p in self.classifier.parameters() if p.requires_grad
                    ],
                    "lr": head_lr,
                }
            )
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
                        "lr": head_lr * cfg.encoder_lr_scale,
                    },
                )
        return groups
