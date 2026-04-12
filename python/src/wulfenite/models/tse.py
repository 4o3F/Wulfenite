"""Target Speaker Extraction wrapper combining CAM++ + SpeakerBeam-SS."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from .campplus import CAMPPlus, EMBEDDING_SIZE, load_campplus_cn_common
from .campplus_encoder import CampPlusSpeakerEncoder
from .speakerbeam_ss import SpeakerBeamSS, SpeakerBeamSSConfig


class WulfeniteTSE(nn.Module):
    """End-to-end TSE model: CAM++ speaker encoder + SpeakerBeam-SS."""

    def __init__(
        self,
        speaker_encoder: CampPlusSpeakerEncoder,
        separator: SpeakerBeamSS | None = None,
    ) -> None:
        super().__init__()
        self.speaker_encoder = speaker_encoder
        self.separator = separator or SpeakerBeamSS()

    @classmethod
    def from_campplus(
        cls,
        campplus_checkpoint: str | Path | None,
        separator_config: SpeakerBeamSSConfig | None = None,
    ) -> "WulfeniteTSE":
        """Build a TSE model with a fine-tunable CAM++ speaker encoder."""
        separator_cfg = separator_config or SpeakerBeamSSConfig()
        if separator_cfg.speaker_embed_dim != EMBEDDING_SIZE:
            raise ValueError(
                "separator_config.speaker_embed_dim must match CAM++ embedding size "
                f"{EMBEDDING_SIZE}; got {separator_cfg.speaker_embed_dim}"
            )
        separator = SpeakerBeamSS(separator_cfg)
        if campplus_checkpoint is None:
            backbone = CAMPPlus()
        else:
            backbone = load_campplus_cn_common(
                campplus_checkpoint,
                freeze=False,
            )
        encoder = CampPlusSpeakerEncoder(backbone=backbone)
        return cls(
            speaker_encoder=encoder,
            separator=separator,
        )

    @torch.no_grad()
    def encode_enrollment(self, waveform: torch.Tensor) -> torch.Tensor:
        """Compute the L2-normalized speaker embedding from raw audio."""
        return self.speaker_encoder.encode_enrollment(waveform)

    def separate(
        self,
        mixture: torch.Tensor,
        speaker_embedding: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Run the separator with a pre-computed embedding."""
        if speaker_embedding.size(0) == 1 and mixture.size(0) > 1:
            speaker_embedding = speaker_embedding.expand(mixture.size(0), -1)
        return self.separator(mixture, speaker_embedding)

    def forward(
        self,
        mixture: torch.Tensor,
        enrollment: torch.Tensor,
        enrollment_fbank: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute speaker embedding on the fly and run separation."""
        if enrollment.dim() == 1:
            enrollment = enrollment.unsqueeze(0)
        if enrollment_fbank is not None and enrollment_fbank.dim() == 2:
            enrollment_fbank = enrollment_fbank.unsqueeze(0)

        raw_emb, norm_emb = self.speaker_encoder(
            enrollment,
            fbank=enrollment_fbank,
        )
        if norm_emb.size(0) == 1 and mixture.size(0) > 1:
            raw_emb = raw_emb.expand(mixture.size(0), -1)
            norm_emb = norm_emb.expand(mixture.size(0), -1)
        out = self.separate(mixture, norm_emb)
        out["embedding"] = norm_emb
        out["raw_embedding"] = raw_emb
        return out
