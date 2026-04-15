"""Personalized DeepFilterNet2."""

from __future__ import annotations

from typing import Any

import torch

from .dfnet2 import DfNet, DfNetStreamState


class PDfNet2(DfNet):
    """DfNet2 with speaker embedding injection after encoder fusion."""

    def __init__(
        self,
        *,
        speaker_emb_dim: int = 192,
        **kwargs: Any,
    ) -> None:
        super().__init__(condition_dim=speaker_emb_dim, **kwargs)
        self.speaker_emb_dim = speaker_emb_dim

    def forward(
        self,
        spec: torch.Tensor,
        conditioning: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return super().forward(spec, conditioning)

    def stream_step(
        self,
        spec_frame: torch.Tensor,
        conditioning: torch.Tensor | None,
        state: DfNetStreamState,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, DfNetStreamState]:
        return super().stream_step(spec_frame, conditioning, state)


__all__ = ["PDfNet2"]
