"""Frozen WeSpeaker ECAPA-TDNN wrapper."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn


def _extract_embedding(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        emb = output
    elif isinstance(output, dict):
        for key in ("embedding", "embeddings", "spk_embedding", "speaker_embedding"):
            if key in output and isinstance(output[key], torch.Tensor):
                emb = output[key]
                break
        else:
            raise RuntimeError("Could not find an embedding tensor in WeSpeaker output")
    elif isinstance(output, (list, tuple)):
        emb = next((item for item in output if isinstance(item, torch.Tensor)), None)
        if emb is None:
            raise RuntimeError("Could not find an embedding tensor in WeSpeaker output")
    else:
        raise RuntimeError(f"Unsupported WeSpeaker output type: {type(output)!r}")

    if emb.dim() == 3:
        emb = emb.mean(dim=-1)
    if emb.dim() != 2:
        raise RuntimeError(f"Speaker embedding must be [B, D], got {tuple(emb.shape)}")
    return emb


class SpeakerEncoder(nn.Module):
    """Load and freeze a WeSpeaker-compatible speaker encoder."""

    def __init__(
        self,
        *,
        checkpoint_path: str | Path | None = None,
        backend: nn.Module | None = None,
        embedding_dim: int | None = None,
        output_dim: int = 192,
    ) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.backend = backend if backend is not None else self._load_wespeaker_backend(checkpoint_path)
        self.backend.eval()
        for param in self.backend.parameters():
            param.requires_grad = False

        inferred_dim = embedding_dim if embedding_dim is not None else self._infer_embedding_dim()
        self.embedding_dim = inferred_dim
        if inferred_dim != output_dim:
            raise ValueError(
                f"SpeakerEncoder is frozen, so backend embedding_dim must match "
                f"output_dim. Got embedding_dim={inferred_dim}, output_dim={output_dim}. "
                f"Use a WeSpeaker model with {output_dim}-D embeddings."
            )
        self.proj = nn.Identity()

    def _load_wespeaker_backend(self, checkpoint_path: str | Path | None) -> nn.Module:
        try:
            import wespeaker  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "wespeaker is not installed. Pass a backend module explicitly or "
                "install a compatible WeSpeaker package before using SpeakerEncoder."
            ) from exc

        path_str = str(checkpoint_path) if checkpoint_path is not None else None
        loader_names = ("load_model_local", "load_model")
        for loader_name in loader_names:
            loader = getattr(wespeaker, loader_name, None)
            if loader is None:
                continue
            model = loader(path_str) if path_str is not None else loader()
            if isinstance(model, nn.Module):
                return model
            candidate = getattr(model, "model", None)
            if isinstance(candidate, nn.Module):
                return candidate
        raise RuntimeError(
            "Unable to construct a WeSpeaker backend from the installed package. "
            "Pass a backend module explicitly to SpeakerEncoder."
        )

    def _infer_embedding_dim(self) -> int:
        dummy = torch.zeros(1, 16000)
        with torch.no_grad():
            emb = _extract_embedding(self.backend(dummy))
        return int(emb.size(-1))

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        if waveform.dim() != 2:
            raise ValueError(
                f"waveform must be [T] or [B, T], got {tuple(waveform.shape)}"
            )
        with torch.no_grad():
            emb = _extract_embedding(self.backend(waveform))
        return self.proj(emb)


__all__ = ["SpeakerEncoder"]
