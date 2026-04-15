"""Frozen native ECAPA-TDNN speaker encoder wrapper."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import pickle
from typing import Any

import torch
from torch import nn

from wulfenite.audio_features import compute_wespeaker_fbank_batch

from .ecapa_tdnn import (
    ECAPA_TDNN_GLOB_c1024,
    ECAPA_TDNN_GLOB_c512,
    ECAPA_TDNN_c1024,
    ECAPA_TDNN_c512,
    detect_ecapa_variant,
)


def _extract_embedding(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        emb = output
    elif isinstance(output, dict):
        for key in ("embedding", "embeddings", "spk_embedding", "speaker_embedding"):
            if key in output and isinstance(output[key], torch.Tensor):
                emb = output[key]
                break
        else:
            raise RuntimeError("Could not find an embedding tensor in backend output")
    elif isinstance(output, (list, tuple)):
        tensors = [item for item in output if isinstance(item, torch.Tensor)]
        emb = next((item for item in reversed(tensors) if item.dim() == 2), None)
        if emb is None:
            emb = next((item for item in reversed(tensors)), None)
        if emb is None:
            raise RuntimeError("Could not find an embedding tensor in backend output")
    else:
        raise RuntimeError(f"Unsupported backend output type: {type(output)!r}")

    if emb.dim() == 3:
        emb = emb.mean(dim=-1)
    if emb.dim() != 2:
        raise RuntimeError(f"Speaker embedding must be [B, D], got {tuple(emb.shape)}")
    return emb


class SpeakerEncoder(nn.Module):
    """Load and freeze a native ECAPA-TDNN speaker encoder."""

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
        self.sample_rate = 16000
        if backend is not None:
            self.backend = backend
            self.backend_input_kind = "waveform"
        else:
            if checkpoint_path is None:
                raise RuntimeError(
                    "checkpoint_path is required when backend is not provided."
                )
            self.backend = self._load_native_backend(checkpoint_path)
            self.backend_input_kind = "features"

        self.backend.eval()
        for param in self.backend.parameters():
            param.requires_grad = False

        inferred_dim = embedding_dim if embedding_dim is not None else self._infer_embedding_dim()
        self.embedding_dim = inferred_dim
        if inferred_dim != output_dim:
            raise ValueError(
                f"SpeakerEncoder is frozen, so backend embedding_dim must match "
                f"output_dim. Got embedding_dim={inferred_dim}, output_dim={output_dim}. "
                f"Use a speaker model with {output_dim}-D embeddings."
            )
        self.proj = nn.Identity()

    def _load_checkpoint_payload(self, checkpoint_path: str | Path) -> object:
        try:
            return torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        except (pickle.UnpicklingError, RuntimeError, ValueError):
            return torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    def _normalize_state_dict(self, payload: object) -> dict[str, torch.Tensor]:
        if not isinstance(payload, Mapping):
            raise ValueError(
                f"Unsupported speaker checkpoint payload type: {type(payload)!r}"
            )

        state_dict: dict[str, torch.Tensor] = {}
        for raw_key, value in payload.items():
            if not isinstance(raw_key, str):
                raise ValueError(f"Checkpoint contains a non-string key: {raw_key!r}")
            if not isinstance(value, torch.Tensor):
                raise ValueError(f"Checkpoint tensor {raw_key!r} is not a torch.Tensor")
            if raw_key in state_dict:
                raise ValueError(f"Duplicate checkpoint key after normalization: {raw_key}")
            state_dict[raw_key] = value

        if "blocks.0.conv.conv.weight" not in state_dict:
            raise ValueError(
                "Expected a SpeechBrain ECAPA-TDNN checkpoint with key "
                "'blocks.0.conv.conv.weight'."
            )

        return state_dict

    def _load_native_backend(self, checkpoint_path: str | Path) -> nn.Module:
        state_dict = self._normalize_state_dict(self._load_checkpoint_payload(checkpoint_path))
        variant = detect_ecapa_variant(state_dict)
        channels = int(variant["channels"])
        feat_dim = int(variant["feat_dim"])
        embed_dim = int(variant["embed_dim"])
        global_context_att = bool(variant["global_context_att"])
        emb_bn = bool(variant["emb_bn"])

        constructors = {
            (512, False): ECAPA_TDNN_c512,
            (512, True): ECAPA_TDNN_GLOB_c512,
            (1024, False): ECAPA_TDNN_c1024,
            (1024, True): ECAPA_TDNN_GLOB_c1024,
        }
        try:
            constructor = constructors[(channels, global_context_att)]
        except KeyError as exc:
            raise ValueError(
                "Unsupported ECAPA-TDNN configuration "
                f"(channels={channels}, global_context_att={global_context_att})"
            ) from exc

        model = constructor(
            feat_dim=feat_dim,
            embed_dim=embed_dim,
            pooling_func="ASTP",
            emb_bn=emb_bn,
        )
        model.load_state_dict(state_dict, strict=True)
        return model

    def _prepare_backend_input(self, waveform: torch.Tensor) -> torch.Tensor:
        if self.backend_input_kind == "waveform":
            return waveform
        return compute_wespeaker_fbank_batch(
            waveform,
            sample_rate=self.sample_rate,
            num_mel_bins=80,
            dither=0.0,
            mean_norm=True,
        )

    def _infer_embedding_dim(self) -> int:
        dummy_waveform = torch.zeros(1, 16000)
        backend_input = self._prepare_backend_input(dummy_waveform)
        with torch.no_grad():
            emb = _extract_embedding(self.backend(backend_input))
        return int(emb.size(-1))

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        if waveform.dim() != 2:
            raise ValueError(
                f"waveform must be [T] or [B, T], got {tuple(waveform.shape)}"
            )
        backend_input = self._prepare_backend_input(waveform)
        with torch.no_grad():
            emb = _extract_embedding(self.backend(backend_input))
        return self.proj(emb)


__all__ = ["SpeakerEncoder"]
