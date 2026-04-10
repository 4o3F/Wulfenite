"""Shared inference helpers for checkpoint-aware model loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from ..models import WulfeniteTSE
from ..training.checkpoint import load_checkpoint, peek_checkpoint_config


def _checkpoint_info_from_payload(
    payload: dict[str, Any],
    *,
    skipped_classifier_keys: list[str] | None = None,
) -> dict[str, Any]:
    info = {
        "epoch": payload.get("epoch", 0),
        "step": payload.get("step", 0),
        "config": payload.get("config"),
        "metrics": payload.get("metrics", {}),
        "wulfenite_version": payload.get("wulfenite_version"),
    }
    if skipped_classifier_keys is not None:
        info["skipped_classifier_keys"] = skipped_classifier_keys
    return info


def _load_learnable_checkpoint(
    checkpoint: Path,
    model: WulfeniteTSE,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load a learnable-encoder checkpoint into a classifier-free model."""
    payload = torch.load(str(checkpoint), map_location=device, weights_only=False)
    state = dict(payload["model_state_dict"])

    classifier_keys = [
        key for key in state
        if key.startswith("speaker_encoder.classifier.")
    ]
    for key in classifier_keys:
        del state[key]

    model.load_state_dict(state, strict=True)
    return _checkpoint_info_from_payload(
        payload,
        skipped_classifier_keys=classifier_keys,
    )


def build_model_from_checkpoint(
    checkpoint: Path,
    campplus_checkpoint: Path | None = None,
    use_learnable_encoder: bool | None = None,
    device: str | torch.device = "cpu",
) -> tuple[WulfeniteTSE, dict[str, Any]]:
    """Build and load the correct TSE model for a checkpoint."""
    cfg = peek_checkpoint_config(checkpoint)

    if use_learnable_encoder is not None:
        learnable = use_learnable_encoder
    elif "use_learnable_encoder" in cfg:
        learnable = bool(cfg["use_learnable_encoder"])
    elif campplus_checkpoint is not None:
        learnable = False
    else:
        raise RuntimeError(
            "Cannot determine encoder type from checkpoint. Pass "
            "--use-learnable-encoder for learnable checkpoints or "
            "--campplus-checkpoint for the frozen CAM++ path."
        )

    dev = torch.device(device)
    if learnable:
        model = WulfeniteTSE.from_learnable_dvector(num_speakers=None)
        info = _load_learnable_checkpoint(checkpoint, model, device="cpu")
    else:
        if campplus_checkpoint is None:
            raise RuntimeError(
                "Frozen CAM++ checkpoints require --campplus-checkpoint."
            )
        model = WulfeniteTSE.from_campplus_checkpoint(
            campplus_checkpoint,
            device="cpu",
        )
        info = load_checkpoint(checkpoint, model=model, map_location="cpu")

    model = model.to(dev).eval()
    return model, info


__all__ = [
    "build_model_from_checkpoint",
]
