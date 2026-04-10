"""Shared inference helpers for checkpoint-aware model loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from ..models import WulfeniteTSE
from ..models.speakerbeam_ss import SpeakerBeamSSConfig
from ..training.checkpoint import peek_checkpoint_config


def _rebuild_separator_config(
    checkpoint_config: dict[str, Any],
) -> SpeakerBeamSSConfig | None:
    """Try to reconstruct SpeakerBeamSSConfig from saved checkpoint config.

    If the checkpoint was saved by the current training loop, its config
    dict contains all ``SpeakerBeamSSConfig`` field names. If any are
    missing (e.g. a very old checkpoint), we fall back to the current
    code defaults so that fresh checkpoints always load correctly.
    """
    if not checkpoint_config:
        return None  # no config saved → use current defaults

    # Map TrainingConfig field names that mirror SpeakerBeamSSConfig
    # (the training config doesn't store a nested separator config but
    # checkpoint metadata may contain separator-level overrides from
    # the future — for now, try the fields we know about)
    kwargs: dict[str, Any] = {}
    field_map = {
        "enc_channels": "enc_channels",
        "bottleneck_channels": "bottleneck_channels",
        "hidden_channels": "hidden_channels",
        "s4d_state_dim": "s4d_state_dim",
        "num_repeats": "num_repeats",
        "r1_blocks": "r1_blocks",
        "r2_blocks": "r2_blocks",
    }
    for ck_key, cfg_key in field_map.items():
        if ck_key in checkpoint_config:
            kwargs[cfg_key] = checkpoint_config[ck_key]
    if kwargs:
        return SpeakerBeamSSConfig(**kwargs)
    return None


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
    skipped_keys = classifier_keys + [
        key
        for key in state
        if key.startswith("speaker_encoder.backbone.")
        or key.startswith("speaker_encoder.to_separator.")
    ]
    for key in skipped_keys:
        del state[key]

    model.load_state_dict(state, strict=True)
    return _checkpoint_info_from_payload(
        payload,
        skipped_classifier_keys=skipped_keys,
    )


def _load_checkpoint_strict(
    checkpoint: Path,
    model: WulfeniteTSE,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load a checkpoint into a model whose architecture already matches."""
    payload = torch.load(str(checkpoint), map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return _checkpoint_info_from_payload(payload)


def build_model_from_checkpoint(
    checkpoint: Path,
    device: str | torch.device = "cpu",
) -> tuple[WulfeniteTSE, dict[str, Any]]:
    """Build and load a Wulfenite TSE model from a checkpoint."""
    cfg = peek_checkpoint_config(checkpoint)
    encoder_type = cfg.get("encoder_type", "learnable")

    # Rebuild separator config from checkpoint metadata when available,
    # so legacy checkpoints with different defaults (e.g. 512/192) load
    # correctly against the current code whose defaults are 4096/256.
    separator_config = _rebuild_separator_config(cfg)

    dev = torch.device(device)
    if encoder_type == "learnable":
        model = WulfeniteTSE.from_learnable_dvector(
            num_speakers=None,
            separator_config=separator_config,
        )
        try:
            info = _load_learnable_checkpoint(checkpoint, model, device="cpu")
        except RuntimeError as exc:
            raise RuntimeError(
                "Checkpoint is incompatible with the learnable d-vector TSE pipeline."
            ) from exc
    elif encoder_type in {"campplus-frozen", "campplus-finetune"}:
        model = WulfeniteTSE.from_campplus(
            campplus_checkpoint=None,
            separator_config=separator_config,
            freeze_backbone=encoder_type == "campplus-frozen",
        )
        try:
            info = _load_checkpoint_strict(checkpoint, model, device="cpu")
        except RuntimeError as exc:
            raise RuntimeError(
                "Checkpoint is incompatible with the CAM++ TSE pipeline."
            ) from exc
    else:
        raise RuntimeError(f"Unsupported encoder_type in checkpoint: {encoder_type}")

    model = model.to(dev).eval()
    return model, info


__all__ = [
    "build_model_from_checkpoint",
]
