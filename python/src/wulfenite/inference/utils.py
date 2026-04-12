"""Shared inference helpers for checkpoint-aware model loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from ..models import WulfeniteTSE
from ..models.speakerbeam_ss import SpeakerBeamSSConfig


def _rebuild_separator_config(
    checkpoint_config: dict[str, Any],
) -> SpeakerBeamSSConfig | None:
    """Try to reconstruct SpeakerBeamSSConfig from saved checkpoint config."""
    if not checkpoint_config:
        return None

    kwargs: dict[str, Any] = {}
    field_map = {
        "enc_channels": "enc_channels",
        "bottleneck_channels": "bottleneck_channels",
        "speaker_embed_dim": "speaker_embed_dim",
        "hidden_channels": "hidden_channels",
        "s4d_state_dim": "s4d_state_dim",
        "r1_repeats": "r1_repeats",
        "r2_repeats": "r2_repeats",
        "conv_blocks_per_repeat": "conv_blocks_per_repeat",
        "s4d_ffn_multiplier": "s4d_ffn_multiplier",
        "target_presence_head": "target_presence_head",
    }
    for ck_key, cfg_key in field_map.items():
        if ck_key in checkpoint_config:
            kwargs[cfg_key] = checkpoint_config[ck_key]
    legacy_stack_map = {
        "num_repeats": "r1_repeats",
        "r1_blocks": "conv_blocks_per_repeat",
        "r2_blocks": "r2_repeats",
    }
    for ck_key, cfg_key in legacy_stack_map.items():
        if cfg_key not in kwargs and ck_key in checkpoint_config:
            kwargs[cfg_key] = checkpoint_config[ck_key]
    if kwargs:
        return SpeakerBeamSSConfig(**kwargs)
    return None


def _checkpoint_info_from_payload(
    payload: dict[str, Any],
    *,
    skipped_legacy_keys: list[str] | None = None,
    skipped_incompatible_keys: list[str] | None = None,
    missing_after_load: list[str] | None = None,
    unexpected_after_load: list[str] | None = None,
) -> dict[str, Any]:
    info = {
        "epoch": payload.get("epoch", 0),
        "step": payload.get("step", 0),
        "config": payload.get("config"),
        "metrics": payload.get("metrics", {}),
        "wulfenite_version": payload.get("wulfenite_version"),
    }
    if skipped_legacy_keys is not None:
        info["skipped_legacy_keys"] = skipped_legacy_keys
    if skipped_incompatible_keys is not None:
        info["skipped_incompatible_keys"] = skipped_incompatible_keys
    if missing_after_load is not None:
        info["missing_after_load"] = missing_after_load
    if unexpected_after_load is not None:
        info["unexpected_after_load"] = unexpected_after_load
    return info


def _load_state_dict_compat(
    payload: dict[str, Any],
    model: WulfeniteTSE,
) -> dict[str, Any]:
    state = dict(payload["model_state_dict"])
    if any(key.startswith("speaker_encoder.frame.") for key in state):
        raise RuntimeError(
            "Legacy learnable d-vector checkpoints are unsupported by the "
            "Phase 3 CAM++-only loader."
        )

    model_state = model.state_dict()
    loadable_state: dict[str, torch.Tensor] = {}
    skipped_legacy_keys: list[str] = []
    skipped_incompatible_keys: list[str] = []
    legacy_prefixes = (
        "speaker_encoder.to_separator.",
        "speaker_encoder.classifier.",
    )
    for key, value in state.items():
        if key.startswith(legacy_prefixes):
            skipped_legacy_keys.append(key)
            continue
        if key not in model_state or model_state[key].shape != value.shape:
            skipped_incompatible_keys.append(key)
            continue
        loadable_state[key] = value

    incompatible = model.load_state_dict(loadable_state, strict=False)
    missing_after_load = list(incompatible.missing_keys)
    unexpected_after_load = list(incompatible.unexpected_keys)

    # Keys that were intentionally skipped (legacy adapter/classifier or
    # shape-incompatible separator parameters) are expected to be missing. Any *other*
    # missing key means the checkpoint is materially incomplete — refuse
    # to return a model with randomly-initialized required weights.
    allowed_missing = {
        key for key in skipped_incompatible_keys if key in model_state
    }
    disallowed_missing = sorted(
        key for key in missing_after_load if key not in allowed_missing
    )
    if disallowed_missing or unexpected_after_load:
        raise RuntimeError(
            "Checkpoint is only partially compatible with the Phase 3 "
            "CAM++-only loader; refusing to continue with randomly "
            f"initialized weights. missing={disallowed_missing} "
            f"unexpected={unexpected_after_load}"
        )

    return _checkpoint_info_from_payload(
        payload,
        skipped_legacy_keys=skipped_legacy_keys,
        skipped_incompatible_keys=skipped_incompatible_keys,
        missing_after_load=missing_after_load,
        unexpected_after_load=unexpected_after_load,
    )


def build_model_from_checkpoint(
    checkpoint: Path,
    device: str | torch.device = "cpu",
) -> tuple[WulfeniteTSE, dict[str, Any]]:
    """Build and load a Wulfenite TSE model from a checkpoint."""
    payload = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    cfg = payload.get("config")
    cfg = cfg if isinstance(cfg, dict) else {}

    separator_config = _rebuild_separator_config(cfg)
    model = WulfeniteTSE.from_campplus(
        campplus_checkpoint=None,
        separator_config=separator_config,
    )
    info = _load_state_dict_compat(payload, model)
    model = model.to(torch.device(device)).eval()
    return model, info


__all__ = [
    "build_model_from_checkpoint",
]
