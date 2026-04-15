"""Train DfNet or pDfNet2 from a TOML configuration file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import tomllib
from typing import Any, Literal

import torch

from wulfenite.data import (
    AudioEntry,
    PSEMixer,
    ReverbConfig,
    merge_speaker_dicts,
    scan_aishell1,
    scan_aishell3,
    scan_magicdata,
    scan_noise_dir,
)
from wulfenite.models import DfNet, PDfNet2, SpeakerEncoder
from wulfenite.training import TrainConfig, train_pdfnet2


ConfigDict = dict[str, Any]


def _load_config(config_path: str, overrides: list[str]) -> ConfigDict:
    with open(config_path, "rb") as handle:
        cfg = tomllib.load(handle)
    for override in overrides:
        key, _, value = override.partition("=")
        parts = key.strip().split(".")
        target = cfg
        for part in parts[:-1]:
            nested = target.setdefault(part, {})
            if not isinstance(nested, dict):
                raise ValueError(f"Override target {'.'.join(parts[:-1])} is not a table.")
            target = nested
        raw_value = value.strip()
        if raw_value.lower() in ("true", "false"):
            target[parts[-1]] = raw_value.lower() == "true"
        else:
            try:
                target[parts[-1]] = int(raw_value)
            except ValueError:
                try:
                    target[parts[-1]] = float(raw_value)
                except ValueError:
                    target[parts[-1]] = raw_value
    return cfg


def _config_table(config: ConfigDict, key: str) -> ConfigDict:
    table = config.get(key)
    if not isinstance(table, dict):
        raise ValueError(f"Missing required [{key}] table.")
    return table


def _optional_path(value: object) -> Path | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"Expected a path-like string, got {type(value)!r}.")
    if not value:
        return None
    return Path(value).expanduser()


def _required_path(table: ConfigDict, key: str) -> Path:
    path = _optional_path(table.get(key))
    if path is None:
        raise ValueError(f"Missing required path: {key}")
    return path


def _scan_dataset_splits(data_cfg: ConfigDict) -> tuple[dict[str, list[AudioEntry]], dict[str, list[AudioEntry]]]:
    train_pools: list[dict[str, list[AudioEntry]]] = []
    val_pools: list[dict[str, list[AudioEntry]]] = []

    aishell1_root = _optional_path(data_cfg.get("aishell1_root"))
    if aishell1_root is not None:
        train_pools.append(scan_aishell1(aishell1_root, splits=("train",)))
        try:
            val_pools.append(scan_aishell1(aishell1_root, splits=("dev",)))
        except RuntimeError:
            pass

    aishell3_root = _optional_path(data_cfg.get("aishell3_root"))
    if aishell3_root is not None:
        train_pools.append(scan_aishell3(aishell3_root, splits=("train",)))
        try:
            val_pools.append(scan_aishell3(aishell3_root, splits=("test",)))
        except RuntimeError:
            pass

    magicdata_root = _optional_path(data_cfg.get("magicdata_root"))
    if magicdata_root is not None:
        train_pools.append(scan_magicdata(magicdata_root, splits=("train",)))
        try:
            val_pools.append(scan_magicdata(magicdata_root, splits=("dev",)))
        except RuntimeError:
            pass

    if not train_pools:
        raise ValueError("At least one dataset root must be configured under [data].")

    train_speakers = merge_speaker_dicts(*train_pools)
    val_speakers = merge_speaker_dicts(*val_pools) if val_pools else {}
    return train_speakers, val_speakers


def _split_speakers(
    speakers: dict[str, list[AudioEntry]],
    *,
    seed: int,
    val_fraction: float = 0.1,
) -> tuple[dict[str, list[AudioEntry]], dict[str, list[AudioEntry]]]:
    speaker_ids = sorted(speakers)
    if len(speaker_ids) < 2:
        return speakers, speakers
    rng = random.Random(seed)
    rng.shuffle(speaker_ids)
    val_count = max(1, int(round(len(speaker_ids) * val_fraction)))
    val_count = min(val_count, len(speaker_ids) - 1)
    val_ids = set(speaker_ids[:val_count])
    train = {speaker_id: speakers[speaker_id] for speaker_id in speaker_ids if speaker_id not in val_ids}
    val = {speaker_id: speakers[speaker_id] for speaker_id in speaker_ids if speaker_id in val_ids}
    return train, val


def _build_reverb(data_cfg: ConfigDict) -> tuple[ReverbConfig, float]:
    reverb_cfg = data_cfg.get("reverb", {})
    if not isinstance(reverb_cfg, dict):
        raise ValueError("[data.reverb] must be a table.")
    sample_rate = int(data_cfg.get("sample_rate", 16000))
    preset = reverb_cfg.get("preset", "mixed")
    if not isinstance(preset, str):
        raise ValueError("[data.reverb].preset must be a string.")
    probability = float(reverb_cfg.get("probability", 0.3))
    return ReverbConfig.from_preset(preset, sample_rate=sample_rate), probability


def _build_train_config(training_cfg: ConfigDict) -> TrainConfig:
    loss_cfg = training_cfg.get("loss", {})
    if not isinstance(loss_cfg, dict):
        raise ValueError("[training.loss] must be a table.")
    batch_size = int(training_cfg.get("batch_size", 8))
    device = training_cfg.get("device", "auto")
    if not isinstance(device, str):
        raise ValueError("[training].device must be a string.")
    scheduler = training_cfg.get("lr_scheduler", "cosine")
    if scheduler not in ("none", "cosine"):
        raise ValueError("[training].lr_scheduler must be 'none' or 'cosine'.")
    return TrainConfig(
        learning_rate=float(training_cfg.get("learning_rate", 1e-3)),
        weight_decay=float(training_cfg.get("weight_decay", 0.0)),
        max_epochs=int(training_cfg.get("max_epochs", 200)),
        batch_size_start=batch_size,
        batch_size_end=batch_size,
        batch_size_ramp_epochs=1,
        grad_clip_norm=float(training_cfg.get("grad_clip_norm", 5.0)),
        patience=int(training_cfg.get("patience", 20)),
        num_workers=int(training_cfg.get("num_workers", 0)),
        checkpoint_dir=Path(str(training_cfg.get("checkpoint_dir", "checkpoints"))),
        device=None if device == "auto" else device,
        lr_scheduler=scheduler,
        lr_warmup_epochs=int(training_cfg.get("lr_warmup_epochs", 5)),
        lr_min_ratio=float(training_cfg.get("lr_min_ratio", 0.01)),
        lambda_spec=float(loss_cfg.get("lambda_spec", 1000.0)),
        lambda_mr=float(loss_cfg.get("lambda_mr", 500.0)),
        lambda_os=float(loss_cfg.get("lambda_os", 500.0)),
    )


def _build_model(model_cfg: ConfigDict) -> tuple[DfNet | PDfNet2, SpeakerEncoder | None]:
    model_type = model_cfg.get("type", "pdfnet2")
    if not isinstance(model_type, str):
        raise ValueError("[model].type must be a string.")
    wespeaker_checkpoint = model_cfg.get("wespeaker_checkpoint")
    checkpoint_path = _optional_path(wespeaker_checkpoint)
    if model_type == "dfnet":
        return DfNet(), None
    if model_type == "pdfnet2":
        speaker_encoder = SpeakerEncoder(checkpoint_path=checkpoint_path)
        return PDfNet2(), speaker_encoder
    raise ValueError(f"Unsupported model.type: {model_type}")


def _resume_model(model: DfNet | PDfNet2, resume_path: Path) -> None:
    checkpoint = torch.load(resume_path, map_location="cpu", weights_only=True)
    state_dict = checkpoint.get("model_state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError(f"Checkpoint at {resume_path} does not contain model_state_dict.")
    model.load_state_dict(state_dict, strict=True)
    epoch = checkpoint.get("epoch")
    print(f"[resume] loaded model weights from {resume_path} (epoch={epoch})")
    if "optimizer_state_dict" in checkpoint:
        print("[resume] optimizer/scheduler state is not restored by the current training API.")


def _print_config(cfg: ConfigDict) -> None:
    print("[config] resolved configuration:")
    print(json.dumps(cfg, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to a training TOML file.")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a TOML value, for example training.batch_size=16",
    )
    args = parser.parse_args()

    cfg = _load_config(args.config, args.override)
    _print_config(cfg)

    data_cfg = _config_table(cfg, "data")
    model_cfg = _config_table(cfg, "model")
    training_cfg = _config_table(cfg, "training")

    train_speakers, val_speakers = _scan_dataset_splits(data_cfg)
    seed = int(data_cfg.get("seed", 0))
    if not val_speakers:
        train_speakers, val_speakers = _split_speakers(train_speakers, seed=seed)
        print("[data] no dedicated validation split found; using a speaker-disjoint fallback split.")

    noise_root = _optional_path(data_cfg.get("noise_root"))
    noises = scan_noise_dir(noise_root) if noise_root is not None else []
    reverb_config, reverb_probability = _build_reverb(data_cfg)

    sample_rate = int(data_cfg.get("sample_rate", 16000))
    segment_length = int(round(float(data_cfg.get("segment_seconds", 4.0)) * sample_rate))
    enrollment_length = int(round(float(data_cfg.get("enrollment_seconds", 6.0)) * sample_rate))

    train_dataset = PSEMixer(
        train_speakers,
        noises,
        epoch_size=int(data_cfg.get("epoch_size", 5000)),
        segment_length=segment_length,
        enrollment_length=enrollment_length,
        sample_rate=sample_rate,
        reverb_config=reverb_config,
        reverb_probability=reverb_probability,
        seed=seed,
    )
    val_dataset = PSEMixer(
        val_speakers,
        noises,
        epoch_size=int(data_cfg.get("val_size", 500)),
        segment_length=segment_length,
        enrollment_length=enrollment_length,
        sample_rate=sample_rate,
        reverb_config=reverb_config,
        reverb_probability=reverb_probability,
        seed=seed + 1,
    )

    model, speaker_encoder = _build_model(model_cfg)
    resume = _optional_path(training_cfg.get("resume"))
    if resume is not None:
        _resume_model(model, resume)

    train_config = _build_train_config(training_cfg)

    print(
        f"[data] train speakers={len(train_speakers)} val speakers={len(val_speakers)} "
        f"noise files={len(noises)}"
    )
    print(f"[model] training {type(model).__name__}")
    history = train_pdfnet2(
        model,
        train_dataset,
        val_dataset,
        train_config,
        speaker_encoder=speaker_encoder,
    )
    if not history:
        print("[summary] training finished with no history records.")
        return
    last = history[-1]
    best_val = min(record["val_loss"] for record in history)
    print(
        "[summary] "
        f"epochs={len(history)} last_train_loss={last['train_loss']:.4f} "
        f"last_val_loss={last['val_loss']:.4f} best_val_loss={best_val:.4f}"
    )


if __name__ == "__main__":
    main()
