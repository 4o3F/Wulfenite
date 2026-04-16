"""Train DfNet or pDfNet2 from a TOML configuration file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import tomllib
from typing import Any

import torch

from wulfenite.data import (
    AudioEntry,
    NoiseEntry,
    PSEMixer,
    ReverbConfig,
    scan_aishell1,
    scan_aishell3,
    scan_magicdata,
    scan_noise_dir,
    scan_noise_dirs,
)
from wulfenite.models import DfNet, PDfNet2, SpeakerEncoder
from wulfenite.training import TrainConfig, train_pdfnet2


ConfigDict = dict[str, Any]
SpeakerDatasets = dict[str, dict[str, list[AudioEntry]]]
NoisePools = list[NoiseEntry] | dict[str, list[NoiseEntry]]


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


def _optional_table(config: ConfigDict, key: str) -> ConfigDict | None:
    table = config.get(key)
    if table is None:
        return None
    if not isinstance(table, dict):
        raise ValueError(f"[{key}] must be a table.")
    return table


def _optional_path(value: object) -> Path | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"Expected a path-like string, got {type(value)!r}.")
    if not value:
        return None
    return Path(value).expanduser()


def _optional_float_dict(
    value: object,
    *,
    name: str,
) -> dict[str, float] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a table of numeric values.")
    parsed: dict[str, float] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise ValueError(f"{name} keys must be strings, got {type(key)!r}.")
        parsed[key] = float(item)
    return parsed


def _parse_range(
    value: object,
    *,
    name: str,
    default: tuple[float, float],
) -> tuple[float, float]:
    if value is None:
        return default
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{name} must be a two-item array.")
    return float(value[0]), float(value[1])


def _parse_bucket_table(
    table: object,
) -> tuple[tuple[float, float, float], ...] | None:
    if table is None:
        return None
    if not isinstance(table, list):
        raise ValueError("bucket table must be an array of tables")
    buckets: list[tuple[float, float, float]] = []
    for item in table:
        if not isinstance(item, dict):
            raise ValueError("each bucket entry must be a table")
        if "weight" not in item or "min_db" not in item or "max_db" not in item:
            raise ValueError("each bucket entry must define weight, min_db, and max_db")
        buckets.append(
            (
                float(item["weight"]),
                float(item["min_db"]),
                float(item["max_db"]),
            )
        )
    return tuple(buckets)


def _scan_dataset_splits(data_cfg: ConfigDict) -> tuple[SpeakerDatasets, SpeakerDatasets]:
    train_pools: SpeakerDatasets = {}
    val_pools: SpeakerDatasets = {}

    aishell1_root = _optional_path(data_cfg.get("aishell1_root"))
    if aishell1_root is not None:
        train_pools["aishell1"] = scan_aishell1(aishell1_root, splits=("train",))
        try:
            val_pools["aishell1"] = scan_aishell1(aishell1_root, splits=("dev",))
        except RuntimeError:
            pass

    aishell3_root = _optional_path(data_cfg.get("aishell3_root"))
    if aishell3_root is not None:
        train_pools["aishell3"] = scan_aishell3(aishell3_root, splits=("train",))
        try:
            val_pools["aishell3"] = scan_aishell3(aishell3_root, splits=("test",))
        except RuntimeError:
            pass

    magicdata_root = _optional_path(data_cfg.get("magicdata_root"))
    if magicdata_root is not None:
        train_pools["magicdata"] = scan_magicdata(magicdata_root, splits=("train",))
        try:
            val_pools["magicdata"] = scan_magicdata(magicdata_root, splits=("dev",))
        except RuntimeError:
            pass

    if not train_pools:
        raise ValueError("At least one dataset root must be configured under [data].")
    return train_pools, val_pools


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


def _split_dataset_speakers(
    datasets: SpeakerDatasets,
    *,
    seed: int,
    val_fraction: float = 0.1,
) -> tuple[SpeakerDatasets, SpeakerDatasets]:
    train_datasets: SpeakerDatasets = {}
    val_datasets: SpeakerDatasets = {}
    for offset, dataset_id in enumerate(sorted(datasets)):
        train_split, val_split = _split_speakers(
            datasets[dataset_id],
            seed=seed + offset,
            val_fraction=val_fraction,
        )
        train_datasets[dataset_id] = train_split
        val_datasets[dataset_id] = val_split
    return train_datasets, val_datasets


def _build_reverb(data_cfg: ConfigDict) -> tuple[ReverbConfig, float]:
    reverb_cfg = _optional_table(data_cfg, "reverb") or {}
    sample_rate = int(data_cfg.get("sample_rate", 16000))
    preset = reverb_cfg.get("preset", "mixed")
    if not isinstance(preset, str):
        raise ValueError("[data.reverb].preset must be a string.")
    probability = float(reverb_cfg.get("probability", 0.3))
    return ReverbConfig.from_preset(preset, sample_rate=sample_rate), probability


def _scan_noise_inputs(data_cfg: ConfigDict) -> NoisePools:
    noise_cfg = _optional_table(data_cfg, "noise") or {}
    min_duration_seconds = float(noise_cfg.get("min_duration_seconds", 1.0))
    category_roots = noise_cfg.get("category_roots")
    if category_roots is not None:
        if not isinstance(category_roots, dict):
            raise ValueError("[data.noise].category_roots must be a table of paths.")
        roots: dict[str, Path | str] = {}
        for category, raw_path in category_roots.items():
            if not isinstance(category, str):
                raise ValueError("[data.noise].category_roots keys must be strings.")
            path = _optional_path(raw_path)
            if path is None:
                raise ValueError(
                    f"[data.noise].category_roots.{category} must be a non-empty path."
                )
            roots[category] = path
        return scan_noise_dirs(roots, min_duration_seconds=min_duration_seconds)

    noise_root = _optional_path(data_cfg.get("noise_root"))
    if noise_root is None:
        return []
    return scan_noise_dir(noise_root, min_duration_seconds=min_duration_seconds)


def _build_mixer_kwargs(
    data_cfg: ConfigDict,
    *,
    epoch_size: int,
    segment_length: int,
    enrollment_length: int,
    sample_rate: int,
    reverb_config: ReverbConfig,
    reverb_probability: float,
    seed: int,
) -> dict[str, object]:
    sampling_cfg = _optional_table(data_cfg, "sampling") or {}
    scene_cfg = _optional_table(data_cfg, "scene") or {}
    reverb_cfg = _optional_table(data_cfg, "reverb") or {}
    augmentation_cfg = _optional_table(data_cfg, "augmentation") or {}
    noise_cfg = _optional_table(data_cfg, "noise") or {}
    mixing_rms_mode = str(augmentation_cfg.get("mixing_rms_mode", "full"))
    if mixing_rms_mode not in ("full", "active"):
        raise ValueError("[data.augmentation].mixing_rms_mode must be 'full' or 'active'.")
    return {
        "dataset_weights": _optional_float_dict(
            sampling_cfg.get("dataset_weights"),
            name="[data.sampling].dataset_weights",
        ),
        "interferer_same_dataset_probability": float(
            sampling_cfg.get("interferer_same_dataset_probability", 1.0)
        ),
        "scene_weights": _optional_float_dict(
            scene_cfg.get("weights"),
            name="[data.scene].weights",
        ),
        "snr_buckets": _parse_bucket_table(scene_cfg.get("snr_buckets")),
        "sir_buckets": _parse_bucket_table(scene_cfg.get("sir_buckets")),
        "reverb_room_weights": _optional_float_dict(
            reverb_cfg.get("room_family_weights"),
            name="[data.reverb].room_family_weights",
        ),
        "gain_probability": float(augmentation_cfg.get("gain_probability", 0.3)),
        "gain_range_db": _parse_range(
            augmentation_cfg.get("gain_range_db"),
            name="[data.augmentation].gain_range_db",
            default=(-6.0, 6.0),
        ),
        "bandwidth_limit_probability": float(
            augmentation_cfg.get("bandwidth_limit_probability", 0.2)
        ),
        "bandwidth_cutoff_range_hz": _parse_range(
            augmentation_cfg.get("bandwidth_cutoff_range_hz"),
            name="[data.augmentation].bandwidth_cutoff_range_hz",
            default=(4000.0, 7000.0),
        ),
        "mixing_rms_mode": mixing_rms_mode,
        "activity_frame_ms": float(augmentation_cfg.get("activity_frame_ms", 32.0)),
        "activity_threshold_db": float(augmentation_cfg.get("activity_threshold_db", -40.0)),
        "noise_category_weights": _optional_float_dict(
            noise_cfg.get("category_weights"),
            name="[data.noise].category_weights",
        ),
        "epoch_size": epoch_size,
        "segment_length": segment_length,
        "enrollment_length": enrollment_length,
        "sample_rate": sample_rate,
        "reverb_config": reverb_config,
        "reverb_probability": reverb_probability,
        "seed": seed,
    }


def _build_train_config(training_cfg: ConfigDict) -> TrainConfig:
    defaults = TrainConfig()
    loss_cfg = training_cfg.get("loss", {})
    if not isinstance(loss_cfg, dict):
        raise ValueError("[training.loss] must be a table.")
    device = training_cfg.get("device", "auto")
    if not isinstance(device, str):
        raise ValueError("[training].device must be a string.")
    scheduler = training_cfg.get("lr_scheduler", "cosine")
    if scheduler not in ("none", "cosine"):
        raise ValueError("[training].lr_scheduler must be 'none' or 'cosine'.")
    has_batch_schedule = any(
        key in training_cfg
        for key in ("batch_size_start", "batch_size_end", "batch_size_ramp_epochs")
    )
    if has_batch_schedule:
        batch_size_start = int(
            training_cfg.get("batch_size_start", training_cfg.get("batch_size", defaults.batch_size_start))
        )
        batch_size_end = int(
            training_cfg.get("batch_size_end", training_cfg.get("batch_size", defaults.batch_size_end))
        )
        batch_size_ramp_epochs = int(
            training_cfg.get("batch_size_ramp_epochs", defaults.batch_size_ramp_epochs)
        )
    else:
        batch_size = int(training_cfg.get("batch_size", defaults.batch_size_start))
        batch_size_start = batch_size
        batch_size_end = batch_size
        batch_size_ramp_epochs = 1
    return TrainConfig(
        learning_rate=float(training_cfg.get("learning_rate", defaults.learning_rate)),
        weight_decay=float(training_cfg.get("weight_decay", defaults.weight_decay)),
        max_epochs=int(training_cfg.get("max_epochs", defaults.max_epochs)),
        batch_size_start=batch_size_start,
        batch_size_end=batch_size_end,
        batch_size_ramp_epochs=batch_size_ramp_epochs,
        grad_clip_norm=float(training_cfg.get("grad_clip_norm", defaults.grad_clip_norm)),
        patience=int(training_cfg.get("patience", defaults.patience)),
        num_workers=int(training_cfg.get("num_workers", 0)),
        checkpoint_dir=Path(str(training_cfg.get("checkpoint_dir", "checkpoints"))),
        device=None if device == "auto" else device,
        lr_scheduler=scheduler,
        lr_warmup_epochs=int(training_cfg.get("lr_warmup_epochs", defaults.lr_warmup_epochs)),
        lr_warmup_start=float(training_cfg.get("lr_warmup_start", defaults.lr_warmup_start)),
        lr_min_ratio=float(training_cfg.get("lr_min_ratio", defaults.lr_min_ratio)),
        lambda_spec=float(loss_cfg.get("lambda_spec", defaults.lambda_spec)),
        lambda_mr=float(loss_cfg.get("lambda_mr", defaults.lambda_mr)),
        lambda_os=float(loss_cfg.get("lambda_os", defaults.lambda_os)),
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


def _count_speakers(datasets: SpeakerDatasets) -> int:
    return sum(len(speakers) for speakers in datasets.values())


def _count_noise_entries(noises: NoisePools) -> int:
    if isinstance(noises, list):
        return len(noises)
    return sum(len(entries) for entries in noises.values())


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

    train_datasets, val_datasets = _scan_dataset_splits(data_cfg)
    seed = int(data_cfg.get("seed", 0))
    if not val_datasets:
        train_datasets, val_datasets = _split_dataset_speakers(train_datasets, seed=seed)
        print("[data] no dedicated validation split found; using a speaker-disjoint fallback split.")

    noises = _scan_noise_inputs(data_cfg)
    reverb_config, reverb_probability = _build_reverb(data_cfg)

    sample_rate = int(data_cfg.get("sample_rate", 16000))
    segment_length = int(round(float(data_cfg.get("segment_seconds", 4.0)) * sample_rate))
    enrollment_length = int(round(float(data_cfg.get("enrollment_seconds", 6.0)) * sample_rate))

    train_dataset = PSEMixer(
        noises=noises,
        datasets=train_datasets,
        **_build_mixer_kwargs(
            data_cfg,
            epoch_size=int(data_cfg.get("epoch_size", 5000)),
            segment_length=segment_length,
            enrollment_length=enrollment_length,
            sample_rate=sample_rate,
            reverb_config=reverb_config,
            reverb_probability=reverb_probability,
            seed=seed,
        ),
    )
    val_dataset = PSEMixer(
        noises=noises,
        datasets=val_datasets,
        **_build_mixer_kwargs(
            data_cfg,
            epoch_size=int(data_cfg.get("val_size", 500)),
            segment_length=segment_length,
            enrollment_length=enrollment_length,
            sample_rate=sample_rate,
            reverb_config=reverb_config,
            reverb_probability=reverb_probability,
            seed=seed + 1,
        ),
    )

    model, speaker_encoder = _build_model(model_cfg)
    resume = _optional_path(training_cfg.get("resume"))
    if resume is not None:
        _resume_model(model, resume)

    train_config = _build_train_config(training_cfg)

    print(
        f"[data] train speakers={_count_speakers(train_datasets)} "
        f"val speakers={_count_speakers(val_datasets)} "
        f"noise files={_count_noise_entries(noises)}"
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
