"""Shared training configuration for the PSE stack."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


@dataclass(slots=True)
class TrainConfig:
    """Training hyper-parameters shared by pDFNet2 and TinyECAPA."""

    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    max_epochs: int = 200
    batch_size_start: int = 8
    batch_size_end: int = 128
    batch_size_ramp_epochs: int = 20
    grad_clip_norm: float = 5.0
    patience: int = 20
    num_workers: int = 0
    checkpoint_dir: Path = field(default_factory=lambda: Path("checkpoints"))
    device: str | None = None
    log_interval: int = 10
    max_steps_per_epoch: int | None = None
    lr_scheduler: Literal["none", "cosine"] = "cosine"
    lr_warmup_epochs: int = 5
    lr_min_ratio: float = 0.01
    lambda_spec: float = 1e3
    lambda_mr: float = 5e2
    lambda_os: float = 5e2
    tiny_ecapa_temperature_init: float = 10.0
    tiny_ecapa_apply_augmentation: bool = True
    tiny_ecapa_reverb_probability: float = 0.5
    tiny_ecapa_noise_snr_range: tuple[float, float] = (0.0, 20.0)
    tiny_ecapa_chunk_seconds: float = 1.0
    tiny_ecapa_chunk_overlap: float = 0.5


__all__ = ["TrainConfig"]
