"""Training configuration for Wulfenite.

A single dataclass captures every knob the training loop reads. The
defaults match the recipe in ``docs/architecture.md`` section 5–6 and
``docs/TRAIN.md`` section 7.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TrainingConfig:
    """All training hyperparameters in one place."""

    # --- Data paths ---
    aishell1_root: Path | None = None
    aishell3_root: Path | None = None
    noise_root: Path | None = None
    campplus_checkpoint: Path | None = None

    # --- Mixer ---
    segment_seconds: float = 4.0
    enrollment_seconds: float = 4.0
    snr_range_db: tuple[float, float] = (-5.0, 5.0)
    target_present_prob: float = 0.85
    noise_snr_range_db: tuple[float, float] = (10.0, 25.0)
    noise_prob: float = 0.80
    reverb_prob: float = 0.85

    # --- Optimization ---
    batch_size: int = 16
    epochs: int = 50
    samples_per_epoch: int = 20000
    val_samples: int = 500
    lr: float = 1e-3
    weight_decay: float = 1e-5
    warmup_ratio: float = 0.03
    grad_clip: float = 5.0

    # --- Loss weights (matches LossWeights defaults) ---
    loss_sdr: float = 1.0
    loss_mr_stft: float = 1.0
    loss_absent: float = 1.0
    loss_presence: float = 0.1

    # --- DataLoader ---
    num_workers: int = 8
    prefetch_factor: int = 4

    # --- Output / logging ---
    out_dir: Path = field(default_factory=lambda: Path("./checkpoints"))
    log_interval: int = 50
    save_every_epoch: bool = True

    # --- Runtime ---
    device: str = "cuda"
    seed: int = 1234
