"""Training configuration for Wulfenite."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TrainingConfig:
    """All training hyperparameters in one place."""

    # --- Data paths ---
    aishell1_root: Path | None = None
    aishell3_root: Path | None = None
    magicdata_root: Path | None = None
    cnceleb_root: Path | None = None
    noise_root: Path | None = None
    campplus_checkpoint: Path | None = None

    # --- Mixer ---
    segment_seconds: float = 4.0
    enrollment_seconds: float = 4.0
    snr_range_db: tuple[float, float] = (-5.0, 5.0)
    target_present_prob: float = 0.85
    transition_prob: float = 0.0
    transition_warmup_ratio: float = 0.0
    transition_ramp_ratio: float = 0.0
    transition_min_fraction: float = 0.25
    transition_min_target_rms: float = 0.01
    noise_snr_range_db: tuple[float, float] = (10.0, 25.0)
    noise_prob: float = 0.80
    reverb_prob: float = 0.85
    rir_pool_size: int = 1000

    # --- Optimization ---
    batch_size: int = 16
    epochs: int = 200
    samples_per_epoch: int = 20000
    val_samples: int = 500
    learning_rate: float = 5e-4
    encoder_lr: float = 1e-5
    weight_decay: float = 0.0
    use_plateau_scheduler: bool = True
    plateau_patience: int = 5
    plateau_factor: float = 0.5
    early_stopping_patience: int = 20
    grad_clip: float = 5.0

    # --- Separator architecture ---
    enc_channels: int = 4096
    bottleneck_channels: int = 256
    speaker_embed_dim: int = 192
    hidden_channels: int = 512
    num_repeats: int = 2
    r1_blocks: int = 3
    r2_blocks: int = 1
    s4d_state_dim: int = 32

    # --- Loss weights (matches LossWeights defaults) ---
    loss_sdr: float = 1.0
    loss_mr_stft: float = 1.0
    loss_absent: float = 0.5
    loss_presence: float = 0.1

    # --- DataLoader ---
    num_workers: int = 8
    prefetch_factor: int = 4
    val_speaker_ratio: float = 0.2

    # --- Output / logging ---
    out_dir: Path = field(default_factory=lambda: Path("./checkpoints"))
    log_interval: int = 50
    save_every_epoch: bool = True

    # --- Runtime ---
    device: str = "cuda"
    seed: int = 1234

    def __post_init__(self) -> None:
        if self.transition_warmup_ratio < 0.0:
            raise ValueError(
                "transition_warmup_ratio must be >= 0.0; got "
                f"{self.transition_warmup_ratio}"
            )
        if self.transition_ramp_ratio < 0.0:
            raise ValueError(
                "transition_ramp_ratio must be >= 0.0; got "
                f"{self.transition_ramp_ratio}"
            )
        if self.transition_warmup_ratio + self.transition_ramp_ratio > 1.0:
            raise ValueError(
                "transition_warmup_ratio + transition_ramp_ratio must be <= 1.0; "
                f"got {self.transition_warmup_ratio + self.transition_ramp_ratio}"
            )
