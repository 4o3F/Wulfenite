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
    segment_seconds: float = 8.0
    enrollment_seconds: float = 4.0
    snr_range_db: tuple[float, float] = (-5.0, 5.0)
    composition_mode: str = "clip_composer"
    family_multiturn_weight: float = 0.60
    family_overlap_heavy_weight: float = 0.25
    family_hard_negative_weight: float = 0.15
    min_events: int = 4
    max_events: int = 8
    min_event_seconds: float = 0.30
    max_event_seconds: float = 1.20
    crossfade_ms: float = 5.0
    optional_third_speaker_prob: float = 0.35
    gain_drift_db_range: tuple[float, float] = (-1.5, 1.5)
    scene_target_only_min_seconds: float = 0.8
    scene_nontarget_only_min_seconds: float = 0.8
    scene_overlap_min_seconds: float = 0.4
    scene_background_min_seconds: float = 0.3
    scene_absence_before_return_min_seconds: float = 1.0
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
    encoder_lr: float = 3e-5
    speaker_modulation_lr_scale: float = 2.0
    weight_decay: float = 0.0
    use_plateau_scheduler: bool = True
    plateau_patience: int = 5
    plateau_factor: float = 0.5
    early_stopping_patience: int = 20
    grad_clip: float = 5.0
    absent_warmup_epochs: int = 10

    # --- Separator architecture ---
    enc_channels: int = 2048
    bottleneck_channels: int = 256
    speaker_embed_dim: int = 192
    hidden_channels: int = 512
    r1_repeats: int = 3
    r2_repeats: int = 1
    conv_blocks_per_repeat: int = 2
    s4d_state_dim: int = 32
    s4d_ffn_multiplier: int = 4
    target_presence_head: bool = False

    # --- Loss weights (matches LossWeights defaults) ---
    loss_sdr: float = 1.0
    loss_mr_stft: float = 1.0
    loss_absent: float = 0.5
    loss_presence: float = 0.1
    loss_recall: float = 0.0
    loss_inactive: float = 0.25
    loss_route: float = 0.5
    loss_overlap_route: float = 0.25
    recall_floor: float = 0.3
    recall_frame_size: int = 320
    inactive_threshold: float = 0.05
    inactive_topk_fraction: float = 0.25
    route_frame_size: int = 160
    route_margin: float = 0.05
    overlap_margin: float = 0.02
    overlap_dominance_margin: float = 0.02
    checkpoint_other_only_alpha: float = 4.0
    checkpoint_wrong_enrollment_beta: float = 2.0
    checkpoint_overlap_wrong_gamma: float = 1.5

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
        if self.composition_mode not in ("clip_composer", "legacy_branch"):
            raise ValueError(
                "composition_mode must be 'clip_composer' or 'legacy_branch'; got "
                f"{self.composition_mode!r}"
            )
        if self.min_events <= 0 or self.max_events < self.min_events:
            raise ValueError(
                f"event bounds must satisfy 0 < min <= max; got "
                f"{self.min_events}, {self.max_events}"
            )
        if self.min_event_seconds <= 0.0 or self.max_event_seconds < self.min_event_seconds:
            raise ValueError(
                "event duration bounds must satisfy 0 < min <= max; got "
                f"{self.min_event_seconds}, {self.max_event_seconds}"
            )
        for name, value in (
            ("scene_target_only_min_seconds", self.scene_target_only_min_seconds),
            (
                "scene_nontarget_only_min_seconds",
                self.scene_nontarget_only_min_seconds,
            ),
            ("scene_overlap_min_seconds", self.scene_overlap_min_seconds),
            ("scene_background_min_seconds", self.scene_background_min_seconds),
            (
                "scene_absence_before_return_min_seconds",
                self.scene_absence_before_return_min_seconds,
            ),
        ):
            if value <= 0.0:
                raise ValueError(f"{name} must be positive; got {value}")
        if self.crossfade_ms < 0.0:
            raise ValueError(
                f"crossfade_ms must be non-negative; got {self.crossfade_ms}"
            )
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
        if not 0.0 < self.inactive_topk_fraction <= 1.0:
            raise ValueError(
                "inactive_topk_fraction must be in (0, 1]; got "
                f"{self.inactive_topk_fraction}"
            )
