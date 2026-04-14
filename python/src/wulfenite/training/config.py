"""Training configuration for Wulfenite."""

from __future__ import annotations

import warnings
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
    crossfade_ms: float = 5.0
    optional_third_speaker_prob: float = 0.35
    gain_drift_db_range: tuple[float, float] = (-1.5, 1.5)
    global_gain_range_db: tuple[float, float] = (-9.0, 9.0)
    scene_target_only_min_seconds: float = 0.8
    scene_nontarget_only_min_seconds: float = 0.8
    scene_overlap_min_seconds: float = 0.4
    scene_background_min_seconds: float = 0.3
    scene_absence_before_return_min_seconds: float = 1.0
    overlap_density_weights: dict[str, float] = field(
        default_factory=lambda: {
            "sparse": 0.20,
            "medium": 0.55,
            "dense": 0.25,
        }
    )
    overlap_ratio_ranges: dict[str, tuple[float, float]] = field(
        default_factory=lambda: {
            "sparse": (0.15, 0.25),
            "medium": (0.25, 0.40),
            "dense": (0.40, 0.55),
        }
    )
    overlap_snr_center_range_db: tuple[float, float] = (-2.0, 4.0)
    overlap_snr_tail_range_db: tuple[float, float] = (-6.0, 8.0)
    overlap_snr_center_prob: float = 0.7
    target_present_prob: float = 0.85
    outsider_view_prob: float = 0.15
    transition_prob: float = 0.0
    transition_warmup_ratio: float = 0.0
    transition_ramp_ratio: float = 0.0
    transition_min_fraction: float = 0.25
    transition_min_target_rms: float = 0.01
    noise_snr_range_db: tuple[float, float] = (0.0, 25.0)
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
    absent_warmup_epochs: int = 15
    inactive_warmup_epochs: int = 15
    route_warmup_epochs: int = 20
    overlap_route_warmup_epochs: int = 20
    ae_warmup_epochs: int = 2
    separator_frontend_lr_scale: float = 0.5

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
    separator_lookahead_frames: int = 0
    lookahead_policy: str = "post_fusion_frontloaded"
    target_presence_head: bool = False
    mask_activation: str = "scaled_sigmoid"

    # --- Loss weights ---
    loss_sdr: float = 1.0
    loss_mr_stft: float = 1.0
    loss_absent: float = 0.15
    loss_presence: float = 0.1
    loss_recall: float = 0.20
    loss_inactive: float = 0.05
    loss_route: float = 0.15
    loss_overlap_route: float = 0.05
    loss_ae: float = 0.10
    recall_floor: float = 0.3
    recall_frame_size: int = 320
    inactive_threshold: float = 0.05
    inactive_topk_fraction: float = 0.25
    route_frame_size: int = 160
    route_margin: float = 0.05
    overlap_margin: float = 0.02
    overlap_dominance_margin: float = 0.02
    checkpoint_other_only_alpha: float = 4.0
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
        if (
            self.composition_mode == "clip_composer"
            and tuple(self.snr_range_db) != (-5.0, 5.0)
        ):
            warnings.warn(
                "snr_range_db is ignored in clip_composer mode; use "
                "overlap_snr_center_range_db / overlap_snr_tail_range_db instead",
                UserWarning,
                stacklevel=2,
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
        if not 0.0 <= self.outsider_view_prob <= 1.0:
            raise ValueError(
                "outsider_view_prob must be in [0, 1]; got "
                f"{self.outsider_view_prob}"
            )
        if not 0.0 <= self.overlap_snr_center_prob <= 1.0:
            raise ValueError(
                "overlap_snr_center_prob must be in [0, 1]; got "
                f"{self.overlap_snr_center_prob}"
            )
        if not 0.0 < self.inactive_topk_fraction <= 1.0:
            raise ValueError(
                "inactive_topk_fraction must be in (0, 1]; got "
                f"{self.inactive_topk_fraction}"
            )
        total_blocks = (self.r1_repeats + self.r2_repeats) * self.conv_blocks_per_repeat
        if not 0 <= self.separator_lookahead_frames <= total_blocks:
            raise ValueError(
                "separator_lookahead_frames must be between 0 and "
                f"{total_blocks}; got {self.separator_lookahead_frames}"
            )
        if self.lookahead_policy != "post_fusion_frontloaded":
            raise ValueError(
                "lookahead_policy must be 'post_fusion_frontloaded'; got "
                f"{self.lookahead_policy!r}"
            )
