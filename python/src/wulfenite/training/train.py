"""Wulfenite training loop."""

from __future__ import annotations

import argparse
import random
import time
from dataclasses import replace
from pathlib import Path
from typing import Callable

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from ..data import (
    ComposerConfig,
    MixerConfig,
    ReverbConfig,
    WulfeniteMixer,
    collate_mixer_batch,
    merge_speaker_dicts,
    scan_aishell1,
    scan_aishell3,
    scan_cnceleb,
    scan_magicdata,
    scan_noise_dir,
)
from ..losses import (
    LossWeights,
    WulfeniteLoss,
    compute_scene_routing_stats,
    compute_sdr_db,
    compute_sdri_db,
)
from ..models import SpeakerBeamSSConfig, WulfeniteTSE
from .checkpoint import save_checkpoint
from .config import TrainingConfig

SchedulerT = torch.optim.lr_scheduler.ReduceLROnPlateau | None


def _split_speakers_for_val(
    speakers: dict[str, list],
    val_ratio: float = 0.2,
    seed: int = 1234,
    *,
    min_speakers_per_split: int = 2,
) -> tuple[dict[str, list], dict[str, list]]:
    """Split speakers into reproducible train/val subsets."""
    speaker_ids = sorted(speakers.keys())
    min_total_speakers = 2 * min_speakers_per_split
    if len(speaker_ids) < min_total_speakers:
        raise RuntimeError(
            "Need at least "
            f"{min_total_speakers} speakers to make speaker-disjoint train/val "
            f"splits with at least {min_speakers_per_split} speakers per split."
        )

    rng = random.Random(seed)
    rng.shuffle(speaker_ids)
    n_val = max(min_speakers_per_split, int(len(speaker_ids) * val_ratio))
    n_val = min(n_val, len(speaker_ids) - min_speakers_per_split)
    if (
        n_val < min_speakers_per_split
        or len(speaker_ids) - n_val < min_speakers_per_split
    ):
        raise RuntimeError(
            "Speaker-disjoint split must leave at least "
            f"{min_speakers_per_split} speakers in each split."
        )

    val_ids = set(speaker_ids[:n_val])
    train_speakers = {k: v for k, v in speakers.items() if k not in val_ids}
    val_speakers = {k: v for k, v in speakers.items() if k in val_ids}
    return train_speakers, val_speakers


def _warmup_weight(base: float, epoch: int, warmup_epochs: int) -> float:
    """Linearly ramp a loss weight from 0 to *base* over *warmup_epochs*."""
    if base <= 0.0 or warmup_epochs <= 0:
        return base
    return base * min(1.0, epoch / warmup_epochs)


def effective_transition_prob(epoch_idx: int, cfg: TrainingConfig) -> float:
    """Curriculum-adjusted transition probability for a zero-based epoch."""
    if cfg.transition_prob <= 0.0:
        return 0.0
    warmup_end = int(cfg.epochs * cfg.transition_warmup_ratio)
    ramp_end = warmup_end + int(cfg.epochs * cfg.transition_ramp_ratio)
    if epoch_idx < warmup_end:
        return 0.0
    if epoch_idx >= ramp_end:
        return cfg.transition_prob
    ramp_len = ramp_end - warmup_end
    if ramp_len <= 0:
        return cfg.transition_prob
    progress = (epoch_idx - warmup_end) / ramp_len
    return progress * cfg.transition_prob


def build_dataset(cfg: TrainingConfig) -> tuple[WulfeniteMixer, WulfeniteMixer]:
    """Scan datasets, split speakers, and assemble train + val mixers."""
    speakers: dict = {}
    if cfg.aishell1_root is not None:
        speakers = scan_aishell1(cfg.aishell1_root, splits=("train", "dev", "test"))
    if cfg.aishell3_root is not None:
        speakers = merge_speaker_dicts(
            speakers, scan_aishell3(cfg.aishell3_root, splits=("train", "test")),
        )
    if cfg.magicdata_root is not None:
        speakers = merge_speaker_dicts(
            speakers,
            scan_magicdata(cfg.magicdata_root, splits=("train", "dev", "test")),
        )
    if cfg.cnceleb_root is not None:
        speakers = merge_speaker_dicts(speakers, scan_cnceleb(cfg.cnceleb_root))
    if not speakers:
        raise RuntimeError(
            "Training requires at least one of --aishell1-root / "
            "--aishell3-root / --magicdata-root / --cnceleb-root to produce "
            "a non-empty speaker pool."
        )

    train_speakers, val_speakers = _split_speakers_for_val(
        speakers,
        val_ratio=cfg.val_speaker_ratio,
        seed=cfg.seed,
        min_speakers_per_split=3 if cfg.composition_mode == "clip_composer" else 2,
    )

    noise_pool = None
    if cfg.noise_root is not None:
        noise_pool = scan_noise_dir(cfg.noise_root)

    if cfg.composition_mode == "clip_composer":
        composer_cfg = ComposerConfig(
            sample_rate=16000,
            segment_seconds=cfg.segment_seconds,
            target_only_min_frames=max(
                1, int(round(cfg.scene_target_only_min_seconds * 100)),
            ),
            nontarget_only_min_frames=max(
                1, int(round(cfg.scene_nontarget_only_min_seconds * 100)),
            ),
            overlap_min_frames=max(1, int(round(cfg.scene_overlap_min_seconds * 100))),
            background_min_frames=max(
                1, int(round(cfg.scene_background_min_seconds * 100)),
            ),
            absence_before_return_min_frames=max(
                1, int(round(cfg.scene_absence_before_return_min_seconds * 100)),
            ),
            crossfade_samples=max(0, int(round(cfg.crossfade_ms * 16))),
            optional_third_speaker_prob=cfg.optional_third_speaker_prob,
            gain_drift_db_range=tuple(cfg.gain_drift_db_range),
            global_gain_range_db=tuple(cfg.global_gain_range_db),
            snr_range_db=tuple(cfg.snr_range_db),
            noise_snr_range_db=tuple(cfg.noise_snr_range_db),
            overlap_density_weights=dict(cfg.overlap_density_weights),
            overlap_ratio_ranges={
                density: tuple(bounds)
                for density, bounds in cfg.overlap_ratio_ranges.items()
            },
            overlap_snr_center_range_db=tuple(cfg.overlap_snr_center_range_db),
            overlap_snr_tail_range_db=tuple(cfg.overlap_snr_tail_range_db),
            overlap_snr_center_prob=cfg.overlap_snr_center_prob,
        )
    else:
        composer_cfg = ComposerConfig()

    train_mixer_cfg = MixerConfig(
        segment_seconds=cfg.segment_seconds,
        enrollment_seconds=cfg.enrollment_seconds,
        snr_range_db=tuple(cfg.snr_range_db),
        composition_mode=cfg.composition_mode,
        composer=composer_cfg,
        target_present_prob=cfg.target_present_prob,
        outsider_view_prob=cfg.outsider_view_prob,
        transition_prob=cfg.transition_prob,
        transition_min_fraction=cfg.transition_min_fraction,
        transition_min_target_rms=cfg.transition_min_target_rms,
        noise_snr_range_db=tuple(cfg.noise_snr_range_db),
        noise_prob=cfg.noise_prob,
        reverb_prob=cfg.reverb_prob,
        rir_pool_size=cfg.rir_pool_size,
        reverb=ReverbConfig(),
    )
    val_mixer_cfg = replace(train_mixer_cfg, transition_prob=0.0)
    train_ds = WulfeniteMixer(
        speakers=train_speakers,
        noise_pool=noise_pool,
        config=train_mixer_cfg,
        samples_per_epoch=cfg.samples_per_epoch,
        seed=None,
    )
    val_ds = WulfeniteMixer(
        speakers=val_speakers,
        noise_pool=noise_pool,
        config=val_mixer_cfg,
        samples_per_epoch=cfg.val_samples,
        seed=cfg.seed,
    )
    return train_ds, val_ds


def build_model(cfg: TrainingConfig) -> WulfeniteTSE:
    """Build the Phase 3 fine-tuned CAM++ TSE model."""
    if cfg.campplus_checkpoint is None:
        raise RuntimeError("Phase 3 training requires --campplus-checkpoint.")

    separator_config = SpeakerBeamSSConfig(
        enc_channels=cfg.enc_channels,
        bottleneck_channels=cfg.bottleneck_channels,
        speaker_embed_dim=cfg.speaker_embed_dim,
        hidden_channels=cfg.hidden_channels,
        r1_repeats=cfg.r1_repeats,
        r2_repeats=cfg.r2_repeats,
        conv_blocks_per_repeat=cfg.conv_blocks_per_repeat,
        s4d_state_dim=cfg.s4d_state_dim,
        s4d_ffn_multiplier=cfg.s4d_ffn_multiplier,
        separator_lookahead_frames=cfg.separator_lookahead_frames,
        lookahead_policy=cfg.lookahead_policy,
        target_presence_head=cfg.target_presence_head,
        mask_activation=cfg.mask_activation,
    )
    return WulfeniteTSE.from_campplus(
        cfg.campplus_checkpoint,
        separator_config=separator_config,
    )


def build_loss(cfg: TrainingConfig) -> WulfeniteLoss:
    return WulfeniteLoss(
        weights=LossWeights(
            sdr=cfg.loss_sdr,
            mr_stft=cfg.loss_mr_stft,
            absent=cfg.loss_absent,
            presence=cfg.loss_presence,
            recall=cfg.loss_recall,
            inactive=cfg.loss_inactive,
            route=cfg.loss_route,
            overlap_route=cfg.loss_overlap_route,
            ae=cfg.loss_ae,
        ),
        recall_frame_size=cfg.recall_frame_size,
        recall_floor=cfg.recall_floor,
        inactive_threshold=cfg.inactive_threshold,
        inactive_topk_fraction=cfg.inactive_topk_fraction,
        route_frame_size=cfg.route_frame_size,
        route_margin=cfg.route_margin,
        overlap_margin=cfg.overlap_margin,
        overlap_dominance_margin=cfg.overlap_dominance_margin,
    )


def build_optimizer(
    model: torch.nn.Module,
    cfg: TrainingConfig,
    total_steps: int | None = None,
) -> tuple[torch.optim.Optimizer, SchedulerT]:
    """Build the optimizer and optional LR scheduler."""
    del total_steps

    encoder_groups = model.speaker_encoder.optimizer_groups(cfg)

    # Separator frontend (encoder + decoder) — lower LR for basis stability.
    frontend_params = [
        p for p in (
            list(model.separator.encoder.parameters())
            + list(model.separator.decoder.parameters())
        ) if p.requires_grad
    ]
    frontend_ids = {id(p) for p in frontend_params}

    speaker_modulation_params = [
        p
        for p in model.separator.speaker_projection.parameters()
        if p.requires_grad
    ]
    modulation_param_ids = {id(p) for p in speaker_modulation_params}

    excluded_ids = frontend_ids | modulation_param_ids
    separator_rest = [
        p for p in model.separator.parameters()
        if p.requires_grad and id(p) not in excluded_ids
    ]

    param_groups = [*encoder_groups]
    param_groups.append(
        {
            "name": "separator_frontend",
            "params": frontend_params,
            "lr": cfg.learning_rate * cfg.separator_frontend_lr_scale,
        }
    )
    param_groups.append(
        {
            "name": "separator_speaker_modulation",
            "params": speaker_modulation_params,
            "lr": cfg.learning_rate * cfg.speaker_modulation_lr_scale,
        }
    )
    param_groups.append(
        {
            "name": "separator_rest",
            "params": separator_rest,
            "lr": cfg.learning_rate,
        }
    )

    optimizer = torch.optim.Adam(
        param_groups,
        weight_decay=cfg.weight_decay,
    )
    scheduler: SchedulerT = None
    if cfg.use_plateau_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            patience=cfg.plateau_patience,
            factor=cfg.plateau_factor,
        )
    return optimizer, scheduler


def _move_batch(batch: dict, device: torch.device, non_blocking: bool) -> dict:
    return {
        k: (v.to(device, non_blocking=non_blocking) if torch.is_tensor(v) else v)
        for k, v in batch.items()
    }


def _encode_speaker_embeddings(
    model: WulfeniteTSE,
    batch: dict,
) -> tuple[torch.Tensor, torch.Tensor]:
    fbank = batch.get("enrollment_fbank")
    return model.speaker_encoder(
        batch["enrollment"],
        fbank=fbank,
    )


def _format_lr(optimizer: torch.optim.Optimizer) -> str:
    parts = []
    for i, group in enumerate(optimizer.param_groups):
        name = group.get("name", f"g{i}")
        parts.append(f"lr_{name}={group['lr']:.2e}")
    return " ".join(parts)


def compute_checkpoint_score(
    metrics: dict[str, float],
    cfg: TrainingConfig,
) -> float:
    return (
        metrics["sdri_db"]
        - cfg.checkpoint_other_only_alpha * metrics.get("other_only_energy_true", 0.0)
        - cfg.checkpoint_overlap_wrong_gamma
        * metrics.get("overlap_energy_wrong", 0.0)
    )


def _should_update_best_checkpoint(
    sdri: float,
    inactive: float,
    *,
    best_sdri: float,
    best_inactive: float,
    score: float | None = None,
    best_score: float | None = None,
    score_tolerance: float = 1e-4,
    sdri_tolerance: float = 0.05,
) -> bool:
    """Return whether the candidate beats the current best checkpoint."""
    if score is not None and best_score is not None:
        if score > best_score + score_tolerance:
            return True
        if abs(score - best_score) <= score_tolerance:
            sdri_improved = sdri > best_sdri + sdri_tolerance
            sdri_tied = abs(sdri - best_sdri) <= sdri_tolerance
            inactive_improved = inactive < best_inactive
            return sdri_improved or (sdri_tied and inactive_improved)
        return False
    sdri_improved = sdri > best_sdri + sdri_tolerance
    sdri_tied = abs(sdri - best_sdri) <= sdri_tolerance
    inactive_improved = inactive < best_inactive
    return sdri_improved or (sdri_tied and inactive_improved)


@torch.no_grad()
def compute_val_sdri_present(
    estimate: torch.Tensor,
    target: torch.Tensor,
    mixture: torch.Tensor,
    target_present: torch.Tensor,
) -> tuple[float, float, int]:
    """Compute present-only SDR / SDRi sums for validation aggregation."""
    present_mask = target_present.bool()
    n_present = int(present_mask.sum().item())
    if n_present == 0:
        return 0.0, 0.0, 0
    sdr_sum = float(
        compute_sdr_db(
            estimate[present_mask],
            target[present_mask],
            reduction="sum",
        ).item()
    )
    sdri_sum = float(
        compute_sdri_db(
            estimate[present_mask],
            target[present_mask],
            mixture[present_mask],
            reduction="sum",
        ).item()
    )
    return sdr_sum, sdri_sum, n_present


@torch.no_grad()
def compute_batch_routing_metrics(
    batch: dict,
    estimate: torch.Tensor,
    criterion: WulfeniteLoss,
) -> dict[str, float]:
    stats = compute_scene_routing_stats(
        estimate,
        batch["target"],
        batch["mixture"],
        target_active_frames=batch.get("target_active_frames"),
        nontarget_active_frames=batch.get("nontarget_active_frames"),
        overlap_frames=batch.get("overlap_frames"),
        background_frames=batch.get("background_frames"),
        scene_id=batch.get("scene_id"),
        view_role_id=batch.get("view_role_id"),
        frame_size=criterion.route_frame_size,
        route_margin=criterion.route_margin,
        overlap_margin=criterion.overlap_margin,
        overlap_dominance_margin=criterion.overlap_dominance_margin,
    )
    return {key: float(value.item()) for key, value in stats.items()}


@torch.no_grad()
def compute_same_diff_cosine_gap(
    model: WulfeniteTSE,
    loader: DataLoader,
    device: torch.device,
    *,
    max_batches: int | None = 8,
) -> tuple[float, float, float]:
    """Compute mean same-speaker / different-speaker cosine similarity."""
    model.eval()
    pin = device.type == "cuda"
    embeddings: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    for i, batch in enumerate(loader):
        batch = _move_batch(batch, device, non_blocking=pin)
        _, norm_emb = _encode_speaker_embeddings(model, batch)
        embeddings.append(norm_emb)
        labels.append(batch["target_speaker_idx"])
        if max_batches is not None and i + 1 >= max_batches:
            break

    if not embeddings:
        return 0.0, 0.0, 0.0

    emb = torch.cat(embeddings, dim=0)
    spk = torch.cat(labels, dim=0)
    if emb.size(0) < 2:
        return 0.0, 0.0, 0.0

    sim = emb @ emb.T
    same = spk.unsqueeze(0).eq(spk.unsqueeze(1))
    eye = torch.eye(sim.size(0), device=sim.device, dtype=torch.bool)
    same = same & ~eye
    diff = (~same) & ~eye

    same_vals = sim[same]
    diff_vals = sim[diff]
    same_mean = float(same_vals.mean().item()) if same_vals.numel() else 0.0
    diff_mean = float(diff_vals.mean().item()) if diff_vals.numel() else 0.0
    return same_mean, diff_mean, same_mean - diff_mean


@torch.no_grad()
def compute_enrollment_shuffle_sdr_drop(
    model: WulfeniteTSE,
    batch: dict,
    device: torch.device,
) -> float:
    """Measure how much SDR drops when enrollments are shuffled."""
    model.eval()
    pin = device.type == "cuda"
    batch = _move_batch(batch, device, non_blocking=pin)

    if batch["enrollment"].size(0) < 2:
        return 0.0

    present_mask = batch["target_present"].bool()
    if int(present_mask.sum().item()) == 0:
        return 0.0

    enrollment_fbank = batch.get("enrollment_fbank")
    out_true = model(
        batch["mixture"],
        batch["enrollment"],
        enrollment_fbank,
    )
    sdr_true = float(
        compute_sdr_db(
            out_true["clean"][present_mask],
            batch["target"][present_mask],
            reduction="mean",
        ).item()
    )

    perm = torch.randperm(batch["enrollment"].size(0), device=batch["enrollment"].device)
    if torch.equal(perm, torch.arange(perm.numel(), device=perm.device)):
        perm = perm.roll(1)

    out_shuf = model(
        batch["mixture"],
        batch["enrollment"][perm],
        enrollment_fbank[perm] if enrollment_fbank is not None else None,
    )
    sdr_shuf = float(
        compute_sdr_db(
            out_shuf["clean"][present_mask],
            batch["target"][present_mask],
            reduction="mean",
        ).item()
    )
    return sdr_true - sdr_shuf


def train_one_epoch(
    model: WulfeniteTSE,
    loader: DataLoader,
    criterion: WulfeniteLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    cfg: TrainingConfig,
    epoch: int,
    global_step: int,
    log_fn: Callable[[str], None],
    *,
    show_progress: bool = True,
) -> tuple[float, int]:
    """Run one epoch of training. Returns ``(mean_loss, new_global_step)``."""
    model.train()

    total_loss = 0.0
    n_batches = 0
    pin = device.type == "cuda"

    pbar = tqdm(
        loader,
        desc=f"epoch {epoch} train",
        total=len(loader),
        disable=not show_progress,
        leave=False,
        dynamic_ncols=True,
    )
    for step, batch in enumerate(pbar):
        batch = _move_batch(batch, device, non_blocking=pin)

        need_ae = criterion.weights.ae > 0.0
        outputs = model(
            batch["mixture"],
            batch["enrollment"],
            batch.get("enrollment_fbank"),
            return_training_aux=need_ae,
        )
        loss, parts = criterion(
            clean=outputs["clean"],
            target=batch["target"],
            mixture=batch["mixture"],
            target_present=batch["target_present"],
            presence_logit=outputs.get("presence_logit"),
            target_active_frames=batch.get("target_active_frames"),
            nontarget_active_frames=batch.get("nontarget_active_frames"),
            overlap_frames=batch.get("overlap_frames"),
            background_frames=batch.get("background_frames"),
            scene_id=batch.get("scene_id"),
            view_role_id=batch.get("view_role_id"),
            ae_reconstruction=outputs.get("ae_reconstruction"),
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                max_norm=cfg.grad_clip,
            )
        optimizer.step()

        total_loss += parts.total
        n_batches += 1
        global_step += 1

        postfix = {
            "loss": f"{parts.total:+.3f}",
            "sdr": f"{parts.sdr:+.2f}",
            "stft": f"{parts.mr_stft:.3f}",
            "recall": f"{parts.recall:.3f}",
            "inactive": f"{parts.inactive:.3f}",
            "route": f"{parts.route:.3f}",
            "ae": f"{parts.ae:.3f}",
            "lr_s": f"{optimizer.param_groups[-1]['lr']:.1e}",
        }
        pbar.set_postfix(refresh=False, **postfix)

        if global_step % cfg.log_interval == 0:
            log_fn(
                f"epoch {epoch} step {step + 1}/{len(loader)} "
                f"loss={parts.total:+.4f} sdr={parts.sdr:+.3f} "
                f"stft={parts.mr_stft:.4f} recall={parts.recall:.4f} "
                f"inactive={parts.inactive:.4f} route={parts.route:.4f} "
                f"overlap_route={parts.overlap_route:.4f} "
                f"absent={parts.absent:.4f} ae={parts.ae:.4f} "
                f"presence={parts.presence:.4f} "
                f"present={parts.n_present}/{parts.n_present + parts.n_absent} "
                f"route_pairs={parts.n_route_pairs} "
                f"{_format_lr(optimizer)}"
            )

    pbar.close()
    mean_loss = total_loss / max(1, n_batches)
    return mean_loss, global_step


@torch.no_grad()
def validate(
    model: WulfeniteTSE,
    loader: DataLoader,
    criterion: WulfeniteLoss,
    device: torch.device,
    *,
    show_progress: bool = True,
) -> tuple[float, dict[str, float]]:
    """Run one pass over the validation set. Returns ``(mean_loss, parts_mean)``."""
    model.eval()
    total = 0.0
    sdr_loss_sum = 0.0
    stft_sum = 0.0
    recall_sum = 0.0
    inactive_sum = 0.0
    absent_sum = 0.0
    presence_sum = 0.0
    route_sum = 0.0
    overlap_route_sum = 0.0
    ae_sum = 0.0
    sdr_db_sum = 0.0
    sdri_db_sum = 0.0
    routing_weight_sum = 0.0
    routing_metric_sums = {
        "target_only_energy_true": 0.0,
        "target_only_energy_wrong": 0.0,
        "other_only_energy_true": 0.0,
        "overlap_energy_true": 0.0,
        "overlap_energy_wrong": 0.0,
        "route_margin_target_only": 0.0,
        "route_margin_overlap": 0.0,
        "wrong_enrollment_leakage": 0.0,
    }
    n = 0
    n_present = 0
    pin = device.type == "cuda"

    iterator = tqdm(
        loader,
        desc="validate",
        total=len(loader),
        disable=not show_progress,
        leave=False,
        dynamic_ncols=True,
    )
    for batch in iterator:
        batch = _move_batch(batch, device, non_blocking=pin)
        need_ae = criterion.weights.ae > 0.0
        outputs = model(
            batch["mixture"],
            batch["enrollment"],
            batch.get("enrollment_fbank"),
            return_training_aux=need_ae,
        )
        _, parts = criterion(
            clean=outputs["clean"],
            target=batch["target"],
            mixture=batch["mixture"],
            target_present=batch["target_present"],
            presence_logit=outputs.get("presence_logit"),
            target_active_frames=batch.get("target_active_frames"),
            nontarget_active_frames=batch.get("nontarget_active_frames"),
            overlap_frames=batch.get("overlap_frames"),
            background_frames=batch.get("background_frames"),
            scene_id=batch.get("scene_id"),
            view_role_id=batch.get("view_role_id"),
            ae_reconstruction=outputs.get("ae_reconstruction"),
        )
        total += parts.total
        sdr_loss_sum += parts.sdr
        stft_sum += parts.mr_stft
        recall_sum += parts.recall
        inactive_sum += parts.inactive
        absent_sum += parts.absent
        presence_sum += parts.presence
        route_sum += parts.route
        overlap_route_sum += parts.overlap_route
        ae_sum += parts.ae
        batch_sdr_db, batch_sdri_db, batch_n_present = compute_val_sdri_present(
            outputs["clean"],
            batch["target"],
            batch["mixture"],
            batch["target_present"],
        )
        routing = compute_batch_routing_metrics(batch, outputs["clean"], criterion)
        pair_count = routing["n_pairs"]
        if pair_count > 0.0:
            routing_weight_sum += pair_count
            for key in routing_metric_sums:
                routing_metric_sums[key] += routing[key] * pair_count
        sdr_db_sum += batch_sdr_db
        sdri_db_sum += batch_sdri_db
        n_present += batch_n_present
        n += 1

    n_safe = max(1, n)
    n_present_safe = max(1, n_present)
    return total / n_safe, {
        "sdr_loss": sdr_loss_sum / n_safe,
        "mr_stft": stft_sum / n_safe,
        "recall": recall_sum / n_safe,
        "inactive": inactive_sum / n_safe,
        "absent": absent_sum / n_safe,
        "presence": presence_sum / n_safe,
        "route": route_sum / n_safe,
        "overlap_route": overlap_route_sum / n_safe,
        "ae": ae_sum / n_safe,
        "sdr_db": sdr_db_sum / n_present_safe,
        "sdri_db": sdri_db_sum / n_present_safe,
        "n_present": float(n_present),
        "routing_pairs": routing_weight_sum,
        **{
            key: (
                value / routing_weight_sum if routing_weight_sum > 0.0 else 0.0
            )
            for key, value in routing_metric_sums.items()
        },
    }


def run_training(
    cfg: TrainingConfig,
    *,
    model: WulfeniteTSE | None = None,
    show_progress: bool = True,
) -> None:
    """Run the full training loop described by ``cfg``."""
    torch.manual_seed(cfg.seed)

    device = torch.device(
        cfg.device if (cfg.device == "cpu" or torch.cuda.is_available()) else "cpu"
    )

    train_ds, val_ds = build_dataset(cfg)

    pin_memory = device.type == "cuda"
    curriculum_active = cfg.transition_prob > 0.0 and (
        cfg.transition_warmup_ratio > 0.0 or cfg.transition_ramp_ratio > 0.0
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=collate_mixer_batch,
        persistent_workers=cfg.num_workers > 0 and not curriculum_active,
        prefetch_factor=cfg.prefetch_factor if cfg.num_workers > 0 else None,
        pin_memory=pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=max(0, cfg.num_workers // 2),
        collate_fn=collate_mixer_batch,
        persistent_workers=cfg.num_workers > 1,
        prefetch_factor=cfg.prefetch_factor if cfg.num_workers > 1 else None,
        pin_memory=pin_memory,
        drop_last=False,
    )

    if model is None:
        model = build_model(cfg)
    model = model.to(device)

    criterion = build_loss(cfg).to(device)

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train.log"

    def log(msg: str) -> None:
        stamped = f"[{time.strftime('%H:%M:%S')}] {msg}"
        tqdm.write(stamped)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(stamped + "\n")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"device={device} trainable_params={trainable}")
    log(
        f"train_steps_per_epoch={len(train_loader)} "
        f"val_steps={len(val_loader)} total_steps={cfg.epochs * len(train_loader)} "
        f"train_speakers={len(train_ds.speaker_ids)} val_speakers={len(val_ds.speaker_ids)}"
    )

    probe_ds = WulfeniteMixer(
        speakers=train_ds.speakers,
        noise_pool=train_ds.noise_pool,
        config=replace(train_ds.cfg, transition_prob=0.0),
        samples_per_epoch=max(cfg.val_samples, cfg.batch_size * 4),
        seed=cfg.seed + 17,
    )
    probe_loader = DataLoader(
        probe_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=max(0, cfg.num_workers // 2),
        collate_fn=collate_mixer_batch,
        persistent_workers=cfg.num_workers > 1,
        prefetch_factor=cfg.prefetch_factor if cfg.num_workers > 1 else None,
        pin_memory=pin_memory,
        drop_last=False,
    )

    optimizer, scheduler = build_optimizer(model, cfg)

    best_sdri = float("-inf")
    best_inactive = float("inf")
    best_score = float("-inf")
    epochs_without_improvement = 0
    global_step = 0
    epoch_iter = tqdm(
        range(1, cfg.epochs + 1),
        desc="training",
        total=cfg.epochs,
        disable=not show_progress,
        dynamic_ncols=True,
    )
    for epoch in epoch_iter:
        epoch_idx = epoch - 1
        eff_prob = effective_transition_prob(epoch_idx, cfg)
        train_ds.cfg.transition_prob = eff_prob
        # Generalized warmup for suppression and regularization losses.
        criterion.weights.recall = cfg.loss_recall
        criterion.weights.absent = _warmup_weight(
            cfg.loss_absent, epoch, cfg.absent_warmup_epochs,
        )
        criterion.weights.inactive = _warmup_weight(
            cfg.loss_inactive, epoch, cfg.inactive_warmup_epochs,
        )
        criterion.weights.route = _warmup_weight(
            cfg.loss_route, epoch, cfg.route_warmup_epochs,
        )
        criterion.weights.overlap_route = _warmup_weight(
            cfg.loss_overlap_route, epoch, cfg.overlap_route_warmup_epochs,
        )
        criterion.weights.ae = _warmup_weight(
            cfg.loss_ae, epoch, cfg.ae_warmup_epochs,
        )
        epoch_t0 = time.time()
        train_loss, global_step = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            cfg,
            epoch,
            global_step,
            log,
            show_progress=show_progress,
        )
        train_time = time.time() - epoch_t0

        val_loss, val_parts = validate(
            model,
            val_loader,
            criterion,
            device,
            show_progress=show_progress,
        )
        if scheduler is not None:
            scheduler.step(val_parts["sdri_db"])
        checkpoint_score = compute_checkpoint_score(val_parts, cfg)

        metrics = {
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_num_speakers": len(train_ds.speaker_ids),
            "absent_weight": criterion.weights.absent,
            "checkpoint_score": checkpoint_score,
            **{f"val_{k}": v for k, v in val_parts.items()},
        }

        same, diff, gap = compute_same_diff_cosine_gap(
            model, probe_loader, device, max_batches=None
        )
        shuffle_batch = None
        for candidate in val_loader:
            if bool(candidate["target_present"].bool().any()):
                shuffle_batch = candidate
                break
        if shuffle_batch is None:
            shuffle_batch = next(iter(val_loader))
        shuffle_drop = compute_enrollment_shuffle_sdr_drop(
            model,
            shuffle_batch,
            device,
        )
        metrics.update(
            {
                "shuffle_sdr_drop": shuffle_drop,
                "same_cosine": same,
                "diff_cosine": diff,
                "cosine_gap": gap,
            }
        )

        log(
            f"epoch {epoch} transition_prob={eff_prob:.3f} "
            f"absent_weight={criterion.weights.absent:.4f} "
            f"train_loss={train_loss:+.4f} val_loss={val_loss:+.4f} "
            f"val_sdr_db={val_parts['sdr_db']:+.3f} "
            f"val_sdri_db={val_parts['sdri_db']:+.3f} "
            f"val_stft={val_parts['mr_stft']:.4f} "
            f"val_recall={val_parts['recall']:.4f} "
            f"val_inactive={val_parts['inactive']:.4f} "
            f"val_route={val_parts['route']:.4f} "
            f"val_overlap_route={val_parts['overlap_route']:.4f} "
            f"val_absent={val_parts['absent']:.4f} time={train_time:.1f}s "
            f"other_only={val_parts['other_only_energy_true']:.4f} "
            f"wrong_leak={val_parts['wrong_enrollment_leakage']:.4f} "
            f"overlap_wrong={val_parts['overlap_energy_wrong']:.4f} "
            f"score={checkpoint_score:+.4f} "
            f"shuffle_drop={shuffle_drop:.3f} "
            f"same_cos={same:.3f} diff_cos={diff:.3f} cos_gap={gap:.3f}"
        )

        if cfg.save_every_epoch:
            save_checkpoint(
                out_dir / f"epoch{epoch:03d}.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                step=global_step,
                config=cfg,
                metrics=metrics,
            )
        sdri = val_parts["sdri_db"]
        inactive_val = val_parts["inactive"]
        if _should_update_best_checkpoint(
            sdri,
            inactive_val,
            best_sdri=best_sdri,
            best_inactive=best_inactive,
            score=checkpoint_score,
            best_score=best_score,
        ):
            best_sdri = max(best_sdri, sdri)
            best_inactive = inactive_val
            best_score = max(best_score, checkpoint_score)
            epochs_without_improvement = 0
            save_checkpoint(
                out_dir / "best.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                step=global_step,
                config=cfg,
                metrics=metrics,
            )
            log(
                f"[best] val_sdri_db={sdri:+.4f} val_inactive={inactive_val:.4f} "
                f"checkpoint_score={checkpoint_score:+.4f} -> saved best.pt"
            )
        else:
            epochs_without_improvement += 1

        epoch_iter.set_postfix(
            train=f"{train_loss:+.3f}",
            sdri=f"{val_parts['sdri_db']:+.3f}",
            best=f"{best_score:+.3f}",
            refresh=False,
        )
        if epochs_without_improvement >= cfg.early_stopping_patience:
            log(
                "[early-stop] "
                f"no checkpoint_score improvement for {epochs_without_improvement} "
                f"epochs (best={best_score:+.4f}); stopping at epoch {epoch}"
            )
            break
    epoch_iter.close()


def _parse_args() -> TrainingConfig:
    parser = argparse.ArgumentParser(description="Train Wulfenite SpeakerBeam-SS")
    parser.add_argument("--aishell1-root", type=Path, default=None)
    parser.add_argument("--aishell3-root", type=Path, default=None)
    parser.add_argument("--magicdata-root", type=Path, default=None)
    parser.add_argument("--cnceleb-root", type=Path, default=None)
    parser.add_argument("--noise-root", type=Path, default=None)

    parser.add_argument("--segment-seconds", type=float, default=8.0)
    parser.add_argument("--enrollment-seconds", type=float, default=4.0)
    parser.add_argument("--snr-range-db", type=float, nargs=2, default=(-5.0, 5.0))
    parser.add_argument(
        "--composition-mode",
        choices=("clip_composer", "legacy_branch"),
        default="clip_composer",
    )
    parser.add_argument("--crossfade-ms", type=float, default=5.0)
    parser.add_argument("--optional-third-speaker-prob", type=float, default=0.35)
    parser.add_argument(
        "--gain-drift-db-range", type=float, nargs=2, default=(-1.5, 1.5),
    )
    parser.add_argument(
        "--global-gain-range-db", type=float, nargs=2, default=(-9.0, 9.0),
    )
    parser.add_argument("--scene-target-only-min-seconds", type=float, default=0.8)
    parser.add_argument("--scene-nontarget-only-min-seconds", type=float, default=0.8)
    parser.add_argument("--scene-overlap-min-seconds", type=float, default=0.4)
    parser.add_argument("--scene-background-min-seconds", type=float, default=0.3)
    parser.add_argument(
        "--scene-absence-before-return-min-seconds", type=float, default=1.0,
    )
    parser.add_argument(
        "--overlap-density-weights",
        type=float,
        nargs=3,
        metavar=("SPARSE", "MEDIUM", "DENSE"),
        default=(0.20, 0.55, 0.25),
    )
    parser.add_argument(
        "--overlap-ratio-sparse", type=float, nargs=2, default=(0.15, 0.25),
    )
    parser.add_argument(
        "--overlap-ratio-medium", type=float, nargs=2, default=(0.25, 0.40),
    )
    parser.add_argument(
        "--overlap-ratio-dense", type=float, nargs=2, default=(0.40, 0.55),
    )
    parser.add_argument(
        "--overlap-snr-center-range-db", type=float, nargs=2, default=(-2.0, 4.0),
    )
    parser.add_argument(
        "--overlap-snr-tail-range-db", type=float, nargs=2, default=(-6.0, 8.0),
    )
    parser.add_argument("--overlap-snr-center-prob", type=float, default=0.7)
    parser.add_argument("--target-present-prob", type=float, default=0.85)
    parser.add_argument("--outsider-view-prob", type=float, default=0.15)
    parser.add_argument("--transition-prob", type=float, default=0.0)
    parser.add_argument("--transition-warmup-ratio", type=float, default=0.0)
    parser.add_argument("--transition-ramp-ratio", type=float, default=0.0)
    parser.add_argument("--transition-min-fraction", type=float, default=0.25)
    parser.add_argument("--transition-min-target-rms", type=float, default=0.01)
    parser.add_argument("--noise-snr-range-db", type=float, nargs=2, default=(0.0, 25.0))
    parser.add_argument("--noise-prob", type=float, default=0.80)
    parser.add_argument("--reverb-prob", type=float, default=0.85)
    parser.add_argument("--rir-pool-size", type=int, default=1000)
    parser.add_argument("--val-speaker-ratio", type=float, default=0.2)

    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--samples-per-epoch", type=int, default=20000)
    parser.add_argument("--val-samples", type=int, default=500)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--encoder-lr", type=float, default=3e-5)
    parser.add_argument(
        "--speaker-modulation-lr-scale",
        "--film-lr-scale",
        dest="speaker_modulation_lr_scale",
        type=float,
        default=2.0,
    )
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--plateau-patience", type=int, default=5)
    parser.add_argument("--plateau-factor", type=float, default=0.5)
    parser.add_argument("--early-stopping-patience", type=int, default=20)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--absent-warmup-epochs", type=int, default=15)
    parser.add_argument("--inactive-warmup-epochs", type=int, default=15)
    parser.add_argument("--route-warmup-epochs", type=int, default=20)
    parser.add_argument("--overlap-route-warmup-epochs", type=int, default=20)
    parser.add_argument("--ae-warmup-epochs", type=int, default=2)
    parser.add_argument("--separator-frontend-lr-scale", type=float, default=0.5)

    parser.add_argument("--campplus-checkpoint", type=Path, default=None)
    parser.add_argument("--enc-channels", type=int, default=2048)
    parser.add_argument("--bottleneck-channels", type=int, default=256)
    parser.add_argument("--speaker-embed-dim", type=int, default=192)
    parser.add_argument("--hidden-channels", type=int, default=512)
    parser.add_argument("--r1-repeats", type=int, default=3)
    parser.add_argument("--r2-repeats", type=int, default=1)
    parser.add_argument("--conv-blocks-per-repeat", type=int, default=2)
    parser.add_argument("--s4d-state-dim", type=int, default=32)
    parser.add_argument("--s4d-ffn-multiplier", type=int, default=4)
    parser.add_argument("--separator-lookahead-frames", type=int, default=0)
    parser.add_argument(
        "--lookahead-policy",
        type=str,
        default="post_fusion_frontloaded",
        choices=["post_fusion_frontloaded"],
    )
    parser.add_argument("--target-presence-head", action="store_true")
    parser.add_argument("--mask-activation", type=str, default="scaled_sigmoid",
                        choices=["relu", "scaled_sigmoid"])

    parser.add_argument("--loss-sdr", type=float, default=1.0)
    parser.add_argument("--loss-mr-stft", type=float, default=1.0)
    parser.add_argument("--loss-absent", type=float, default=0.15)
    parser.add_argument("--loss-presence", type=float, default=0.1)
    parser.add_argument("--loss-recall", type=float, default=0.20)
    parser.add_argument("--loss-inactive", type=float, default=0.05)
    parser.add_argument("--loss-route", type=float, default=0.15)
    parser.add_argument("--loss-overlap-route", type=float, default=0.05)
    parser.add_argument("--loss-ae", type=float, default=0.10)
    parser.add_argument("--recall-floor", type=float, default=0.3)
    parser.add_argument("--recall-frame-size", type=int, default=320)
    parser.add_argument("--inactive-threshold", type=float, default=0.05)
    parser.add_argument("--inactive-topk-fraction", type=float, default=0.25)
    parser.add_argument("--route-frame-size", type=int, default=160)
    parser.add_argument("--route-margin", type=float, default=0.05)
    parser.add_argument("--overlap-margin", type=float, default=0.02)
    parser.add_argument("--overlap-dominance-margin", type=float, default=0.02)
    parser.add_argument("--checkpoint-other-only-alpha", type=float, default=4.0)
    parser.add_argument("--checkpoint-overlap-wrong-gamma", type=float, default=1.5)

    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--prefetch-factor", type=int, default=4)

    parser.add_argument("--out-dir", type=Path, default=Path("./checkpoints"))
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--no-save-every-epoch", action="store_true")

    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1234)

    args = parser.parse_args()

    return TrainingConfig(
        aishell1_root=args.aishell1_root,
        aishell3_root=args.aishell3_root,
        magicdata_root=args.magicdata_root,
        cnceleb_root=args.cnceleb_root,
        noise_root=args.noise_root,
        campplus_checkpoint=args.campplus_checkpoint,
        segment_seconds=args.segment_seconds,
        enrollment_seconds=args.enrollment_seconds,
        snr_range_db=tuple(args.snr_range_db),
        composition_mode=args.composition_mode,
        crossfade_ms=args.crossfade_ms,
        optional_third_speaker_prob=args.optional_third_speaker_prob,
        gain_drift_db_range=tuple(args.gain_drift_db_range),
        global_gain_range_db=tuple(args.global_gain_range_db),
        scene_target_only_min_seconds=args.scene_target_only_min_seconds,
        scene_nontarget_only_min_seconds=args.scene_nontarget_only_min_seconds,
        scene_overlap_min_seconds=args.scene_overlap_min_seconds,
        scene_background_min_seconds=args.scene_background_min_seconds,
        scene_absence_before_return_min_seconds=(
            args.scene_absence_before_return_min_seconds
        ),
        overlap_density_weights={
            "sparse": args.overlap_density_weights[0],
            "medium": args.overlap_density_weights[1],
            "dense": args.overlap_density_weights[2],
        },
        overlap_ratio_ranges={
            "sparse": tuple(args.overlap_ratio_sparse),
            "medium": tuple(args.overlap_ratio_medium),
            "dense": tuple(args.overlap_ratio_dense),
        },
        overlap_snr_center_range_db=tuple(args.overlap_snr_center_range_db),
        overlap_snr_tail_range_db=tuple(args.overlap_snr_tail_range_db),
        overlap_snr_center_prob=args.overlap_snr_center_prob,
        target_present_prob=args.target_present_prob,
        outsider_view_prob=args.outsider_view_prob,
        transition_prob=args.transition_prob,
        transition_warmup_ratio=args.transition_warmup_ratio,
        transition_ramp_ratio=args.transition_ramp_ratio,
        transition_min_fraction=args.transition_min_fraction,
        transition_min_target_rms=args.transition_min_target_rms,
        noise_snr_range_db=tuple(args.noise_snr_range_db),
        noise_prob=args.noise_prob,
        reverb_prob=args.reverb_prob,
        rir_pool_size=args.rir_pool_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        samples_per_epoch=args.samples_per_epoch,
        val_samples=args.val_samples,
        learning_rate=args.lr,
        encoder_lr=args.encoder_lr,
        speaker_modulation_lr_scale=args.speaker_modulation_lr_scale,
        weight_decay=args.weight_decay,
        use_plateau_scheduler=True,
        plateau_patience=args.plateau_patience,
        plateau_factor=args.plateau_factor,
        early_stopping_patience=args.early_stopping_patience,
        grad_clip=args.grad_clip,
        absent_warmup_epochs=args.absent_warmup_epochs,
        inactive_warmup_epochs=args.inactive_warmup_epochs,
        route_warmup_epochs=args.route_warmup_epochs,
        overlap_route_warmup_epochs=args.overlap_route_warmup_epochs,
        ae_warmup_epochs=args.ae_warmup_epochs,
        separator_frontend_lr_scale=args.separator_frontend_lr_scale,
        enc_channels=args.enc_channels,
        bottleneck_channels=args.bottleneck_channels,
        speaker_embed_dim=args.speaker_embed_dim,
        hidden_channels=args.hidden_channels,
        r1_repeats=args.r1_repeats,
        r2_repeats=args.r2_repeats,
        conv_blocks_per_repeat=args.conv_blocks_per_repeat,
        s4d_state_dim=args.s4d_state_dim,
        s4d_ffn_multiplier=args.s4d_ffn_multiplier,
        separator_lookahead_frames=args.separator_lookahead_frames,
        lookahead_policy=args.lookahead_policy,
        target_presence_head=args.target_presence_head,
        mask_activation=args.mask_activation,
        loss_sdr=args.loss_sdr,
        loss_mr_stft=args.loss_mr_stft,
        loss_absent=args.loss_absent,
        loss_presence=args.loss_presence,
        loss_recall=args.loss_recall,
        loss_inactive=args.loss_inactive,
        loss_route=args.loss_route,
        loss_overlap_route=args.loss_overlap_route,
        loss_ae=args.loss_ae,
        recall_floor=args.recall_floor,
        recall_frame_size=args.recall_frame_size,
        inactive_threshold=args.inactive_threshold,
        inactive_topk_fraction=args.inactive_topk_fraction,
        route_frame_size=args.route_frame_size,
        route_margin=args.route_margin,
        overlap_margin=args.overlap_margin,
        overlap_dominance_margin=args.overlap_dominance_margin,
        checkpoint_other_only_alpha=args.checkpoint_other_only_alpha,
        checkpoint_overlap_wrong_gamma=args.checkpoint_overlap_wrong_gamma,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        val_speaker_ratio=args.val_speaker_ratio,
        out_dir=args.out_dir,
        log_interval=args.log_interval,
        save_every_epoch=not args.no_save_every_epoch,
        device=args.device,
        seed=args.seed,
    )


def main() -> None:
    cfg = _parse_args()
    run_training(cfg)


if __name__ == "__main__":
    main()
