"""Wulfenite training loop.

Ties together :mod:`wulfenite.data`, :mod:`wulfenite.models`, and
:mod:`wulfenite.losses` into a standard PyTorch training loop with
whole-utterance forward, Adam + ReduceLROnPlateau, gradient clipping,
and epoch-level checkpointing.

Call from the command line:

    uv run --directory python python -m wulfenite.training.train \\
        --aishell1-root ~/datasets/aishell1 \\
        --aishell3-root ~/datasets/aishell3 \\
        --noise-root ~/datasets/musan/noise \\
        --out-dir ./checkpoints/phase1

Or from Python (used by tests):

    from wulfenite.training.train import run_training
    from wulfenite.training.config import TrainingConfig

    run_training(TrainingConfig(...), model=my_model)  # model optional
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from ..data import (
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
    compute_sdr_db,
    compute_sdri_db,
)
from ..models import (
    LearnableDVector,
    SpeakerEncoderOutput,
    SpeakerBeamSSConfig,
    WulfeniteTSE,
    compute_fbank_batch,
)
from .checkpoint import save_checkpoint
from .config import TrainingConfig

SchedulerT = torch.optim.lr_scheduler.ReduceLROnPlateau | None


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _split_speakers_for_val(
    speakers: dict[str, list],
    val_ratio: float = 0.2,
    seed: int = 1234,
) -> tuple[dict[str, list], dict[str, list]]:
    """Split speakers into reproducible train/val subsets."""
    speaker_ids = sorted(speakers.keys())
    if len(speaker_ids) < 4:
        raise RuntimeError(
            "Need at least 4 speakers to make speaker-disjoint train/val splits."
        )

    rng = random.Random(seed)
    rng.shuffle(speaker_ids)
    n_val = max(2, int(len(speaker_ids) * val_ratio))
    n_val = min(n_val, len(speaker_ids) - 2)
    if n_val < 2 or len(speaker_ids) - n_val < 2:
        raise RuntimeError(
            "Speaker-disjoint split must leave at least 2 speakers in each split."
        )

    val_ids = set(speaker_ids[:n_val])
    train_speakers = {k: v for k, v in speakers.items() if k not in val_ids}
    val_speakers = {k: v for k, v in speakers.items() if k in val_ids}
    return train_speakers, val_speakers


def build_dataset(cfg: TrainingConfig) -> tuple[WulfeniteMixer, WulfeniteMixer]:
    """Scan datasets, split speakers, and assemble train + val mixers."""
    speakers: dict = {}
    if cfg.aishell1_root is not None:
        # Use all splits — our own speaker-disjoint split supersedes
        # the dataset's original train/dev/test partition.
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
    )

    noise_pool = None
    if cfg.noise_root is not None:
        noise_pool = scan_noise_dir(cfg.noise_root)

    mixer_cfg = MixerConfig(
        segment_seconds=cfg.segment_seconds,
        enrollment_seconds=cfg.enrollment_seconds,
        snr_range_db=tuple(cfg.snr_range_db),
        target_present_prob=cfg.target_present_prob,
        noise_snr_range_db=tuple(cfg.noise_snr_range_db),
        noise_prob=cfg.noise_prob,
        reverb_prob=cfg.reverb_prob,
        rir_pool_size=cfg.rir_pool_size,
        reverb=ReverbConfig(),
    )
    train_ds = WulfeniteMixer(
        speakers=train_speakers,
        noise_pool=noise_pool,
        config=mixer_cfg,
        samples_per_epoch=cfg.samples_per_epoch,
        seed=None,  # fresh mixtures per call
    )
    val_ds = WulfeniteMixer(
        speakers=val_speakers,
        noise_pool=noise_pool,
        config=mixer_cfg,
        samples_per_epoch=cfg.val_samples,
        seed=cfg.seed,  # reproducible validation
    )
    return train_ds, val_ds


def build_model(
    cfg: TrainingConfig,
    *,
    num_speakers: int | None = None,
) -> WulfeniteTSE:
    """Build the TSE model for the configured speaker encoder type."""
    separator_config = SpeakerBeamSSConfig(
        enc_channels=cfg.enc_channels,
        bottleneck_channels=cfg.bottleneck_channels,
        hidden_channels=cfg.hidden_channels,
        num_repeats=cfg.num_repeats,
        r1_blocks=cfg.r1_blocks,
        r2_blocks=cfg.r2_blocks,
        s4d_state_dim=cfg.s4d_state_dim,
    )

    if cfg.encoder_type == "learnable":
        if num_speakers is None:
            raise RuntimeError(
                "num_speakers must be provided to build the training model."
            )
        return WulfeniteTSE.from_learnable_dvector(
            num_speakers=num_speakers,
            separator_config=separator_config,
        )

    if cfg.encoder_type == "campplus-frozen":
        if cfg.campplus_checkpoint is None:
            raise RuntimeError(
                "campplus-frozen requires --campplus-checkpoint."
            )
        return WulfeniteTSE.from_campplus(
            cfg.campplus_checkpoint,
            separator_config=separator_config,
            freeze_backbone=True,
            num_speakers=num_speakers,
            projection_type=cfg.campplus_projection_type,
            projection_hidden_dim=cfg.campplus_projection_hidden_dim,
            arcface_scale=cfg.arcface_scale,
            arcface_margin=cfg.arcface_margin,
        )

    if cfg.encoder_type == "campplus-finetune":
        if cfg.campplus_checkpoint is None:
            raise RuntimeError(
                "campplus-finetune requires --campplus-checkpoint."
            )
        return WulfeniteTSE.from_campplus(
            cfg.campplus_checkpoint,
            separator_config=separator_config,
            freeze_backbone=False,
            num_speakers=num_speakers,
            projection_type=cfg.campplus_projection_type,
            projection_hidden_dim=cfg.campplus_projection_hidden_dim,
            arcface_scale=cfg.arcface_scale,
            arcface_margin=cfg.arcface_margin,
        )

    raise ValueError(f"Unsupported encoder_type: {cfg.encoder_type}")


def build_loss(cfg: TrainingConfig) -> WulfeniteLoss:
    return WulfeniteLoss(
        weights=LossWeights(
            sdr=cfg.loss_sdr,
            mr_stft=cfg.loss_mr_stft,
            absent=cfg.loss_absent,
            presence=cfg.loss_presence,
            speaker_cls=cfg.loss_speaker_cls,
        ),
    )


def build_optimizer(
    model: torch.nn.Module,
    cfg: TrainingConfig,
    total_steps: int | None = None,
) -> tuple[torch.optim.Optimizer, SchedulerT]:
    """Build the optimizer and optional LR scheduler."""
    encoder_groups = model.speaker_encoder.optimizer_groups(cfg)
    param_groups = [*encoder_groups]
    param_groups.append(
        {
            "name": "separator",
            "params": [p for p in model.separator.parameters() if p.requires_grad],
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


def lambda_cls_schedule(
    epoch: int,
    total_epochs: int,
    start: float = 0.2,
    end: float = 0.2,
    decay_epochs: int = 25,
) -> float:
    """Legacy helper for experiments with speaker-classification schedules."""
    if decay_epochs <= 0 or epoch >= decay_epochs:
        return end
    frac = epoch / decay_epochs
    return start + (end - start) * frac


# ---------------------------------------------------------------------------
# Train / validate epochs
# ---------------------------------------------------------------------------


def _move_batch(batch: dict, device: torch.device, non_blocking: bool) -> dict:
    return {
        k: (v.to(device, non_blocking=non_blocking) if torch.is_tensor(v) else v)
        for k, v in batch.items()
    }


def _speaker_encoder_outputs(
    model: WulfeniteTSE,
    batch: dict,
    speaker_labels: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Return raw embedding, separator embedding, and optional logits."""
    fbank = batch.get("enrollment_fbank")
    if isinstance(model.speaker_encoder, LearnableDVector):
        if fbank is None:
            fbank = compute_fbank_batch(batch["enrollment"])
        return model.speaker_encoder(fbank)

    encoder_out = model.speaker_encoder(
        batch["enrollment"],
        fbank=fbank,
        speaker_labels=speaker_labels,
    )
    if not isinstance(encoder_out, SpeakerEncoderOutput):
        raise TypeError(
            "speaker_encoder must return either the learnable "
            "d-vector tuple or SpeakerEncoderOutput."
        )
    return (
        encoder_out.native_embedding,
        encoder_out.separator_embedding,
        encoder_out.speaker_logits,
    )


def _format_lr(optimizer: torch.optim.Optimizer) -> str:
    parts = []
    for i, group in enumerate(optimizer.param_groups):
        name = group.get("name", f"g{i}")
        parts.append(f"lr_{name}={group['lr']:.2e}")
    return " ".join(parts)


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
def compute_speaker_top1_accuracy(
    model: WulfeniteTSE,
    loader: DataLoader,
    device: torch.device,
    *,
    max_batches: int | None = 8,
) -> float:
    """Compute top-1 speaker accuracy on a train-speaker probe set."""
    model.eval()
    pin = device.type == "cuda"
    total = 0
    correct = 0
    for i, batch in enumerate(loader):
        batch = _move_batch(batch, device, non_blocking=pin)
        _, _, logits = _speaker_encoder_outputs(
            model,
            batch,
            speaker_labels=None,
        )
        if logits is None:
            raise RuntimeError("Training diagnostics require a classifier head.")
        pred = logits.argmax(dim=-1)
        target = batch["target_speaker_idx"]
        correct += int((pred == target).sum().item())
        total += int(target.numel())
        if max_batches is not None and i + 1 >= max_batches:
            break
    return correct / max(1, total)


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
        _, norm_emb, _ = _speaker_encoder_outputs(model, batch)
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

        outputs = model(
            batch["mixture"],
            batch["enrollment"],
            batch.get("enrollment_fbank"),
            target_speaker_idx=batch.get("target_speaker_idx"),
        )
        loss, parts = criterion(
            clean=outputs["clean"],
            target=batch["target"],
            mixture=batch["mixture"],
            target_present=batch["target_present"],
            presence_logit=outputs.get("presence_logit"),
            speaker_logits=outputs.get("speaker_logits"),
            target_speaker_idx=batch.get("target_speaker_idx"),
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
            "cls": f"{parts.speaker_cls:.3f}",
            "lr_s": f"{optimizer.param_groups[-1]['lr']:.1e}",
        }
        pbar.set_postfix(refresh=False, **postfix)

        if global_step % cfg.log_interval == 0:
            log_fn(
                f"epoch {epoch} step {step + 1}/{len(loader)} "
                f"loss={parts.total:+.4f} sdr={parts.sdr:+.3f} "
                f"stft={parts.mr_stft:.4f} absent={parts.absent:.4f} "
                f"presence={parts.presence:.4f} speaker_cls={parts.speaker_cls:.4f} "
                f"present={parts.n_present}/{parts.n_present + parts.n_absent} "
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
    absent_sum = 0.0
    presence_sum = 0.0
    sdr_db_sum = 0.0
    sdri_db_sum = 0.0
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
        outputs = model(
            batch["mixture"],
            batch["enrollment"],
            batch.get("enrollment_fbank"),
            target_speaker_idx=batch.get("target_speaker_idx"),
        )
        _, parts = criterion(
            clean=outputs["clean"],
            target=batch["target"],
            mixture=batch["mixture"],
            target_present=batch["target_present"],
            presence_logit=outputs.get("presence_logit"),
            # Validation speakers are disjoint from train speakers, so
            # the auxiliary classifier is intentionally excluded here.
            # Speaker discrimination is tracked via the separate probe
            # loader top-1 / cosine-gap diagnostics instead, and
            # ``speaker_cls`` is omitted from the returned metrics dict
            # so it does not appear as a misleading "val_speaker_cls=0"
            # in logs or checkpoint metadata.
            speaker_logits=None,
            target_speaker_idx=None,
        )
        total += parts.total
        sdr_loss_sum += parts.sdr
        stft_sum += parts.mr_stft
        absent_sum += parts.absent
        presence_sum += parts.presence
        batch_sdr_db, batch_sdri_db, batch_n_present = compute_val_sdri_present(
            outputs["clean"],
            batch["target"],
            batch["mixture"],
            batch["target_present"],
        )
        sdr_db_sum += batch_sdr_db
        sdri_db_sum += batch_sdri_db
        n_present += batch_n_present
        n += 1

    n_safe = max(1, n)
    n_present_safe = max(1, n_present)
    return total / n_safe, {
        "sdr_loss": sdr_loss_sum / n_safe,
        "mr_stft": stft_sum / n_safe,
        "absent": absent_sum / n_safe,
        "presence": presence_sum / n_safe,
        "sdr_db": sdr_db_sum / n_present_safe,
        "sdri_db": sdri_db_sum / n_present_safe,
        "n_present": float(n_present),
    }


def run_encoder_pretrain(
    cfg: TrainingConfig,
    model: WulfeniteTSE,
    train_loader: DataLoader,
    probe_loader: DataLoader,
    device: torch.device,
    epochs: int,
    log_fn: Callable[[str], None],
    *,
    show_progress: bool = True,
) -> None:
    """Phase A: pretrain only the speaker encoder + classifier."""
    if epochs <= 0:
        return

    for p in model.separator.parameters():
        p.requires_grad_(False)

    optimizer = torch.optim.Adam(
        model.speaker_encoder.optimizer_groups(cfg, base_lr=cfg.encoder_pretrain_lr),
        weight_decay=cfg.weight_decay,
    )
    pin = device.type == "cuda"

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0
        pbar = tqdm(
            train_loader,
            desc=f"pretrain {epoch}",
            total=len(train_loader),
            disable=not show_progress,
            leave=False,
            dynamic_ncols=True,
        )
        for batch in pbar:
            batch = _move_batch(batch, device, non_blocking=pin)
            _, _, logits = _speaker_encoder_outputs(
                model,
                batch,
                speaker_labels=batch["target_speaker_idx"],
            )
            if logits is None:
                raise RuntimeError(
                    "Speaker-encoder pretrain requires a classifier head."
                )
            loss = F.cross_entropy(logits, batch["target_speaker_idx"])

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.speaker_encoder.parameters() if p.requires_grad],
                    max_norm=cfg.grad_clip,
                )
            optimizer.step()

            total_loss += float(loss.detach())
            n_batches += 1
            pbar.set_postfix(
                loss=f"{float(loss.detach()):.3f}",
                lr=f"{optimizer.param_groups[0]['lr']:.1e}",
                refresh=False,
            )
        pbar.close()

        top1 = compute_speaker_top1_accuracy(
            model, probe_loader, device, max_batches=None
        )
        same, diff, gap = compute_same_diff_cosine_gap(
            model, probe_loader, device, max_batches=None
        )
        log_fn(
            f"[pretrain] epoch {epoch}/{epochs} "
            f"loss={total_loss / max(1, n_batches):.4f} "
            f"top1={top1:.3f} same_cos={same:.3f} diff_cos={diff:.3f} gap={gap:.3f}"
        )

    for p in model.separator.parameters():
        p.requires_grad_(True)


# ---------------------------------------------------------------------------
# Top-level entrypoint
# ---------------------------------------------------------------------------


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
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=False,  # virtual epoch
        num_workers=cfg.num_workers,
        collate_fn=collate_mixer_batch,
        persistent_workers=cfg.num_workers > 0,
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
        model = build_model(
            cfg,
            num_speakers=len(train_ds.speaker_ids),
        )
    elif getattr(model.speaker_encoder, "supports_classifier", False):
        classifier = getattr(model.speaker_encoder, "classifier", None)
        if classifier is None:
            raise ValueError(
                "Training requires a speaker encoder with a classifier head."
            )
        if classifier.out_features != len(train_ds.speaker_ids):
            raise ValueError(
                "Provided model classifier size must match the training speaker set."
            )
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
        config=train_ds.cfg,
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
    if getattr(model.speaker_encoder, "supports_pretrain", False):
        run_encoder_pretrain(
            cfg,
            model,
            train_loader,
            probe_loader,
            device,
            cfg.encoder_pretrain_epochs,
            log,
            show_progress=show_progress,
        )
    elif cfg.encoder_pretrain_epochs > 0:
        log("[pretrain] skipped: selected speaker encoder does not support it")

    optimizer, scheduler = build_optimizer(model, cfg)

    best_sdri = float("-inf")
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

        metrics = {
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_num_speakers": len(train_ds.speaker_ids),
            **{f"val_{k}": v for k, v in val_parts.items()},
        }

        diag_parts = []
        if getattr(model.speaker_encoder, "supports_classifier", False):
            top1 = compute_speaker_top1_accuracy(
                model, probe_loader, device, max_batches=None
            )
            metrics["speaker_top1"] = top1
            diag_parts.append(f"top1={top1:.3f}")
        same, diff, gap = compute_same_diff_cosine_gap(
            model, val_loader, device, max_batches=None
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
        diag_parts.extend(
            [
                f"shuffle_drop={shuffle_drop:.3f}",
                f"same_cos={same:.3f}",
                f"diff_cos={diff:.3f}",
                f"cos_gap={gap:.3f}",
            ]
        )

        log(
            f"epoch {epoch} train_loss={train_loss:+.4f} val_loss={val_loss:+.4f} "
            f"val_sdr_db={val_parts['sdr_db']:+.3f} "
            f"val_sdri_db={val_parts['sdri_db']:+.3f} "
            f"val_stft={val_parts['mr_stft']:.4f} "
            f"val_absent={val_parts['absent']:.4f} time={train_time:.1f}s "
            + (" ".join(diag_parts) if diag_parts else "")
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
        if val_parts["sdri_db"] > best_sdri:
            best_sdri = val_parts["sdri_db"]
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
            log(f"[best] val_sdri_db improved to {best_sdri:+.4f} -> saved best.pt")
        else:
            epochs_without_improvement += 1

        epoch_iter.set_postfix(
            train=f"{train_loss:+.3f}",
            sdri=f"{val_parts['sdri_db']:+.3f}",
            best=f"{best_sdri:+.3f}",
            refresh=False,
        )
        if epochs_without_improvement >= cfg.early_stopping_patience:
            log(
                "[early-stop] "
                f"no val_sdri_db improvement for {epochs_without_improvement} epochs "
                f"(best={best_sdri:+.4f}); stopping at epoch {epoch}"
            )
            break
    epoch_iter.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> TrainingConfig:
    parser = argparse.ArgumentParser(description="Train Wulfenite SpeakerBeam-SS")
    # Data
    parser.add_argument("--aishell1-root", type=Path, default=None)
    parser.add_argument("--aishell3-root", type=Path, default=None)
    parser.add_argument("--magicdata-root", type=Path, default=None)
    parser.add_argument("--cnceleb-root", type=Path, default=None)
    parser.add_argument("--noise-root", type=Path, default=None)
    # Mixer
    parser.add_argument("--segment-seconds", type=float, default=4.0)
    parser.add_argument("--enrollment-seconds", type=float, default=4.0)
    parser.add_argument("--snr-range-db", type=float, nargs=2, default=(-5.0, 5.0))
    parser.add_argument("--target-present-prob", type=float, default=0.85)
    parser.add_argument("--noise-snr-range-db", type=float, nargs=2, default=(10.0, 25.0))
    parser.add_argument("--noise-prob", type=float, default=0.80)
    parser.add_argument("--reverb-prob", type=float, default=0.85)
    parser.add_argument("--rir-pool-size", type=int, default=1000)
    parser.add_argument("--val-speaker-ratio", type=float, default=0.2)
    # Optimizer
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--samples-per-epoch", type=int, default=20000)
    parser.add_argument("--val-samples", type=int, default=500)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--plateau-patience", type=int, default=5)
    parser.add_argument("--plateau-factor", type=float, default=0.5)
    parser.add_argument("--early-stopping-patience", type=int, default=20)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--encoder-pretrain-epochs", type=int, default=5)
    parser.add_argument("--encoder-pretrain-lr", type=float, default=3e-4)
    parser.add_argument("--encoder-lr-scale", type=float, default=0.25)
    parser.add_argument(
        "--encoder-type",
        choices=("learnable", "campplus-frozen", "campplus-finetune"),
        default="learnable",
    )
    parser.add_argument("--campplus-checkpoint", type=Path, default=None)
    parser.add_argument(
        "--campplus-projection-type",
        choices=("mlp", "linear"),
        default="mlp",
    )
    parser.add_argument(
        "--campplus-projection-hidden-dim",
        type=int,
        default=384,
    )
    parser.add_argument("--arcface-scale", type=float, default=30.0)
    parser.add_argument("--arcface-margin", type=float, default=0.2)
    parser.add_argument("--enc-channels", type=int, default=4096)
    parser.add_argument("--bottleneck-channels", type=int, default=256)
    parser.add_argument("--hidden-channels", type=int, default=512)
    parser.add_argument("--num-repeats", type=int, default=2)
    parser.add_argument("--r1-blocks", type=int, default=3)
    parser.add_argument("--r2-blocks", type=int, default=1)
    parser.add_argument("--s4d-state-dim", type=int, default=32)
    # Loss
    parser.add_argument("--loss-sdr", type=float, default=1.0)
    parser.add_argument("--loss-mr-stft", type=float, default=1.0)
    parser.add_argument("--loss-absent", type=float, default=0.5)
    parser.add_argument("--loss-presence", type=float, default=0.1)
    parser.add_argument("--loss-speaker-cls", type=float, default=0.2)
    # DataLoader
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    # Output
    parser.add_argument("--out-dir", type=Path, default=Path("./checkpoints"))
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--no-save-every-epoch", action="store_true")
    # Runtime
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
        campplus_projection_type=args.campplus_projection_type,
        campplus_projection_hidden_dim=args.campplus_projection_hidden_dim,
        arcface_scale=args.arcface_scale,
        arcface_margin=args.arcface_margin,
        segment_seconds=args.segment_seconds,
        enrollment_seconds=args.enrollment_seconds,
        snr_range_db=tuple(args.snr_range_db),
        target_present_prob=args.target_present_prob,
        noise_snr_range_db=tuple(args.noise_snr_range_db),
        noise_prob=args.noise_prob,
        reverb_prob=args.reverb_prob,
        rir_pool_size=args.rir_pool_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        samples_per_epoch=args.samples_per_epoch,
        val_samples=args.val_samples,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        use_plateau_scheduler=True,
        plateau_patience=args.plateau_patience,
        plateau_factor=args.plateau_factor,
        early_stopping_patience=args.early_stopping_patience,
        grad_clip=args.grad_clip,
        encoder_pretrain_epochs=args.encoder_pretrain_epochs,
        encoder_pretrain_lr=args.encoder_pretrain_lr,
        encoder_lr_scale=args.encoder_lr_scale,
        enc_channels=args.enc_channels,
        bottleneck_channels=args.bottleneck_channels,
        hidden_channels=args.hidden_channels,
        num_repeats=args.num_repeats,
        r1_blocks=args.r1_blocks,
        r2_blocks=args.r2_blocks,
        s4d_state_dim=args.s4d_state_dim,
        loss_sdr=args.loss_sdr,
        loss_mr_stft=args.loss_mr_stft,
        loss_absent=args.loss_absent,
        loss_presence=args.loss_presence,
        loss_speaker_cls=args.loss_speaker_cls,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        val_speaker_ratio=args.val_speaker_ratio,
        out_dir=args.out_dir,
        log_interval=args.log_interval,
        save_every_epoch=not args.no_save_every_epoch,
        device=args.device,
        seed=args.seed,
        encoder_type=args.encoder_type,
    )


def main() -> None:
    cfg = _parse_args()
    run_training(cfg)


if __name__ == "__main__":
    main()
