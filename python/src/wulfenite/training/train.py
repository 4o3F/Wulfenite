"""Wulfenite training loop.

Ties together :mod:`wulfenite.data`, :mod:`wulfenite.models`, and
:mod:`wulfenite.losses` into a standard PyTorch training loop with
whole-utterance forward, AdamW + cosine+warmup LR schedule, gradient
clipping, and epoch-level checkpointing.

Call from the command line:

    uv run --directory python python -m wulfenite.training.train \\
        --aishell1-root ~/datasets/aishell1 \\
        --aishell3-root ~/datasets/aishell3 \\
        --noise-root ~/datasets/musan/noise \\
        --campplus-checkpoint ~/datasets/campplus/campplus_cn_common.bin \\
        --out-dir ./checkpoints/phase1

Or from Python (used by tests):

    from wulfenite.training.train import run_training
    from wulfenite.training.config import TrainingConfig

    run_training(TrainingConfig(...), model=my_model)  # model optional
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Callable

import torch
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
    scan_noise_dir,
)
from ..losses import LossWeights, WulfeniteLoss
from ..models import WulfeniteTSE
from .checkpoint import save_checkpoint
from .config import TrainingConfig


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def build_dataset(cfg: TrainingConfig) -> tuple[WulfeniteMixer, WulfeniteMixer]:
    """Scan datasets and assemble train + val mixers."""
    speakers: dict = {}
    if cfg.aishell1_root is not None:
        speakers = scan_aishell1(cfg.aishell1_root)
    if cfg.aishell3_root is not None:
        speakers = merge_speaker_dicts(speakers, scan_aishell3(cfg.aishell3_root))
    if not speakers:
        raise RuntimeError(
            "Training requires at least one of --aishell1-root / "
            "--aishell3-root to produce a non-empty speaker pool."
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
        reverb=ReverbConfig(),
    )
    train_ds = WulfeniteMixer(
        speakers=speakers,
        noise_pool=noise_pool,
        config=mixer_cfg,
        samples_per_epoch=cfg.samples_per_epoch,
        seed=None,  # fresh mixtures per call
    )
    val_ds = WulfeniteMixer(
        speakers=speakers,
        noise_pool=noise_pool,
        config=mixer_cfg,
        samples_per_epoch=cfg.val_samples,
        seed=cfg.seed,  # reproducible validation
    )
    return train_ds, val_ds


def build_model(cfg: TrainingConfig) -> WulfeniteTSE:
    """Build the TSE model, loading the frozen CAM++ checkpoint."""
    if cfg.campplus_checkpoint is None:
        raise RuntimeError(
            "TrainingConfig.campplus_checkpoint must be set; pass "
            "--campplus-checkpoint on the command line."
        )
    return WulfeniteTSE.from_campplus_checkpoint(
        cfg.campplus_checkpoint, device="cpu",
    )


def build_loss(cfg: TrainingConfig) -> WulfeniteLoss:
    return WulfeniteLoss(
        weights=LossWeights(
            sdr=cfg.loss_sdr,
            mr_stft=cfg.loss_mr_stft,
            absent=cfg.loss_absent,
            presence=cfg.loss_presence,
        ),
    )


def build_optimizer(
    model: torch.nn.Module,
    cfg: TrainingConfig,
    total_steps: int,
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
    """AdamW + cosine-with-warmup schedule on the trainable parameters only.

    CAM++ is frozen so its parameters never appear in the optimizer.
    """
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=cfg.lr, weight_decay=cfg.weight_decay,
    )

    warmup_steps = max(1, int(cfg.warmup_ratio * total_steps))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(progress * math.pi))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    return optimizer, scheduler


# ---------------------------------------------------------------------------
# Train / validate epochs
# ---------------------------------------------------------------------------


def _move_batch(batch: dict, device: torch.device, non_blocking: bool) -> dict:
    return {
        k: (v.to(device, non_blocking=non_blocking) if torch.is_tensor(v) else v)
        for k, v in batch.items()
    }


def train_one_epoch(
    model: WulfeniteTSE,
    loader: DataLoader,
    criterion: WulfeniteLoss,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
    cfg: TrainingConfig,
    epoch: int,
    global_step: int,
    log_fn: Callable[[str], None],
    *,
    show_progress: bool = True,
) -> tuple[float, int]:
    """Run one epoch of training. Returns ``(mean_loss, new_global_step)``.

    When ``show_progress`` is true (the CLI default), a tqdm bar shows
    per-step progress plus live loss values via ``set_postfix``. When
    false (tests), the inner loop is silent — periodic log lines still
    go to the log file at ``cfg.log_interval``.
    """
    model.train()
    # CAM++ stays in eval mode — it is frozen and we do not want BN /
    # dropout to behave as if we were training it.
    model.speaker_encoder.eval()

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

        outputs = model(batch["mixture"], batch["enrollment"])
        loss, parts = criterion(
            clean=outputs["clean"],
            target=batch["target"],
            mixture=batch["mixture"],
            target_present=batch["target_present"],
            presence_logit=outputs.get("presence_logit"),
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                max_norm=cfg.grad_clip,
            )
        optimizer.step()
        scheduler.step()

        total_loss += parts.total
        n_batches += 1
        global_step += 1

        pbar.set_postfix(
            loss=f"{parts.total:+.3f}",
            sdr=f"{parts.sdr:+.2f}",
            stft=f"{parts.mr_stft:.3f}",
            lr=f"{optimizer.param_groups[0]['lr']:.1e}",
            refresh=False,
        )

        if global_step % cfg.log_interval == 0:
            log_fn(
                f"epoch {epoch} step {step + 1}/{len(loader)} "
                f"loss={parts.total:+.4f} sdr={parts.sdr:+.3f} "
                f"stft={parts.mr_stft:.4f} absent={parts.absent:.4f} "
                f"presence={parts.presence:.4f} "
                f"present={parts.n_present}/{parts.n_present + parts.n_absent} "
                f"lr={optimizer.param_groups[0]['lr']:.2e}"
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
    sdr_sum = 0.0
    stft_sum = 0.0
    absent_sum = 0.0
    presence_sum = 0.0
    n = 0
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
        outputs = model(batch["mixture"], batch["enrollment"])
        _, parts = criterion(
            clean=outputs["clean"],
            target=batch["target"],
            mixture=batch["mixture"],
            target_present=batch["target_present"],
            presence_logit=outputs.get("presence_logit"),
        )
        total += parts.total
        sdr_sum += parts.sdr
        stft_sum += parts.mr_stft
        absent_sum += parts.absent
        presence_sum += parts.presence
        n += 1

    n_safe = max(1, n)
    return total / n_safe, {
        "sdr": sdr_sum / n_safe,
        "mr_stft": stft_sum / n_safe,
        "absent": absent_sum / n_safe,
        "presence": presence_sum / n_safe,
    }


# ---------------------------------------------------------------------------
# Top-level entrypoint
# ---------------------------------------------------------------------------


def run_training(
    cfg: TrainingConfig,
    *,
    model: WulfeniteTSE | None = None,
    show_progress: bool = True,
) -> None:
    """Run the full training loop described by ``cfg``.

    Args:
        cfg: populated :class:`TrainingConfig`.
        model: optional pre-built model. If omitted, the model is
            built from ``cfg.campplus_checkpoint`` via
            :func:`build_model`. Tests pass a pre-built model to
            avoid needing the real CAM++ weights on disk.
    """
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
        model = build_model(cfg)
    model = model.to(device)

    criterion = build_loss(cfg).to(device)

    total_steps = cfg.epochs * len(train_loader)
    optimizer, scheduler = build_optimizer(model, cfg, total_steps)

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train.log"

    def log(msg: str) -> None:
        stamped = f"[{time.strftime('%H:%M:%S')}] {msg}"
        # Use tqdm.write so the message plays nicely with active progress
        # bars — it clears the current bar line, prints the message, then
        # redraws the bar below.
        tqdm.write(stamped)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(stamped + "\n")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"device={device} trainable_params={trainable}")
    log(
        f"train_steps_per_epoch={len(train_loader)} "
        f"val_steps={len(val_loader)} total_steps={total_steps}"
    )

    best_val = float("inf")
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
            model, train_loader, criterion, optimizer, scheduler,
            device, cfg, epoch, global_step, log,
            show_progress=show_progress,
        )
        train_time = time.time() - epoch_t0

        val_loss, val_parts = validate(
            model, val_loader, criterion, device, show_progress=show_progress,
        )
        log(
            f"epoch {epoch} train_loss={train_loss:+.4f} val_loss={val_loss:+.4f} "
            f"val_sdr={val_parts['sdr']:+.3f} val_stft={val_parts['mr_stft']:.4f} "
            f"val_absent={val_parts['absent']:.4f} time={train_time:.1f}s"
        )

        metrics = {
            "train_loss": train_loss,
            "val_loss": val_loss,
            **{f"val_{k}": v for k, v in val_parts.items()},
        }
        if cfg.save_every_epoch:
            save_checkpoint(
                out_dir / f"epoch{epoch:03d}.pt",
                model=model, optimizer=optimizer, scheduler=scheduler,
                epoch=epoch, step=global_step, config=cfg, metrics=metrics,
            )
        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(
                out_dir / "best.pt",
                model=model, optimizer=optimizer, scheduler=scheduler,
                epoch=epoch, step=global_step, config=cfg, metrics=metrics,
            )
            log(f"[best] val_loss improved to {best_val:+.4f} → saved best.pt")

        epoch_iter.set_postfix(
            train=f"{train_loss:+.3f}",
            val=f"{val_loss:+.3f}",
            best=f"{best_val:+.3f}",
            refresh=False,
        )
    epoch_iter.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> TrainingConfig:
    parser = argparse.ArgumentParser(description="Train Wulfenite SpeakerBeam-SS")
    # Data
    parser.add_argument("--aishell1-root", type=Path, default=None)
    parser.add_argument("--aishell3-root", type=Path, default=None)
    parser.add_argument("--noise-root", type=Path, default=None)
    parser.add_argument("--campplus-checkpoint", type=Path, required=True)
    # Mixer
    parser.add_argument("--segment-seconds", type=float, default=4.0)
    parser.add_argument("--enrollment-seconds", type=float, default=4.0)
    parser.add_argument("--snr-range-db", type=float, nargs=2, default=(-5.0, 5.0))
    parser.add_argument("--target-present-prob", type=float, default=0.85)
    parser.add_argument("--noise-snr-range-db", type=float, nargs=2, default=(10.0, 25.0))
    parser.add_argument("--noise-prob", type=float, default=0.80)
    parser.add_argument("--reverb-prob", type=float, default=0.85)
    # Optimizer
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--samples-per-epoch", type=int, default=20000)
    parser.add_argument("--val-samples", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    # Loss
    parser.add_argument("--loss-sdr", type=float, default=1.0)
    parser.add_argument("--loss-mr-stft", type=float, default=1.0)
    parser.add_argument("--loss-absent", type=float, default=1.0)
    parser.add_argument("--loss-presence", type=float, default=0.1)
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
        noise_root=args.noise_root,
        campplus_checkpoint=args.campplus_checkpoint,
        segment_seconds=args.segment_seconds,
        enrollment_seconds=args.enrollment_seconds,
        snr_range_db=tuple(args.snr_range_db),
        target_present_prob=args.target_present_prob,
        noise_snr_range_db=tuple(args.noise_snr_range_db),
        noise_prob=args.noise_prob,
        reverb_prob=args.reverb_prob,
        batch_size=args.batch_size,
        epochs=args.epochs,
        samples_per_epoch=args.samples_per_epoch,
        val_samples=args.val_samples,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        grad_clip=args.grad_clip,
        loss_sdr=args.loss_sdr,
        loss_mr_stft=args.loss_mr_stft,
        loss_absent=args.loss_absent,
        loss_presence=args.loss_presence,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
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
