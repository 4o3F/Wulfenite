"""Training loop for pDFNet2 and pDFNet2+."""

from __future__ import annotations

from collections.abc import Iterable
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from wulfenite.losses import PDfNet2Loss
from wulfenite.models import DfNet, PDfNet2, PDfNet2Plus, SpeakerEncoder

from .config import TrainConfig


def _resolve_device(device: str | None) -> torch.device:
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def scheduled_batch_size(config: TrainConfig, epoch: int) -> int:
    if config.batch_size_ramp_epochs <= 1:
        return config.batch_size_end
    progress = min(epoch, config.batch_size_ramp_epochs - 1) / (config.batch_size_ramp_epochs - 1)
    size = config.batch_size_start + progress * (config.batch_size_end - config.batch_size_start)
    return int(round(size))


def _build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    config: TrainConfig,
) -> torch.optim.lr_scheduler.LambdaLR | None:
    if config.lr_scheduler == "none":
        return None
    warmup_epochs = min(max(0, config.lr_warmup_epochs), max(config.max_epochs - 1, 0))
    min_ratio = min(max(config.lr_min_ratio, 0.0), 1.0)

    def lr_lambda(epoch: int) -> float:
        if warmup_epochs > 0 and epoch < warmup_epochs:
            return float(epoch + 1) / float(warmup_epochs)
        decay_epochs = max(1, config.max_epochs - warmup_epochs - 1)
        progress = min(max(epoch - warmup_epochs, 0), decay_epochs) / decay_epochs
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_ratio + (1.0 - min_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def _unpack_batch(
    batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mixture, target, enrollment, _speaker_ids = batch
    return mixture, target, enrollment


def _compute_embedding(
    model: torch.nn.Module,
    enrollment: torch.Tensor,
    speaker_encoder: SpeakerEncoder | None,
) -> torch.Tensor | None:
    if isinstance(model, DfNet) and not isinstance(model, PDfNet2):
        return None
    if speaker_encoder is None:
        raise RuntimeError(
            "A speaker_encoder is required when training PDfNet2 or PDfNet2Plus."
        )
    with torch.no_grad():
        return speaker_encoder(enrollment)


def _forward_model(
    model: torch.nn.Module,
    mixture: torch.Tensor,
    speaker_emb: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(model, PDfNet2Plus):
        assert speaker_emb is not None
        enhanced_spec, _gains, _lsnr, _alpha = model(mixture, speaker_emb)
        enhanced_waveform = model.pdfnet2.spec_to_waveform(
            enhanced_spec,
            length=mixture.size(-1),
        )
        return enhanced_waveform, enhanced_spec
    if isinstance(model, PDfNet2):
        spec, _ = model.waveform_to_spec(mixture)
        enhanced_spec, _gains, _lsnr, _alpha = model(spec, speaker_emb)
        enhanced_waveform = model.spec_to_waveform(enhanced_spec, length=mixture.size(-1))
        return enhanced_waveform, enhanced_spec
    if isinstance(model, DfNet):
        spec, _ = model.waveform_to_spec(mixture)
        enhanced_spec, _gains, _lsnr, _alpha = model(spec)
        enhanced_waveform = model.spec_to_waveform(enhanced_spec, length=mixture.size(-1))
        return enhanced_waveform, enhanced_spec
    raise TypeError(f"Unsupported model type: {type(model)!r}")


def run_pdfnet2_epoch(
    model: torch.nn.Module,
    dataloader: Iterable[tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str]]],
    *,
    loss_fn: PDfNet2Loss,
    device: torch.device,
    speaker_encoder: SpeakerEncoder | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    grad_clip_norm: float = 5.0,
    max_steps: int | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    if speaker_encoder is not None:
        speaker_encoder.eval()

    totals = {
        "loss": 0.0,
        "spectral": 0.0,
        "multi_res": 0.0,
        "over_suppression": 0.0,
    }
    steps = 0
    phase = "train" if training else "val"
    pbar = tqdm(dataloader, desc=phase, leave=False, dynamic_ncols=True)
    for batch in pbar:
        mixture, target, enrollment = _unpack_batch(batch)
        mixture = mixture.to(device)
        target = target.to(device)
        enrollment = enrollment.to(device)

        speaker_emb = _compute_embedding(model, enrollment, speaker_encoder)
        if speaker_emb is not None:
            speaker_emb = speaker_emb.to(device)

        if training:
            assert optimizer is not None
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            enhanced_waveform, enhanced_spec = _forward_model(model, mixture, speaker_emb)
            target_spec_source = model.pdfnet2 if isinstance(model, PDfNet2Plus) else model
            if not isinstance(target_spec_source, DfNet):
                raise TypeError(f"Unsupported model type: {type(target_spec_source)!r}")
            target_spec, _ = target_spec_source.waveform_to_spec(target)
            total_loss, terms = loss_fn(
                enhanced_waveform,
                target,
                estimate_spec=enhanced_spec.squeeze(1),
                target_spec=target_spec.squeeze(1),
            )
            if training:
                assert optimizer is not None
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                optimizer.step()

        loss_val = float(total_loss.detach().cpu())
        totals["loss"] += loss_val
        totals["spectral"] += float(terms["spectral"].detach().cpu())
        totals["multi_res"] += float(terms["multi_res"].detach().cpu())
        totals["over_suppression"] += float(terms["over_suppression"].detach().cpu())
        steps += 1
        pbar.set_postfix(loss=f"{loss_val:.4f}")
        if max_steps is not None and steps >= max_steps:
            break
    pbar.close()

    if steps == 0:
        raise RuntimeError("run_pdfnet2_epoch received an empty dataloader")
    return {name: value / steps for name, value in totals.items()}


def train_pdfnet2(
    model: torch.nn.Module,
    train_dataset: torch.utils.data.Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]],
    val_dataset: torch.utils.data.Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]],
    config: TrainConfig,
    *,
    speaker_encoder: SpeakerEncoder | None = None,
) -> list[dict[str, float]]:
    device = _resolve_device(config.device)
    model.to(device)
    if speaker_encoder is not None:
        speaker_encoder.to(device)
        speaker_encoder.eval()

    loss_fn = PDfNet2Loss(
        lambda_spec=config.lambda_spec,
        lambda_mr=config.lambda_mr,
        lambda_os=config.lambda_os,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = _build_lr_scheduler(optimizer, config)

    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    stale_epochs = 0
    history: list[dict[str, float]] = []

    epoch_pbar = tqdm(range(config.max_epochs), desc="epochs", dynamic_ncols=True)
    for epoch in epoch_pbar:
        set_epoch = getattr(train_dataset, "set_epoch", None)
        if callable(set_epoch):
            set_epoch(epoch)
        batch_size = scheduled_batch_size(config, epoch)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=config.num_workers,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=config.num_workers,
        )

        train_metrics = run_pdfnet2_epoch(
            model,
            train_loader,
            loss_fn=loss_fn,
            device=device,
            speaker_encoder=speaker_encoder,
            optimizer=optimizer,
            grad_clip_norm=config.grad_clip_norm,
            max_steps=config.max_steps_per_epoch,
        )
        val_metrics = run_pdfnet2_epoch(
            model,
            val_loader,
            loss_fn=loss_fn,
            device=device,
            speaker_encoder=speaker_encoder,
            optimizer=None,
            grad_clip_norm=config.grad_clip_norm,
            max_steps=config.max_steps_per_epoch,
        )

        record = {
            "epoch": float(epoch),
            "batch_size": float(batch_size),
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train_loss": train_metrics["loss"],
            "val_loss": val_metrics["loss"],
            "train_spectral": train_metrics["spectral"],
            "val_spectral": val_metrics["spectral"],
            "train_multi_res": train_metrics["multi_res"],
            "val_multi_res": val_metrics["multi_res"],
            "train_over_suppression": train_metrics["over_suppression"],
            "val_over_suppression": val_metrics["over_suppression"],
        }
        history.append(record)
        epoch_pbar.set_postfix(
            train=f"{train_metrics['loss']:.4f}",
            val=f"{val_metrics['loss']:.4f}",
            lr=f"{optimizer.param_groups[0]['lr']:.2e}",
        )

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "history": history,
        }
        torch.save(checkpoint, config.checkpoint_dir / "last.pt")

        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            stale_epochs = 0
            torch.save(checkpoint, config.checkpoint_dir / "best.pt")
        else:
            stale_epochs += 1

        if scheduler is not None:
            scheduler.step()
        if stale_epochs >= config.patience:
            break
    return history


__all__ = ["scheduled_batch_size", "run_pdfnet2_epoch", "train_pdfnet2"]
