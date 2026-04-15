"""Contrastive KD training loop for TinyECAPA."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import random
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from wulfenite.data import ReverbConfig, add_noise_at_snr, apply_rir, synth_room_rir
from wulfenite.models import TinyECAPA

from .config import TrainConfig
from .train_pdfnet2 import _build_lr_scheduler


class ContrastiveKDLoss(nn.Module):
    """CLAP-style contrastive KD objective for chunk embeddings."""

    def __init__(self, init_temperature: float = 10.0) -> None:
        super().__init__()
        self.log_temperature = nn.Parameter(torch.log(torch.tensor(float(init_temperature))))

    @property
    def temperature(self) -> torch.Tensor:
        return self.log_temperature.exp()

    def forward(
        self,
        teacher_embeddings: torch.Tensor,
        student_chunk_embeddings: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if teacher_embeddings.dim() != 2:
            raise ValueError(
                f"teacher_embeddings must be [N, D], got {tuple(teacher_embeddings.shape)}"
            )
        if student_chunk_embeddings.dim() != 3:
            raise ValueError(
                "student_chunk_embeddings must be [N, D, T'], "
                f"got {tuple(student_chunk_embeddings.shape)}"
            )
        teacher = F.normalize(teacher_embeddings, dim=-1)
        student = F.normalize(student_chunk_embeddings, dim=1)
        similarity = torch.einsum("nd,mdt->nmt", teacher, student).mean(dim=-1)
        logits = similarity * self.temperature
        labels = torch.arange(logits.size(0), device=logits.device)
        loss = 0.5 * (
            F.cross_entropy(logits, labels) +
            F.cross_entropy(logits.transpose(0, 1), labels)
        )
        return loss, logits


def augment_speaker_batch(
    waveform: torch.Tensor,
    *,
    noise_bank: list[torch.Tensor] | None = None,
    reverb_config: ReverbConfig | None = None,
    reverb_probability: float = 0.5,
    noise_snr_range: tuple[float, float] = (0.0, 20.0),
    seed: int | None = None,
) -> torch.Tensor:
    """Apply lightweight MUSAN/RIR-style augmentation to a batch."""
    if waveform.dim() != 2:
        raise ValueError(f"waveform must be [B, T], got {tuple(waveform.shape)}")
    rng = random.Random(seed)
    out = waveform.clone()
    reverb_config = reverb_config if reverb_config is not None else ReverbConfig()
    for idx in range(out.size(0)):
        sample = out[idx]
        if rng.random() < reverb_probability:
            sample = apply_rir(sample, synth_room_rir(reverb_config, rng))
        if noise_bank:
            noise = noise_bank[rng.randrange(len(noise_bank))].to(sample.device, sample.dtype)
            snr_db = rng.uniform(*noise_snr_range)
            sample = add_noise_at_snr(sample, noise, snr_db)
        out[idx] = sample
    return out


def _resolve_device(device: str | None) -> torch.device:
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _unpack_batch(batch: Any) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(batch, dict):
        return batch["student_waveform"], batch["teacher_waveform"]
    if isinstance(batch, (list, tuple)) and len(batch) >= 2:
        student, teacher = batch[:2]
        return student, teacher
    raise TypeError(
        "TinyECAPA dataloader batches must be a dict with "
        "'student_waveform'/'teacher_waveform' keys or a tuple "
        "(student_waveform, teacher_waveform, ...)."
    )


def _is_raw_state_dict(checkpoint: object) -> bool:
    if not isinstance(checkpoint, Mapping):
        return False
    if not checkpoint:
        return False
    return all(isinstance(key, str) and isinstance(value, torch.Tensor) for key, value in checkpoint.items())


def load_tiny_ecapa_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
    model_kwargs: dict[str, Any] | None = None,
) -> TinyECAPA:
    """Load a TinyECAPA checkpoint saved by ``train_tiny_ecapa``."""
    checkpoint = torch.load(path, map_location=map_location, weights_only=True)
    saved_kwargs: Mapping[str, Any]
    if isinstance(checkpoint, Mapping) and "student_state_dict" in checkpoint:
        state_dict = checkpoint["student_state_dict"]
        saved_kwargs = checkpoint.get("model_kwargs", {})
    elif _is_raw_state_dict(checkpoint):
        state_dict = checkpoint
        saved_kwargs = {}
    else:
        raise ValueError(
            "TinyECAPA checkpoint must contain a 'student_state_dict' entry or a raw state dict."
        )
    if not _is_raw_state_dict(state_dict):
        raise ValueError("TinyECAPA checkpoint contains an invalid student_state_dict payload.")
    if not isinstance(saved_kwargs, Mapping):
        raise ValueError("TinyECAPA checkpoint model_kwargs must be a mapping.")
    merged_kwargs = {**dict(saved_kwargs), **(model_kwargs or {})}
    model = TinyECAPA(**merged_kwargs)
    try:
        model.load_state_dict(dict(state_dict), strict=True)
    except RuntimeError as exc:
        raise ValueError("TinyECAPA checkpoint does not match the expected architecture.") from exc
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


def run_tiny_ecapa_epoch(
    student: TinyECAPA,
    teacher: nn.Module,
    dataloader: torch.utils.data.DataLoader[object],
    *,
    loss_fn: ContrastiveKDLoss,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    noise_bank: list[torch.Tensor] | None = None,
    reverb_config: ReverbConfig | None = None,
    apply_augmentation: bool = False,
    augment_seed: int | None = None,
    reverb_probability: float = 0.5,
    noise_snr_range: tuple[float, float] = (0.0, 20.0),
    chunk_seconds: float = 1.0,
    overlap: float = 0.5,
    max_steps: int | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    student.train(training)
    teacher.eval()
    loss_fn.train(training)

    total_loss = 0.0
    steps = 0
    for batch in dataloader:
        student_waveform, teacher_waveform = _unpack_batch(batch)
        student_waveform = student_waveform.to(device)
        teacher_waveform = teacher_waveform.to(device)
        if training and apply_augmentation:
            batch_seed = None if augment_seed is None else augment_seed + steps
            student_waveform = augment_speaker_batch(
                student_waveform,
                noise_bank=noise_bank,
                reverb_config=reverb_config,
                reverb_probability=reverb_probability,
                noise_snr_range=noise_snr_range,
                seed=batch_seed,
            )

        if training:
            assert optimizer is not None
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            with torch.no_grad():
                teacher_emb = teacher(teacher_waveform)
            student_chunks = student.forward_chunks(
                student_waveform,
                chunk_seconds=chunk_seconds,
                overlap=overlap,
            )
            loss, _logits = loss_fn(teacher_emb, student_chunks)
            if training:
                assert optimizer is not None
                loss.backward()
                optimizer.step()

        total_loss += float(loss.detach().cpu())
        steps += 1
        if max_steps is not None and steps >= max_steps:
            break

    if steps == 0:
        raise RuntimeError("run_tiny_ecapa_epoch received an empty dataloader")
    return {"loss": total_loss / steps}


def train_tiny_ecapa(
    student: TinyECAPA,
    teacher: nn.Module,
    train_loader: torch.utils.data.DataLoader[object],
    val_loader: torch.utils.data.DataLoader[object],
    config: TrainConfig,
    *,
    noise_bank: list[torch.Tensor] | None = None,
    reverb_config: ReverbConfig | None = None,
) -> list[dict[str, float]]:
    device = _resolve_device(config.device)
    student.to(device)
    teacher.to(device)
    teacher.eval()

    loss_fn = ContrastiveKDLoss(config.tiny_ecapa_temperature_init).to(device)
    optimizer = torch.optim.Adam(
        list(student.parameters()) + list(loss_fn.parameters()),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = _build_lr_scheduler(optimizer, config)

    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, float]] = []
    best_val = float("inf")
    stale_epochs = 0

    for epoch in range(config.max_epochs):
        set_epoch = getattr(train_loader.dataset, "set_epoch", None)
        if callable(set_epoch):
            set_epoch(epoch)
        train_metrics = run_tiny_ecapa_epoch(
            student,
            teacher,
            train_loader,
            loss_fn=loss_fn,
            device=device,
            optimizer=optimizer,
            noise_bank=noise_bank,
            reverb_config=reverb_config,
            apply_augmentation=config.tiny_ecapa_apply_augmentation,
            augment_seed=epoch,
            reverb_probability=config.tiny_ecapa_reverb_probability,
            noise_snr_range=config.tiny_ecapa_noise_snr_range,
            chunk_seconds=config.tiny_ecapa_chunk_seconds,
            overlap=config.tiny_ecapa_chunk_overlap,
            max_steps=config.max_steps_per_epoch,
        )
        val_metrics = run_tiny_ecapa_epoch(
            student,
            teacher,
            val_loader,
            loss_fn=loss_fn,
            device=device,
            optimizer=None,
            noise_bank=noise_bank,
            reverb_config=reverb_config,
            apply_augmentation=False,
            reverb_probability=config.tiny_ecapa_reverb_probability,
            noise_snr_range=config.tiny_ecapa_noise_snr_range,
            chunk_seconds=config.tiny_ecapa_chunk_seconds,
            overlap=config.tiny_ecapa_chunk_overlap,
            max_steps=config.max_steps_per_epoch,
        )

        record = {
            "epoch": float(epoch),
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train_loss": train_metrics["loss"],
            "val_loss": val_metrics["loss"],
            "temperature": float(loss_fn.temperature.detach().cpu()),
        }
        history.append(record)

        checkpoint = {
            "checkpoint_type": "tiny_ecapa_kd",
            "epoch": epoch,
            "format_version": 1,
            "model_kwargs": {"sample_rate": student.sample_rate},
            "student_state_dict": student.state_dict(),
            "loss_state_dict": loss_fn.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "history": history,
        }
        torch.save(checkpoint, config.checkpoint_dir / "tiny_ecapa_last.pt")

        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            stale_epochs = 0
            torch.save(checkpoint, config.checkpoint_dir / "tiny_ecapa_best.pt")
        else:
            stale_epochs += 1

        if scheduler is not None:
            scheduler.step()
        if stale_epochs >= config.patience:
            break
    return history


__all__ = [
    "ContrastiveKDLoss",
    "augment_speaker_batch",
    "load_tiny_ecapa_checkpoint",
    "run_tiny_ecapa_epoch",
    "train_tiny_ecapa",
]
