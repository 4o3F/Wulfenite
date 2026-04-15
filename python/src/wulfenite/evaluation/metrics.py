"""Objective evaluation metrics for speech enhancement experiments."""

from __future__ import annotations

import importlib
from typing import Any
from typing import Literal

import torch


MissingPolicy = Literal["raise", "nan"]


def _as_batch(waveform: torch.Tensor) -> torch.Tensor:
    if waveform.dim() == 1:
        return waveform.unsqueeze(0)
    if waveform.dim() != 2:
        raise ValueError(f"waveform must be [T] or [B, T], got {tuple(waveform.shape)}")
    return waveform


def _import_optional(module_name: str) -> Any:
    return importlib.import_module(module_name)


def _handle_missing(module_name: str, on_missing: MissingPolicy) -> float:
    if on_missing == "nan":
        return float("nan")
    raise RuntimeError(f"Install `{module_name}` to compute this metric.")


def si_sdr(
    estimate: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute scale-invariant SDR for each sample in a batch."""
    estimate_batch = _as_batch(estimate)
    target_batch = _as_batch(target)
    if estimate_batch.shape != target_batch.shape:
        raise ValueError(
            f"estimate {tuple(estimate_batch.shape)} vs target {tuple(target_batch.shape)}"
        )
    # Zero-mean normalization (standard SI-SDR definition).
    estimate_batch = estimate_batch - estimate_batch.mean(dim=-1, keepdim=True)
    target_batch = target_batch - target_batch.mean(dim=-1, keepdim=True)
    target_energy = target_batch.pow(2).sum(dim=-1, keepdim=True).clamp_min(eps)
    projection = (
        (estimate_batch * target_batch).sum(dim=-1, keepdim=True) / target_energy
    ) * target_batch
    noise = estimate_batch - projection
    ratio = projection.pow(2).sum(dim=-1) / noise.pow(2).sum(dim=-1).clamp_min(eps)
    return 10.0 * torch.log10(ratio.clamp_min(eps))


def pesq_score(
    estimate: torch.Tensor,
    target: torch.Tensor,
    sample_rate: int = 16000,
    on_missing: MissingPolicy = "raise",
) -> float:
    """Compute mean PESQ over a batch, lazily importing the dependency."""
    try:
        pesq_module = _import_optional("pesq")
    except ImportError:
        return _handle_missing("pesq", on_missing)
    estimate_batch = _as_batch(estimate).detach().cpu()
    target_batch = _as_batch(target).detach().cpu()
    mode = "wb" if sample_rate == 16000 else "nb"
    scores = [
        float(pesq_module.pesq(sample_rate, ref.numpy(), deg.numpy(), mode))
        for deg, ref in zip(estimate_batch, target_batch, strict=True)
    ]
    return float(sum(scores) / len(scores))


def stoi_score(
    estimate: torch.Tensor,
    target: torch.Tensor,
    sample_rate: int = 16000,
    on_missing: MissingPolicy = "raise",
) -> float:
    """Compute mean STOI over a batch, lazily importing the dependency."""
    try:
        pystoi_module = _import_optional("pystoi")
    except ImportError:
        return _handle_missing("pystoi", on_missing)
    estimate_batch = _as_batch(estimate).detach().cpu()
    target_batch = _as_batch(target).detach().cpu()
    scores = [
        float(pystoi_module.stoi(ref.numpy(), deg.numpy(), sample_rate, extended=False))
        for deg, ref in zip(estimate_batch, target_batch, strict=True)
    ]
    return float(sum(scores) / len(scores))


def evaluate_pair(
    estimate: torch.Tensor,
    target: torch.Tensor,
    sample_rate: int = 16000,
    on_missing: MissingPolicy = "nan",
) -> dict[str, float]:
    """Compute SI-SDR and, when available, PESQ/STOI for one pair or batch."""
    si_sdr_score = float(si_sdr(estimate, target).mean().item())
    return {
        "si_sdr": si_sdr_score,
        "pesq": pesq_score(estimate, target, sample_rate=sample_rate, on_missing=on_missing),
        "stoi": stoi_score(estimate, target, sample_rate=sample_rate, on_missing=on_missing),
    }


__all__ = ["MissingPolicy", "si_sdr", "pesq_score", "stoi_score", "evaluate_pair"]
