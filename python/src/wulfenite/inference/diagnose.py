"""Offline checkpoint diagnostics for Wulfenite TSE models.

Runs a trained checkpoint against deterministic ``WulfeniteMixer``
samples and prints a structured report focused on over-suppression and
target dropout.

Usage:

    uv run --directory python python -m wulfenite.inference.diagnose \
        --checkpoint ./assets/best.pt \
        --aishell1-root /path/to/aishell1 \
        --magicdata-root /path/to/magicdata \
        --cnceleb-root /path/to/cnceleb \
        --num-samples 200 \
        --device cuda
"""

from __future__ import annotations

import argparse
import contextlib
import io
import math
import pathlib
import platform
import random
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from ..data import (
    MixerConfig,
    WulfeniteMixer,
    merge_speaker_dicts,
    scan_aishell1,
    scan_cnceleb,
    scan_magicdata,
)
from ..audio_features import compute_fbank_batch
from ..losses.sdr import compute_sdr_db
from .utils import build_model_from_checkpoint


# ---------------------------------------------------------------------------
# Cross-platform checkpoint compatibility shim
# ---------------------------------------------------------------------------
# Checkpoints saved on Linux may contain ``pathlib.PosixPath`` objects
# inside the pickled ``config`` dict. Aliasing the class here lets them
# unpickle on Windows as well.
if platform.system() == "Windows":
    pathlib.PosixPath = pathlib.WindowsPath  # type: ignore[misc, assignment]


DEFAULT_SEED = 1234
ACTIVE_FRAME_DROP_DB = 40.0
MASK_HISTOGRAM_BINS = 2048
MASK_NEAR_ZERO = 0.1
MASK_NEAR_ONE = 0.9
EPS = 1e-8


class MaskValueStats:
    """Streaming histogram-based mask statistics over values in ``[0, 1]``."""

    def __init__(
        self,
        *,
        bins: int = MASK_HISTOGRAM_BINS,
        low: float = 0.0,
        high: float = 1.0,
        near_zero: float = MASK_NEAR_ZERO,
        near_one: float = MASK_NEAR_ONE,
    ) -> None:
        self.bins = bins
        self.low = low
        self.high = high
        self.near_zero = near_zero
        self.near_one = near_one
        self.hist = torch.zeros(bins, dtype=torch.float64)
        self.count = 0
        self.sum = 0.0
        self.sum_sq = 0.0
        self.lt_near_zero = 0
        self.gt_near_one = 0

    def update(self, values: torch.Tensor) -> None:
        """Accumulate a tensor of mask values."""
        if values.numel() == 0:
            return
        flat = values.detach().reshape(-1)
        self.count += int(flat.numel())
        flat64 = flat.to(torch.float64)
        self.sum += float(flat64.sum().item())
        self.sum_sq += float((flat64 * flat64).sum().item())
        self.lt_near_zero += int((flat < self.near_zero).sum().item())
        self.gt_near_one += int((flat > self.near_one).sum().item())
        hist = torch.histc(
            flat.to(torch.float32),
            bins=self.bins,
            min=self.low,
            max=self.high,
        )
        self.hist += hist.cpu().to(torch.float64)

    def _percentile(self, q: float) -> float | None:
        if self.count == 0:
            return None
        cdf = torch.cumsum(self.hist, dim=0)
        rank = max(1.0, q * self.count)
        idx = int(torch.searchsorted(cdf, rank, right=False).item())
        idx = max(0, min(self.bins - 1, idx))
        bin_width = (self.high - self.low) / self.bins
        return self.low + (idx + 0.5) * bin_width

    def summary(self) -> dict[str, float | int | None]:
        if self.count == 0:
            return {
                "count": 0,
                "mean": None,
                "std": None,
                "p05": None,
                "p25": None,
                "p50": None,
                "p75": None,
                "p95": None,
                "lt_near_zero_pct": None,
                "gt_near_one_pct": None,
            }
        mean = self.sum / self.count
        variance = max(0.0, self.sum_sq / self.count - mean * mean)
        return {
            "count": self.count,
            "mean": mean,
            "std": math.sqrt(variance),
            "p05": self._percentile(0.05),
            "p25": self._percentile(0.25),
            "p50": self._percentile(0.50),
            "p75": self._percentile(0.75),
            "p95": self._percentile(0.95),
            "lt_near_zero_pct": 100.0 * self.lt_near_zero / self.count,
            "gt_near_one_pct": 100.0 * self.gt_near_one / self.count,
        }


class MaskCapture:
    """Forward-hook sink for the separator's post-sigmoid mask tensor."""

    def __init__(self) -> None:
        self.tensor: torch.Tensor | None = None

    def __call__(
        self,
        _module: torch.nn.Module,
        _inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        self.tensor = output.detach()

    def pop(self) -> torch.Tensor:
        if self.tensor is None:
            raise RuntimeError("Mask hook did not capture any tensor.")
        tensor = self.tensor
        self.tensor = None
        return tensor


def _count_utterances(speakers: dict[str, list[Any]]) -> int:
    return sum(len(utts) for utts in speakers.values())


def _split_speakers_for_eval(
    speakers: dict[str, list[Any]],
    *,
    val_ratio: float,
    seed: int,
) -> tuple[dict[str, list[Any]], str]:
    """Mirror the training speaker-disjoint validation split when possible."""
    speaker_ids = sorted(speakers.keys())
    if len(speaker_ids) < 4:
        return speakers, "all_speakers_fallback"

    shuffled = list(speaker_ids)
    rng = random.Random(seed)
    rng.shuffle(shuffled)

    n_val = max(2, int(len(speaker_ids) * val_ratio))
    n_val = min(n_val, len(speaker_ids) - 2)
    if n_val < 2 or len(speaker_ids) - n_val < 2:
        return speakers, "all_speakers_fallback"

    val_ids = set(shuffled[:n_val])
    val_speakers = {k: v for k, v in speakers.items() if k in val_ids}
    return val_speakers, "speaker_disjoint_val_split"


def _build_speaker_pool(args: argparse.Namespace) -> dict[str, list[Any]]:
    speakers = scan_aishell1(
        args.aishell1_root,
        splits=("train", "dev", "test"),
    )
    if args.magicdata_root is not None:
        speakers = merge_speaker_dicts(
            speakers,
            scan_magicdata(args.magicdata_root, splits=("train", "dev", "test")),
        )
    if args.cnceleb_root is not None:
        speakers = merge_speaker_dicts(speakers, scan_cnceleb(args.cnceleb_root))
    if not speakers:
        raise RuntimeError(
            "No speakers found. Provide at least --aishell1-root with valid 16 kHz "
            "mono speech data."
        )
    return speakers


def _mixer_config_from_checkpoint(checkpoint_cfg: dict[str, Any]) -> MixerConfig:
    """Build a deterministic diagnostic mixer configuration.

    Reuses the checkpoint's core validation settings where available,
    but disables additive noise because the diagnostic CLI does not
    accept a separate noise corpus path and synthetic fallback noise
    would make attenuation analysis harder to interpret.
    """
    reverb_prob = float(checkpoint_cfg.get("reverb_prob", 0.85))
    if "enrollment_seconds_range" in checkpoint_cfg:
        seconds_range = tuple(checkpoint_cfg["enrollment_seconds_range"])
        enrollment_seconds = float(max(seconds_range))
    else:
        enrollment_seconds = float(checkpoint_cfg.get("enrollment_seconds", 4.0))
    return MixerConfig(
        segment_seconds=float(checkpoint_cfg.get("segment_seconds", 4.0)),
        enrollment_seconds_range=(enrollment_seconds, enrollment_seconds),
        snr_range_db=tuple(checkpoint_cfg.get("snr_range_db", (-5.0, 5.0))),
        target_present_prob=float(checkpoint_cfg.get("target_present_prob", 0.85)),
        apply_reverb=reverb_prob > 0.0,
        reverb_prob=reverb_prob,
        rir_pool_size=int(checkpoint_cfg.get("rir_pool_size", 1000)),
        apply_noise=False,
        noise_prob=0.0,
    )


def _frame_energy(
    waveform: torch.Tensor,
    *,
    frame_size: int,
    hop_size: int,
    pad_left: int,
) -> torch.Tensor:
    """Compute 20 ms / 10 ms-aligned frame energy matching separator frames."""
    x = waveform.unsqueeze(0).unsqueeze(0)
    x = F.pad(x, (pad_left, 0))
    frames = x.unfold(-1, frame_size, hop_size)
    return frames.square().mean(dim=-1).squeeze(0).squeeze(0)


def _target_active_mask(frame_energy: torch.Tensor) -> torch.Tensor:
    """Label target-active frames by a per-sample energy peak threshold."""
    if frame_energy.numel() == 0:
        return torch.zeros(0, dtype=torch.bool, device=frame_energy.device)
    peak = float(frame_energy.max().item())
    if peak <= 0.0:
        return torch.zeros_like(frame_energy, dtype=torch.bool)
    threshold = peak * (10.0 ** (-ACTIVE_FRAME_DROP_DB / 10.0))
    threshold = max(threshold, EPS)
    return frame_energy >= threshold


def _cosine_gap(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
) -> dict[str, float | int]:
    if embeddings.numel() == 0 or embeddings.size(0) < 2:
        return {
            "same_cos": 0.0,
            "diff_cos": 0.0,
            "cos_gap": 0.0,
            "same_pairs": 0,
            "diff_pairs": 0,
            "dim": embeddings.size(-1) if embeddings.ndim == 2 else 0,
        }

    emb = F.normalize(embeddings, p=2, dim=-1)
    sim = emb @ emb.T
    same = labels.unsqueeze(0).eq(labels.unsqueeze(1))
    eye = torch.eye(sim.size(0), device=sim.device, dtype=torch.bool)
    same = same & ~eye
    diff = (~same) & ~eye

    same_vals = sim[same]
    diff_vals = sim[diff]
    same_mean = float(same_vals.mean().item()) if same_vals.numel() else 0.0
    diff_mean = float(diff_vals.mean().item()) if diff_vals.numel() else 0.0
    return {
        "same_cos": same_mean,
        "diff_cos": diff_mean,
        "cos_gap": same_mean - diff_mean,
        "same_pairs": int(same_vals.numel()),
        "diff_pairs": int(diff_vals.numel()),
        "dim": int(emb.size(-1)),
    }


def _summarize_tensor(values: torch.Tensor) -> dict[str, float | int | None]:
    flat = values.reshape(-1).to(torch.float64)
    if flat.numel() == 0:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "p05": None,
            "p25": None,
            "p50": None,
            "p75": None,
            "p95": None,
        }
    qs = torch.quantile(
        flat,
        torch.tensor([0.05, 0.25, 0.50, 0.75, 0.95], dtype=torch.float64),
    )
    return {
        "count": int(flat.numel()),
        "mean": float(flat.mean().item()),
        "std": float(flat.std(unbiased=False).item()),
        "p05": float(qs[0].item()),
        "p25": float(qs[1].item()),
        "p50": float(qs[2].item()),
        "p75": float(qs[3].item()),
        "p95": float(qs[4].item()),
    }


def _fmt(value: float | int | None, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return str(value)
    return f"{value:.{digits}f}"


def _fmt_pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.1f}%"


def _format_summary(name: str, summary: dict[str, float | int | None]) -> str:
    return (
        f"{name}: n={_fmt(summary['count'], 0)} "
        f"mean={_fmt(summary['mean'])} std={_fmt(summary['std'])} "
        f"p05={_fmt(summary['p05'])} p25={_fmt(summary['p25'])} "
        f"p50={_fmt(summary['p50'])} p75={_fmt(summary['p75'])} "
        f"p95={_fmt(summary['p95'])}"
    )


def _format_mask_summary(name: str, summary: dict[str, float | int | None]) -> str:
    return (
        f"{name}: n={_fmt(summary['count'], 0)} "
        f"mean={_fmt(summary['mean'])} std={_fmt(summary['std'])} "
        f"p05={_fmt(summary['p05'])} p25={_fmt(summary['p25'])} "
        f"p50={_fmt(summary['p50'])} p75={_fmt(summary['p75'])} "
        f"p95={_fmt(summary['p95'])} "
        f"<0.1={_fmt_pct(summary['lt_near_zero_pct'])} "
        f">0.9={_fmt_pct(summary['gt_near_one_pct'])}"
    )


def _build_report(
    *,
    checkpoint: Path,
    info: dict[str, Any],
    split_mode: str,
    eval_seed: int,
    speakers_total: int,
    utterances_total: int,
    speakers_eval: int,
    utterances_eval: int,
    num_samples: int,
    num_present: int,
    num_absent: int,
    mixer_cfg: MixerConfig,
    native_stats: dict[str, float | int],
    normalized_stats: dict[str, float | int],
    mask_overall: dict[str, float | int | None],
    mask_active: dict[str, float | int | None],
    mask_silent: dict[str, float | int | None],
    energy_ratio: dict[str, float | int | None],
    severe_lt_03_pct: float | None,
    severe_lt_01_pct: float | None,
    sdr_stats: dict[str, float | int | None],
    sdr_lt_5_pct: float | None,
    sdr_lt_0_pct: float | None,
    presence_logit_stats: dict[str, float | int | None] | None,
    presence_prob_stats: dict[str, float | int | None] | None,
    present_below_half_pct: float | None,
) -> str:
    metrics = info.get("metrics") or {}
    checkpoint_cfg = info.get("config") or {}
    lines = [
        "WULFENITE_CHECKPOINT_DIAGNOSTIC",
        "",
        "[checkpoint]",
        f"path={checkpoint}",
        f"epoch={info.get('epoch', '?')} step={info.get('step', '?')}",
        f"encoder_type={checkpoint_cfg.get('encoder_type', 'campplus')}",
        (
            "saved_metrics="
            f"val_loss={metrics.get('val_loss', 'n/a')} "
            f"val_sdr_db={metrics.get('val_sdr_db', 'n/a')} "
            f"val_sdri_db={metrics.get('val_sdri_db', 'n/a')} "
            f"same_cosine={metrics.get('same_cosine', 'n/a')} "
            f"diff_cosine={metrics.get('diff_cosine', 'n/a')} "
            f"cosine_gap={metrics.get('cosine_gap', 'n/a')}"
        ),
        "",
        "[dataset]",
        f"speaker_pool={speakers_total} speakers / {utterances_total} utterances",
        f"eval_pool={speakers_eval} speakers / {utterances_eval} utterances",
        f"split_mode={split_mode} eval_seed={eval_seed}",
        f"samples={num_samples} present={num_present} absent={num_absent}",
        (
            "mixer="
            f"segment_seconds={mixer_cfg.segment_seconds} "
            f"enrollment_seconds_range={mixer_cfg.enrollment_seconds_range} "
            f"snr_range_db={mixer_cfg.snr_range_db} "
            f"target_present_prob={mixer_cfg.target_present_prob} "
            f"apply_reverb={mixer_cfg.apply_reverb} "
            f"apply_noise={mixer_cfg.apply_noise}"
        ),
        "",
        "[embeddings]",
        (
            "native="
            f"dim={native_stats['dim']} "
            f"same_cos={_fmt(native_stats['same_cos'])} "
            f"diff_cos={_fmt(native_stats['diff_cos'])} "
            f"cos_gap={_fmt(native_stats['cos_gap'])}"
        ),
        (
            "normalized="
            f"dim={normalized_stats['dim']} "
            f"same_cos={_fmt(normalized_stats['same_cos'])} "
            f"diff_cos={_fmt(normalized_stats['diff_cos'])} "
            f"cos_gap={_fmt(normalized_stats['cos_gap'])}"
        ),
        (
            "normalization_delta="
            f"cos_gap_change={_fmt(normalized_stats['cos_gap'] - native_stats['cos_gap'])}"
        ),
        (
            "pairs="
            f"same={native_stats['same_pairs']} diff={native_stats['diff_pairs']}"
        ),
        "",
        "[mask]",
        "scope=present_samples_only",
        (
            "target_active_definition="
            f"frame_energy >= peak_frame_energy - {ACTIVE_FRAME_DROP_DB:.0f} dB "
            "(20 ms window, 10 ms hop, separator-aligned)"
        ),
        _format_mask_summary("overall", mask_overall),
        _format_mask_summary("target_active", mask_active),
        _format_mask_summary("target_silent", mask_silent),
        "",
        "[frame_energy_ratio]",
        "scope=present_sample target-active frames only",
        _format_summary("ratio", energy_ratio),
        f"ratio_lt_0.3={_fmt_pct(severe_lt_03_pct)}",
        f"ratio_lt_0.1={_fmt_pct(severe_lt_01_pct)}",
        "",
        "[sdr_distribution]",
        "scope=present_samples_only",
        _format_summary("sdr_db", sdr_stats),
        f"sdr_lt_5db={_fmt_pct(sdr_lt_5_pct)}",
        f"sdr_lt_0db={_fmt_pct(sdr_lt_0_pct)}",
        "",
        "[presence_head]",
    ]
    if presence_logit_stats is None or presence_prob_stats is None:
        lines.append("status=disabled")
    else:
        lines.append("scope=present_samples_only")
        lines.append(_format_summary("presence_logit", presence_logit_stats))
        lines.append(_format_summary("presence_prob", presence_prob_stats))
        lines.append(f"present_prob_lt_0.5={_fmt_pct(present_below_half_pct)}")
    return "\n".join(lines)


def run_diagnostic(
    checkpoint: Path,
    aishell1_root: Path,
    *,
    magicdata_root: Path | None = None,
    cnceleb_root: Path | None = None,
    num_samples: int = 200,
    device: str = "cpu",
) -> dict[str, Any]:
    """Run the full offline diagnostic and print a structured report."""
    if num_samples <= 0:
        raise ValueError(f"num_samples must be > 0, got {num_samples}")

    dev = torch.device(device)
    model, info = build_model_from_checkpoint(checkpoint, device=dev)
    checkpoint_cfg = info.get("config") or {}
    eval_seed = int(checkpoint_cfg.get("seed", DEFAULT_SEED))
    val_ratio = float(checkpoint_cfg.get("val_speaker_ratio", 0.2))

    torch.manual_seed(eval_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(eval_seed)

    args = argparse.Namespace(
        aishell1_root=aishell1_root,
        magicdata_root=magicdata_root,
        cnceleb_root=cnceleb_root,
    )
    speaker_pool = _build_speaker_pool(args)
    eval_speakers, split_mode = _split_speakers_for_eval(
        speaker_pool,
        val_ratio=val_ratio,
        seed=eval_seed,
    )
    mixer_cfg = _mixer_config_from_checkpoint(checkpoint_cfg)
    with contextlib.redirect_stdout(io.StringIO()):
        mixer = WulfeniteMixer(
            speakers=eval_speakers,
            config=mixer_cfg,
            samples_per_epoch=num_samples,
            seed=eval_seed,
        )

    mask_capture = MaskCapture()
    hook_handle = model.separator.mask_head.register_forward_hook(mask_capture)

    cfg = model.separator.config
    frame_size = cfg.enc_kernel_size
    hop_size = cfg.enc_stride
    pad_left = cfg.enc_kernel_size - cfg.enc_stride

    native_embeddings: list[torch.Tensor] = []
    normalized_embeddings: list[torch.Tensor] = []
    labels: list[int] = []
    energy_ratios: list[torch.Tensor] = []
    sdr_values: list[float] = []
    presence_logits: list[float] = []
    presence_probs: list[float] = []
    mask_overall = MaskValueStats()
    mask_active = MaskValueStats()
    mask_silent = MaskValueStats()
    num_present = 0
    num_absent = 0

    try:
        with torch.inference_mode():
            iterator = tqdm(
                range(num_samples),
                desc="diagnose",
                total=num_samples,
                dynamic_ncols=True,
            )
            for index in iterator:
                sample = mixer[index]
                mixture = sample["mixture"].unsqueeze(0).to(dev)
                target = sample["target"].unsqueeze(0).to(dev)
                enrollment = sample["enrollment"].unsqueeze(0).to(dev)
                enrollment_fbank = compute_fbank_batch(enrollment)

                outputs = model(
                    mixture,
                    enrollment,
                    enrollment_fbank,
                )
                mask = mask_capture.pop()

                native_embeddings.append(outputs["raw_embedding"][0].detach().cpu())
                normalized_embeddings.append(outputs["embedding"][0].detach().cpu())
                labels.append(int(sample["target_speaker_idx"].item()))

                is_present = bool(sample["target_present"].item() >= 0.5)
                if not is_present:
                    num_absent += 1
                    continue

                num_present += 1
                clean = outputs["clean"]
                sdr = compute_sdr_db(clean, target, reduction="none")
                sdr_values.append(float(sdr[0].item()))

                target_frame_energy = _frame_energy(
                    target[0],
                    frame_size=frame_size,
                    hop_size=hop_size,
                    pad_left=pad_left,
                )
                output_frame_energy = _frame_energy(
                    clean[0],
                    frame_size=frame_size,
                    hop_size=hop_size,
                    pad_left=pad_left,
                )
                active = _target_active_mask(target_frame_energy)

                mask = mask[0]
                mask_overall.update(mask)
                if active.any():
                    mask_active.update(mask[:, active])
                    ratios = output_frame_energy[active] / (target_frame_energy[active] + EPS)
                    energy_ratios.append(ratios.detach().cpu())
                if (~active).any():
                    mask_silent.update(mask[:, ~active])

                presence_logit = outputs.get("presence_logit")
                if presence_logit is not None:
                    logit = float(presence_logit[0].item())
                    presence_logits.append(logit)
                    presence_probs.append(float(torch.sigmoid(presence_logit[0]).item()))
    finally:
        hook_handle.remove()

    native_tensor = (
        torch.stack(native_embeddings, dim=0) if native_embeddings else torch.empty(0, 0)
    )
    normalized_tensor = (
        torch.stack(normalized_embeddings, dim=0)
        if normalized_embeddings else torch.empty(0, 0)
    )
    label_tensor = torch.tensor(labels, dtype=torch.long)

    native_stats = _cosine_gap(native_tensor, label_tensor)
    normalized_stats = _cosine_gap(normalized_tensor, label_tensor)

    energy_ratio_tensor = (
        torch.cat(energy_ratios, dim=0) if energy_ratios else torch.empty(0)
    )
    energy_ratio_stats = _summarize_tensor(energy_ratio_tensor)
    severe_lt_03_pct = (
        float((energy_ratio_tensor < 0.3).to(torch.float64).mean().item() * 100.0)
        if energy_ratio_tensor.numel()
        else None
    )
    severe_lt_01_pct = (
        float((energy_ratio_tensor < 0.1).to(torch.float64).mean().item() * 100.0)
        if energy_ratio_tensor.numel()
        else None
    )

    sdr_tensor = torch.tensor(sdr_values, dtype=torch.float64)
    sdr_stats = _summarize_tensor(sdr_tensor)
    sdr_lt_5_pct = (
        float((sdr_tensor < 5.0).to(torch.float64).mean().item() * 100.0)
        if sdr_tensor.numel()
        else None
    )
    sdr_lt_0_pct = (
        float((sdr_tensor < 0.0).to(torch.float64).mean().item() * 100.0)
        if sdr_tensor.numel()
        else None
    )

    presence_logit_stats: dict[str, float | int | None] | None
    presence_prob_stats: dict[str, float | int | None] | None
    present_below_half_pct: float | None
    if presence_logits:
        presence_logit_tensor = torch.tensor(presence_logits, dtype=torch.float64)
        presence_prob_tensor = torch.tensor(presence_probs, dtype=torch.float64)
        presence_logit_stats = _summarize_tensor(presence_logit_tensor)
        presence_prob_stats = _summarize_tensor(presence_prob_tensor)
        present_below_half_pct = float(
            (presence_prob_tensor < 0.5).to(torch.float64).mean().item() * 100.0
        )
    else:
        presence_logit_stats = None
        presence_prob_stats = None
        present_below_half_pct = None

    report = _build_report(
        checkpoint=checkpoint,
        info=info,
        split_mode=split_mode,
        eval_seed=eval_seed,
        speakers_total=len(speaker_pool),
        utterances_total=_count_utterances(speaker_pool),
        speakers_eval=len(eval_speakers),
        utterances_eval=_count_utterances(eval_speakers),
        num_samples=num_samples,
        num_present=num_present,
        num_absent=num_absent,
        mixer_cfg=mixer_cfg,
        native_stats=native_stats,
        normalized_stats=normalized_stats,
        mask_overall=mask_overall.summary(),
        mask_active=mask_active.summary(),
        mask_silent=mask_silent.summary(),
        energy_ratio=energy_ratio_stats,
        severe_lt_03_pct=severe_lt_03_pct,
        severe_lt_01_pct=severe_lt_01_pct,
        sdr_stats=sdr_stats,
        sdr_lt_5_pct=sdr_lt_5_pct,
        sdr_lt_0_pct=sdr_lt_0_pct,
        presence_logit_stats=presence_logit_stats,
        presence_prob_stats=presence_prob_stats,
        present_below_half_pct=present_below_half_pct,
    )
    print(report)
    return {
        "report": report,
        "num_present": num_present,
        "num_absent": num_absent,
        "native": native_stats,
        "normalized": normalized_stats,
        "mask": {
            "overall": mask_overall.summary(),
            "target_active": mask_active.summary(),
            "target_silent": mask_silent.summary(),
        },
        "frame_energy_ratio": {
            **energy_ratio_stats,
            "lt_0.3_pct": severe_lt_03_pct,
            "lt_0.1_pct": severe_lt_01_pct,
        },
        "sdr_distribution": {
            **sdr_stats,
            "lt_5db_pct": sdr_lt_5_pct,
            "lt_0db_pct": sdr_lt_0_pct,
        },
        "presence_head": {
            "logit": presence_logit_stats,
            "prob": presence_prob_stats,
            "present_prob_lt_0.5_pct": present_below_half_pct,
        },
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Trained Wulfenite checkpoint (.pt).",
    )
    parser.add_argument(
        "--aishell1-root",
        type=Path,
        required=True,
        help="AISHELL-1 root used to build the evaluation speaker pool.",
    )
    parser.add_argument("--magicdata-root", type=Path, default=None)
    parser.add_argument("--cnceleb-root", type=Path, default=None)
    parser.add_argument(
        "--num-samples",
        type=int,
        default=200,
        help="Number of deterministic mixer samples to evaluate.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_diagnostic(
        checkpoint=args.checkpoint,
        aishell1_root=args.aishell1_root,
        magicdata_root=args.magicdata_root,
        cnceleb_root=args.cnceleb_root,
        num_samples=args.num_samples,
        device=args.device,
    )


if __name__ == "__main__":
    main()
