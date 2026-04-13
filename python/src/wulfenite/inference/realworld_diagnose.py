"""Real-world diagnosis CLI for one mixture with true vs wrong enrollments.

This script focuses on the failure mode where the separator keeps a high
mask on non-target speech in real conversational audio. It runs several
diagnostics from one command:

1. True-vs-wrong enrollment comparison on the same mixture.
2. Chunk-level streaming mask / RMS traces for each enrollment.
3. Optional region-level summaries from a user-provided CSV annotation.
4. Optional isolated-cut checks to separate local vs context failure.
5. Speaker-modulation statistics for each enrollment.
6. Optional synthetic hard-negative probe using local training corpora.

Typical usage:

    uv run --directory python python -m wulfenite.inference.realworld_diagnose \
        --checkpoint ../assets/best.pt \
        --mixture ../assets/samples/real_mixture.wav \
        --target-enrollment ../assets/samples/real_enrollment.wav \
        --wrong-enrollment ../assets/samples/wrong_enrollment.wav \
        --regions ../assets/samples/real_regions.csv \
        --out-dir ../assets/diagnostics/real_case
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
import platform
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import soundfile as sf
import torch

from ..audio_features import compute_fbank_batch
from ..data import (
    MixerConfig,
    WulfeniteMixer,
    merge_speaker_dicts,
    scan_aishell1,
    scan_cnceleb,
    scan_magicdata,
)
from ..data.composer import ComposerConfig
from .utils import build_model_from_checkpoint


if platform.system() == "Windows":
    pathlib.PosixPath = pathlib.WindowsPath  # type: ignore[misc, assignment]


SAMPLE_RATE = 16000


@dataclass(frozen=True)
class RegionSpec:
    name: str
    label: str
    start_sec: float
    end_sec: float


def _load_mono(path: Path) -> torch.Tensor:
    data, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if sr != SAMPLE_RATE:
        raise ValueError(
            f"{path} has sample_rate={sr}, expected {SAMPLE_RATE}. "
            "Resample to 16 kHz mono first."
        )
    if data.ndim == 2:
        data = data.mean(axis=1)
    return torch.from_numpy(data)


def _rms(x: torch.Tensor) -> float:
    return float((x.square().mean() + 1e-12).sqrt().item())


def _energy_ratio(estimate: torch.Tensor, mixture: torch.Tensor) -> float:
    est_e = float(estimate.square().sum().item())
    mix_e = float(mixture.square().sum().item())
    return est_e / (mix_e + 1e-12)


def _summarize(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {
            "mean": None,
            "median": None,
            "min": None,
            "max": None,
        }
    return {
        "mean": float(statistics.fmean(values)),
        "median": float(statistics.median(values)),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def _parse_regions(path: Path) -> list[RegionSpec]:
    regions: list[RegionSpec] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"start_sec", "end_sec", "label"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(
                f"{path} must be CSV with headers start_sec,end_sec,label "
                "and optional name"
            )
        for i, row in enumerate(reader):
            start_sec = float(row["start_sec"])
            end_sec = float(row["end_sec"])
            if end_sec <= start_sec:
                raise ValueError(
                    f"region row {i + 2} has end_sec <= start_sec: "
                    f"{start_sec} -> {end_sec}"
                )
            name = row.get("name") or f"region_{i:03d}"
            label = row["label"].strip()
            if not label:
                raise ValueError(f"region row {i + 2} has empty label")
            regions.append(
                RegionSpec(
                    name=name.strip(),
                    label=label,
                    start_sec=start_sec,
                    end_sec=end_sec,
                )
            )
    return regions


def _whole_presence_prob(
    model,
    mixture: torch.Tensor,
    enrollment: torch.Tensor,
) -> float | None:
    with torch.no_grad():
        outputs = model(mixture.unsqueeze(0), enrollment.unsqueeze(0))
    if "presence_logit" not in outputs:
        return None
    return float(torch.sigmoid(outputs["presence_logit"][0]).item())


def _speaker_modulation_stats(
    model,
    enrollment: torch.Tensor,
) -> dict[str, Any]:
    with torch.no_grad():
        raw_emb, norm_emb = model.speaker_encoder(enrollment.unsqueeze(0))
        speaker_scale = model.separator.speaker_projection(norm_emb)
    return {
        "raw_embedding": raw_emb[0].cpu(),
        "norm_embedding": norm_emb[0].cpu(),
        "speaker_scale": speaker_scale[0].cpu(),
        "summary": {
            "embedding_dim": int(norm_emb.shape[-1]),
            "speaker_scale_l2": float(speaker_scale.norm().item()),
            "speaker_scale_abs_mean": float(speaker_scale.abs().mean().item()),
            "speaker_scale_mean": float(speaker_scale.mean().item()),
            "speaker_scale_std": float(speaker_scale.std().item()),
        },
    }


def _stream_one(
    separator,
    mixture: torch.Tensor,
    speaker_embedding: torch.Tensor,
    *,
    chunk_samples: int,
    s4d_decay: float,
) -> tuple[torch.Tensor, list[dict[str, float]]]:
    if chunk_samples <= 0:
        raise ValueError("chunk_samples must be positive")
    pad_len = (chunk_samples - mixture.shape[0] % chunk_samples) % chunk_samples
    padded = mixture
    if pad_len:
        padded = torch.nn.functional.pad(mixture, (0, pad_len))
    padded = padded.unsqueeze(0)

    state = separator.initial_streaming_state(
        batch_size=1,
        device=padded.device,
    )
    clean_pieces: list[torch.Tensor] = []
    rows: list[dict[str, float]] = []
    n_chunks = padded.shape[-1] // chunk_samples
    for i in range(n_chunks):
        start = i * chunk_samples
        end = start + chunk_samples
        mix_chunk = padded[..., start:end]
        clean_chunk, state = separator.streaming_step(
            mix_chunk,
            speaker_embedding,
            state,
            s4d_state_decay=s4d_decay,
        )
        clean_chunk = clean_chunk[0].detach().cpu()
        mix_chunk_1d = mix_chunk[0].detach().cpu()
        clean_pieces.append(clean_chunk)
        mix_rms = _rms(mix_chunk_1d)
        out_rms = _rms(clean_chunk)
        rows.append(
            {
                "chunk_idx": float(i),
                "start_sec": start / SAMPLE_RATE,
                "end_sec": end / SAMPLE_RATE,
                "mix_rms": mix_rms,
                "out_rms": out_rms,
                "out_to_mix_rms": out_rms / max(mix_rms, 1e-12),
                "out_to_mix_energy": _energy_ratio(clean_chunk, mix_chunk_1d),
                "mask_mean": float(state["mask_mean"]),
                "mask_max": float(state["mask_max"]),
                "mask_min": float(state["mask_min"]),
            }
        )
    clean = torch.cat(clean_pieces, dim=-1)
    if pad_len:
        clean = clean[:-pad_len]
    return clean, rows


def _pairwise_stats(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, float]:
    emb_a = baseline["speaker_modulation"]["norm_embedding"]
    emb_b = candidate["speaker_modulation"]["norm_embedding"]
    scale_a = baseline["speaker_modulation"]["speaker_scale"]
    scale_b = candidate["speaker_modulation"]["speaker_scale"]
    out_a = baseline["stream_clean"]
    out_b = candidate["stream_clean"]
    cosine = float(torch.dot(emb_a, emb_b).item())
    return {
        "embedding_cosine": cosine,
        "embedding_l2": float((emb_a - emb_b).norm().item()),
        "speaker_scale_diff_l2": float((scale_a - scale_b).norm().item()),
        "speaker_scale_diff_abs_mean": float(
            (scale_a - scale_b).abs().mean().item()
        ),
        "stream_output_l1_mean": float((out_a - out_b).abs().mean().item()),
        "stream_output_l2": float((out_a - out_b).norm().item()),
    }


def _overlapping_chunk_rows(
    rows: list[dict[str, float]],
    region: RegionSpec,
) -> list[dict[str, float]]:
    return [
        row
        for row in rows
        if row["end_sec"] > region.start_sec and row["start_sec"] < region.end_sec
    ]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with path.open("w", encoding="utf-8", newline="") as f:
            f.write("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _save_wav(path: Path, waveform: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), waveform.cpu().numpy(), SAMPLE_RATE)


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
            "No speakers found. Provide at least --aishell1-root with valid speech data."
        )
    return speakers


def _mixer_config_for_hard_negatives(checkpoint_cfg: dict[str, Any]) -> MixerConfig:
    if "enrollment_seconds_range" in checkpoint_cfg:
        seconds_range = tuple(checkpoint_cfg["enrollment_seconds_range"])
        enrollment_seconds = float(max(seconds_range))
    else:
        enrollment_seconds = float(checkpoint_cfg.get("enrollment_seconds", 4.0))
    composer_cfg = ComposerConfig(
        sample_rate=SAMPLE_RATE,
        segment_seconds=float(checkpoint_cfg.get("segment_seconds", 4.0)),
        target_only_min_frames=max(
            1,
            int(
                round(
                    float(
                        checkpoint_cfg.get("scene_target_only_min_seconds", 0.8)
                    ) * 100
                )
            ),
        ),
        nontarget_only_min_frames=max(
            1,
            int(
                round(
                    float(
                        checkpoint_cfg.get("scene_nontarget_only_min_seconds", 0.8)
                    ) * 100
                )
            ),
        ),
        overlap_min_frames=max(
            1,
            int(
                round(
                    float(
                        checkpoint_cfg.get("scene_overlap_min_seconds", 0.4)
                    ) * 100
                )
            ),
        ),
        background_min_frames=max(
            1,
            int(
                round(
                    float(
                        checkpoint_cfg.get("scene_background_min_seconds", 0.3)
                    ) * 100
                )
            ),
        ),
        absence_before_return_min_frames=max(
            1,
            int(
                round(
                    float(
                        checkpoint_cfg.get(
                            "scene_absence_before_return_min_seconds",
                            1.0,
                        )
                    ) * 100
                )
            ),
        ),
        crossfade_samples=max(
            0,
            int(round(float(checkpoint_cfg.get("crossfade_ms", 5.0)) * 16)),
        ),
        optional_third_speaker_prob=float(
            checkpoint_cfg.get("optional_third_speaker_prob", 0.35)
        ),
        gain_drift_db_range=tuple(
            checkpoint_cfg.get("gain_drift_db_range", (-1.5, 1.5))
        ),
        snr_range_db=tuple(checkpoint_cfg.get("snr_range_db", (-5.0, 5.0))),
        noise_snr_range_db=tuple(
            checkpoint_cfg.get("noise_snr_range_db", (10.0, 25.0))
        ),
    )
    reverb_prob = float(checkpoint_cfg.get("reverb_prob", 0.85))
    return MixerConfig(
        segment_seconds=float(checkpoint_cfg.get("segment_seconds", 4.0)),
        enrollment_seconds=enrollment_seconds,
        snr_range_db=tuple(checkpoint_cfg.get("snr_range_db", (-5.0, 5.0))),
        composition_mode="clip_composer",
        composer=composer_cfg,
        apply_reverb=reverb_prob > 0.0,
        reverb_prob=reverb_prob,
        rir_pool_size=int(checkpoint_cfg.get("rir_pool_size", 1000)),
        apply_noise=False,
        noise_prob=0.0,
    )


def _synthetic_hard_negative_probe(
    model,
    checkpoint_cfg: dict[str, Any],
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    speakers = _build_speaker_pool(args)
    mixer_cfg = _mixer_config_for_hard_negatives(checkpoint_cfg)
    mixer = WulfeniteMixer(
        speakers=speakers,
        config=mixer_cfg,
        samples_per_epoch=args.synthetic_hard_negative_samples,
        seed=args.seed,
    )
    output_ratios: list[float] = []
    presence_probs: list[float] = []
    with torch.inference_mode():
        for index in range(args.synthetic_hard_negative_samples):
            sample = mixer[index]
            if bool(sample["target_present"].item() >= 0.5):
                continue
            mixture = sample["mixture"].unsqueeze(0).to(device)
            enrollment = sample["enrollment"].unsqueeze(0).to(device)
            enrollment_fbank = compute_fbank_batch(enrollment)
            outputs = model(mixture, enrollment, enrollment_fbank)
            clean = outputs["clean"][0].detach().cpu()
            mix = sample["mixture"]
            output_ratios.append(_energy_ratio(clean, mix))
            if "presence_logit" in outputs:
                presence_probs.append(
                    float(torch.sigmoid(outputs["presence_logit"][0]).item())
                )
    return {
        "num_samples": len(output_ratios),
        "output_to_mix_energy": _summarize(output_ratios),
        "presence_prob": _summarize(presence_probs),
    }


def run_realworld_diagnosis(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device)
    model, info = build_model_from_checkpoint(args.checkpoint, device=device)

    chunk_samples = int(round(args.chunk_ms * SAMPLE_RATE / 1000.0))
    if chunk_samples <= 0 or chunk_samples % model.separator.config.enc_stride != 0:
        raise ValueError(
            f"--chunk-ms must map to a positive multiple of "
            f"{model.separator.config.enc_stride * 1000 / SAMPLE_RATE:.0f} ms"
        )

    mixture = _load_mono(args.mixture)
    regions = _parse_regions(args.regions) if args.regions is not None else []

    enrollments: list[tuple[str, Path]] = [("target", args.target_enrollment)]
    for i, path in enumerate(args.wrong_enrollment):
        enrollments.append((f"wrong_{i}", path))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    chunk_rows: list[dict[str, Any]] = []
    region_rows: list[dict[str, Any]] = []
    results_by_tag: dict[str, dict[str, Any]] = {}

    mix_seconds = mixture.shape[0] / SAMPLE_RATE
    print(f"[load] checkpoint {args.checkpoint} epoch={info.get('epoch')}")
    print(f"[mixture] {args.mixture} seconds={mix_seconds:.2f}")

    for tag, enrollment_path in enrollments:
        enrollment = _load_mono(enrollment_path)
        speaker_modulation = _speaker_modulation_stats(model, enrollment.to(device))
        presence_prob = _whole_presence_prob(
            model,
            mixture.to(device),
            enrollment.to(device),
        )
        with torch.no_grad():
            speaker_embedding = (
                speaker_modulation["norm_embedding"].unsqueeze(0).to(device)
            )
            stream_clean, stream_rows = _stream_one(
                model.separator,
                mixture.to(device),
                speaker_embedding,
                chunk_samples=chunk_samples,
                s4d_decay=args.s4d_decay,
            )
        output_path = args.out_dir / f"stream_{tag}.wav"
        _save_wav(output_path, stream_clean)

        for row in stream_rows:
            chunk_rows.append(
                {
                    "enrollment_tag": tag,
                    "enrollment_path": str(enrollment_path),
                    **row,
                }
            )

        results_by_tag[tag] = {
            "tag": tag,
            "path": str(enrollment_path),
            "presence_prob": presence_prob,
            "speaker_modulation": speaker_modulation,
            "stream_clean": stream_clean,
            "stream_rows": stream_rows,
            "stream_output": str(output_path),
            "stream_rms": _rms(stream_clean),
            "stream_peak_dbfs": 20.0
            * math.log10(max(float(stream_clean.abs().max().item()), 1e-9)),
        }
        print(
            f"[enrollment] {tag} presence_prob="
            f"{'n/a' if presence_prob is None else f'{presence_prob:.4f}'} "
            f"stream_rms={results_by_tag[tag]['stream_rms']:.6f}"
        )

    target_result = results_by_tag["target"]
    comparisons: list[dict[str, Any]] = []
    for tag, result in results_by_tag.items():
        if tag == "target":
            continue
        comparisons.append(
            {
                "baseline": "target",
                "candidate": tag,
                **_pairwise_stats(target_result, result),
            }
        )

    if regions:
        print(f"[regions] loaded {len(regions)} annotated regions from {args.regions}")
        for region in regions:
            start = max(0, int(round(region.start_sec * SAMPLE_RATE)))
            end = min(mixture.shape[0], int(round(region.end_sec * SAMPLE_RATE)))
            mix_cut = mixture[start:end]
            if mix_cut.numel() == 0:
                continue
            if args.save_region_wavs:
                _save_wav(args.out_dir / f"{region.name}_mixture.wav", mix_cut)
            for tag, enrollment_path in enrollments:
                result = results_by_tag[tag]
                full_cut = result["stream_clean"][start:end]
                full_rows = _overlapping_chunk_rows(result["stream_rows"], region)
                full_mask_mean = _summarize([row["mask_mean"] for row in full_rows])
                full_out_ratio = _energy_ratio(full_cut, mix_cut)

                enrollment = _load_mono(enrollment_path)
                with torch.no_grad():
                    speaker_embedding = (
                        result["speaker_modulation"]["norm_embedding"]
                        .unsqueeze(0)
                        .to(device)
                    )
                    iso_clean, iso_rows = _stream_one(
                        model.separator,
                        mix_cut.to(device),
                        speaker_embedding,
                        chunk_samples=chunk_samples,
                        s4d_decay=args.s4d_decay,
                    )
                iso_out_ratio = _energy_ratio(iso_clean, mix_cut)
                iso_mask_mean = _summarize([row["mask_mean"] for row in iso_rows])

                if args.save_region_wavs:
                    _save_wav(
                        args.out_dir / f"{region.name}_full_{tag}.wav",
                        full_cut,
                    )
                    _save_wav(
                        args.out_dir / f"{region.name}_isolated_{tag}.wav",
                        iso_clean,
                    )

                region_rows.append(
                    {
                        "name": region.name,
                        "label": region.label,
                        "start_sec": region.start_sec,
                        "end_sec": region.end_sec,
                        "enrollment_tag": tag,
                        "enrollment_path": str(enrollment_path),
                        "full_mix_rms": _rms(mix_cut),
                        "full_out_rms": _rms(full_cut),
                        "full_out_to_mix_energy": full_out_ratio,
                        "full_mask_mean": full_mask_mean["mean"],
                        "full_mask_min": full_mask_mean["min"],
                        "full_mask_max": full_mask_mean["max"],
                        "isolated_out_rms": _rms(iso_clean),
                        "isolated_out_to_mix_energy": iso_out_ratio,
                        "isolated_mask_mean": iso_mask_mean["mean"],
                        "isolated_mask_min": iso_mask_mean["min"],
                        "isolated_mask_max": iso_mask_mean["max"],
                        "context_minus_isolated_energy": full_out_ratio - iso_out_ratio,
                    }
                )

    synthetic_probe = None
    checkpoint_cfg = info.get("config") if isinstance(info.get("config"), dict) else {}
    if args.synthetic_hard_negative_samples > 0:
        if args.aishell1_root is None:
            raise ValueError(
                "--synthetic-hard-negative-samples requires --aishell1-root "
                "(optionally plus --magicdata-root / --cnceleb-root)"
            )
        print(
            f"[synthetic] running {args.synthetic_hard_negative_samples} "
            "hard-negative samples"
        )
        synthetic_probe = _synthetic_hard_negative_probe(
            model,
            checkpoint_cfg,
            args,
            device,
        )

    region_summary: dict[str, dict[str, Any]] = {}
    for row in region_rows:
        key = f"{row['enrollment_tag']}::{row['label']}"
        bucket = region_summary.setdefault(
            key,
            {
                "enrollment_tag": row["enrollment_tag"],
                "label": row["label"],
                "full_out_to_mix_energy": [],
                "isolated_out_to_mix_energy": [],
                "context_minus_isolated_energy": [],
            },
        )
        bucket["full_out_to_mix_energy"].append(row["full_out_to_mix_energy"])
        bucket["isolated_out_to_mix_energy"].append(row["isolated_out_to_mix_energy"])
        bucket["context_minus_isolated_energy"].append(
            row["context_minus_isolated_energy"]
        )
    region_summary_rows = [
        {
            "enrollment_tag": value["enrollment_tag"],
            "label": value["label"],
            "n_regions": len(value["full_out_to_mix_energy"]),
            "full_out_to_mix_energy": _summarize(value["full_out_to_mix_energy"]),
            "isolated_out_to_mix_energy": _summarize(
                value["isolated_out_to_mix_energy"]
            ),
            "context_minus_isolated_energy": _summarize(
                value["context_minus_isolated_energy"]
            ),
        }
        for value in region_summary.values()
    ]

    serializable_enrollments = []
    for tag, result in results_by_tag.items():
        serializable_enrollments.append(
            {
                "tag": tag,
                "path": result["path"],
                "presence_prob": result["presence_prob"],
                "stream_output": result["stream_output"],
                "stream_rms": result["stream_rms"],
                "stream_peak_dbfs": result["stream_peak_dbfs"],
                "speaker_modulation": result["speaker_modulation"]["summary"],
                "chunk_metrics": {
                    "mask_mean": _summarize(
                        [row["mask_mean"] for row in result["stream_rows"]]
                    ),
                    "out_to_mix_energy": _summarize(
                        [row["out_to_mix_energy"] for row in result["stream_rows"]]
                    ),
                    "out_to_mix_rms": _summarize(
                        [row["out_to_mix_rms"] for row in result["stream_rows"]]
                    ),
                },
            }
        )

    summary = {
        "checkpoint": str(args.checkpoint),
        "epoch": info.get("epoch"),
        "mixture": str(args.mixture),
        "mixture_seconds": mix_seconds,
        "chunk_ms": args.chunk_ms,
        "s4d_decay": args.s4d_decay,
        "enrollments": serializable_enrollments,
        "comparisons": comparisons,
        "regions": [asdict(region) for region in regions],
        "region_summary": region_summary_rows,
        "synthetic_hard_negative_probe": synthetic_probe,
    }

    _write_csv(args.out_dir / "chunk_metrics.csv", chunk_rows)
    if region_rows:
        _write_csv(args.out_dir / "region_metrics.csv", region_rows)
    with (args.out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    print(f"[write] {args.out_dir / 'summary.json'}")
    print(f"[write] {args.out_dir / 'chunk_metrics.csv'}")
    if region_rows:
        print(f"[write] {args.out_dir / 'region_metrics.csv'}")
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--mixture", type=Path, required=True)
    parser.add_argument("--target-enrollment", type=Path, required=True)
    parser.add_argument(
        "--wrong-enrollment",
        type=Path,
        nargs="*",
        default=[],
        help="One or more definitely wrong-speaker enrollments.",
    )
    parser.add_argument(
        "--regions",
        type=Path,
        default=None,
        help="Optional CSV with headers start_sec,end_sec,label[,name].",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--chunk-ms", type=float, default=20.0)
    parser.add_argument(
        "--s4d-decay",
        type=float,
        default=1.0,
        help="Per-latent-step decay factor for S4D recurrent state.",
    )
    parser.add_argument(
        "--save-region-wavs",
        action="store_true",
        help="Also write mixture / full-context / isolated WAVs for each region.",
    )
    parser.add_argument(
        "--synthetic-hard-negative-samples",
        type=int,
        default=0,
        help="Optional number of synthetic hard-negative samples to probe.",
    )
    parser.add_argument("--aishell1-root", type=Path, default=None)
    parser.add_argument("--magicdata-root", type=Path, default=None)
    parser.add_argument("--cnceleb-root", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_realworld_diagnosis(args)


if __name__ == "__main__":
    main()
