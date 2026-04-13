"""Sample long-scene mixer outputs and dump them as wav files.

This is a listening/debugging utility for the unified long-scene
composer. It writes each sampled scene as:

- ``mixture.wav``
- per-role clean tracks (``role_A.wav`` / ``role_B.wav`` / ``role_C.wav``)
- representative enrollment clips
- ``preview.wav`` containing mixture + stems concatenated with silences
- ``metadata.json`` describing the slot structure and sampled speakers

Usage:

    uv run --directory python python -m wulfenite.scripts.sample_scenes \
        --aishell1-root ../assets/aishell1 \
        --aishell3-root ../assets/aishell3 \
        --magicdata-root ../assets/magicdata \
        --noise-root ../assets/musan/noise \
        --out-dir ../assets/scene_samples/flight_check \
        --max-scenes 6
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import soundfile as sf
import torch

from ..data import (
    ComposerConfig,
    MixerConfig,
    WulfeniteMixer,
    merge_speaker_dicts,
    scan_aishell1,
    scan_aishell3,
    scan_cnceleb,
    scan_magicdata,
    scan_noise_dir,
)


SAMPLE_RATE = 16000


def _save_wav(path: Path, waveform: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), waveform.detach().cpu().numpy(), SAMPLE_RATE)


def _build_speaker_pool(args: argparse.Namespace) -> dict[str, list[Any]]:
    speakers: dict[str, list[Any]] = {}
    if args.aishell1_root is not None:
        speakers = scan_aishell1(args.aishell1_root, splits=("train", "dev", "test"))
    if args.aishell3_root is not None:
        speakers = merge_speaker_dicts(
            speakers,
            scan_aishell3(args.aishell3_root, splits=("train", "test")),
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
            "No speakers found. Provide at least one valid speech corpus root."
        )
    return speakers


def _build_mixer(args: argparse.Namespace) -> WulfeniteMixer:
    speakers = _build_speaker_pool(args)
    noise_pool = None
    if args.noise_root is not None:
        noise_pool = scan_noise_dir(args.noise_root)

    composer_cfg = ComposerConfig(
        sample_rate=SAMPLE_RATE,
        segment_seconds=args.segment_seconds,
        target_only_min_frames=max(
            1, int(round(args.scene_target_only_min_seconds * 100)),
        ),
        nontarget_only_min_frames=max(
            1, int(round(args.scene_nontarget_only_min_seconds * 100)),
        ),
        overlap_min_frames=max(1, int(round(args.scene_overlap_min_seconds * 100))),
        background_min_frames=max(
            1, int(round(args.scene_background_min_seconds * 100)),
        ),
        absence_before_return_min_frames=max(
            1, int(round(args.scene_absence_before_return_min_seconds * 100)),
        ),
        crossfade_samples=max(0, int(round(args.crossfade_ms * 16))),
        optional_third_speaker_prob=args.optional_third_speaker_prob,
        gain_drift_db_range=tuple(args.gain_drift_db_range),
        snr_range_db=tuple(args.snr_range_db),
        noise_snr_range_db=tuple(args.noise_snr_range_db),
    )
    cfg = MixerConfig(
        sample_rate=SAMPLE_RATE,
        segment_seconds=args.segment_seconds,
        enrollment_seconds=args.enrollment_seconds,
        snr_range_db=tuple(args.snr_range_db),
        composition_mode="clip_composer",
        composer=composer_cfg,
        apply_reverb=args.reverb_prob > 0.0,
        reverb_prob=args.reverb_prob,
        rir_pool_size=args.rir_pool_size,
        apply_noise=args.noise_root is not None and args.noise_prob > 0.0,
        noise_prob=args.noise_prob,
        noise_snr_range_db=tuple(args.noise_snr_range_db),
    )
    return WulfeniteMixer(
        speakers=speakers,
        noise_pool=noise_pool,
        config=cfg,
        samples_per_epoch=args.max_scenes,
        seed=args.seed,
    )


def _preview_waveform(
    mixture: torch.Tensor,
    role_tracks: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, list[str]]:
    silence = torch.zeros(int(0.25 * SAMPLE_RATE), dtype=torch.float32)
    segments = [mixture]
    labels = ["mixture"]
    for role in sorted(role_tracks.keys()):
        segments.extend([silence, role_tracks[role]])
        labels.append(f"role_{role}")
    return torch.cat(segments, dim=0), labels


def _scene_metadata(scene: dict) -> dict[str, Any]:
    plan = scene["plan"]
    bundle = scene["bundle"]
    return {
        "scene_id": int(scene["scene_id"]),
        "duration_sec": bundle.mixture.numel() / SAMPLE_RATE,
        "snr_db": float(bundle.metadata.get("snr_db", 0.0)),
        "source_speaker_ids": dict(bundle.metadata["source_speaker_ids"]),
        "outsider_speaker_id": bundle.metadata["outsider_speaker_id"],
        "active_seconds_by_role": {
            role: float(mask.sum().item()) / 100.0
            for role, mask in bundle.active_frames_by_role.items()
        },
        "overlap_seconds": float(bundle.overlap_frames.sum().item()) / 100.0,
        "background_seconds": float(bundle.background_frames.sum().item()) / 100.0,
        "slots": [
            {
                "index": slot.index,
                "event_type": slot.event_type.value,
                "anchor_name": slot.anchor_name,
                "start_sec": slot.start_frame / 100.0,
                "end_sec": slot.end_frame / 100.0,
                "active_roles": list(slot.active_roles),
            }
            for slot in plan.slots
        ],
        "views": [
            {
                "view_role": str(view["view_role"]),
                "target_present": float(view["target_present"].item()),
                "target_speaker_idx": int(view["target_speaker_idx"].item()),
            }
            for view in scene["views"]
        ],
    }


def export_scene_samples(
    mixer: WulfeniteMixer,
    out_dir: Path,
    *,
    max_scenes: int,
    write_all_enrollment_modes: bool = False,
) -> list[dict[str, Any]]:
    if max_scenes <= 0:
        raise ValueError(f"max_scenes must be positive; got {max_scenes}")
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []

    for index in range(max_scenes):
        scene = mixer.sample_scene(index)
        bundle = scene["bundle"]
        scene_dir = out_dir / f"scene_{index:03d}"
        scene_dir.mkdir(parents=True, exist_ok=True)

        _save_wav(scene_dir / "mixture.wav", bundle.mixture)
        preview, preview_order = _preview_waveform(bundle.mixture, bundle.source_tracks)
        _save_wav(scene_dir / "preview.wav", preview)

        for role, track in sorted(bundle.source_tracks.items()):
            _save_wav(scene_dir / f"role_{role}.wav", track)

        enrollment_modes = list(bundle.metadata.get("enrollment_modes", ()))
        for role, candidates in sorted(bundle.enrollment_pool.items()):
            role_name = role.lower()
            if write_all_enrollment_modes:
                for mode, candidate in zip(enrollment_modes, candidates):
                    _save_wav(
                        scene_dir / f"enrollment_{role_name}_{mode}.wav",
                        candidate,
                    )
            else:
                mode = enrollment_modes[0] if enrollment_modes else "default"
                _save_wav(
                    scene_dir / f"enrollment_{role_name}_{mode}.wav",
                    candidates[0],
                )

        metadata = _scene_metadata(scene)
        metadata["preview_order"] = preview_order
        metadata_path = scene_dir / "metadata.json"
        metadata_path.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=True),
            encoding="utf-8",
        )
        manifest.append(
            {
                "scene_id": index,
                "scene_dir": str(scene_dir),
                "source_speaker_ids": metadata["source_speaker_ids"],
                "outsider_speaker_id": metadata["outsider_speaker_id"],
            }
        )

    (out_dir / "manifest.json").write_text(
        json.dumps(
            {
                "sample_rate": SAMPLE_RATE,
                "max_scenes": max_scenes,
                "scene_dirs": manifest,
            },
            indent=2,
            ensure_ascii=True,
        ),
        encoding="utf-8",
    )
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aishell1-root", type=Path, default=None)
    parser.add_argument("--aishell3-root", type=Path, default=None)
    parser.add_argument("--magicdata-root", type=Path, default=None)
    parser.add_argument("--cnceleb-root", type=Path, default=None)
    parser.add_argument("--noise-root", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-scenes", type=int, default=6)
    parser.add_argument("--segment-seconds", type=float, default=8.0)
    parser.add_argument("--enrollment-seconds", type=float, default=4.0)
    parser.add_argument("--snr-range-db", type=float, nargs=2, default=(-5.0, 5.0))
    parser.add_argument(
        "--noise-snr-range-db", type=float, nargs=2, default=(10.0, 25.0),
    )
    parser.add_argument("--noise-prob", type=float, default=0.80)
    parser.add_argument("--reverb-prob", type=float, default=0.85)
    parser.add_argument("--rir-pool-size", type=int, default=64)
    parser.add_argument("--crossfade-ms", type=float, default=5.0)
    parser.add_argument("--optional-third-speaker-prob", type=float, default=0.35)
    parser.add_argument(
        "--gain-drift-db-range", type=float, nargs=2, default=(-1.5, 1.5),
    )
    parser.add_argument("--scene-target-only-min-seconds", type=float, default=0.8)
    parser.add_argument("--scene-nontarget-only-min-seconds", type=float, default=0.8)
    parser.add_argument("--scene-overlap-min-seconds", type=float, default=0.4)
    parser.add_argument("--scene-background-min-seconds", type=float, default=0.3)
    parser.add_argument(
        "--scene-absence-before-return-min-seconds",
        type=float,
        default=1.0,
    )
    parser.add_argument("--write-all-enrollment-modes", action="store_true")
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    mixer = _build_mixer(args)
    manifest = export_scene_samples(
        mixer,
        args.out_dir,
        max_scenes=args.max_scenes,
        write_all_enrollment_modes=args.write_all_enrollment_modes,
    )
    print(f"[done] wrote {len(manifest)} scene(s) to {args.out_dir}")


if __name__ == "__main__":
    main()
