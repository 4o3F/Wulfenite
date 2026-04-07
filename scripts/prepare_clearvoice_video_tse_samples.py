#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path

import imageio_ffmpeg
import soundfile as sf


DEFAULT_SAMPLE_RATE = 16_000
DEFAULT_ENROLLMENT_SECONDS = 2.0
DEFAULT_HOP_SECONDS = 0.25
DEFAULT_SMOKE_SECONDS = 4.0
VIDEO_EXTENSIONS = {".avi", ".mp4", ".mov", ".webm"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract mono 16 kHz mixture wavs from ClearerVoice video samples and "
            "auto-select short enrollment clips for smoke testing."
        )
    )
    parser.add_argument("input_dir", type=Path, help="Directory containing input video samples.")
    parser.add_argument("output_dir", type=Path, help="Directory where prepared wavs will be written.")
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=DEFAULT_SAMPLE_RATE,
        help="Target sample rate for extracted wav files.",
    )
    parser.add_argument(
        "--enrollment-seconds",
        type=float,
        default=DEFAULT_ENROLLMENT_SECONDS,
        help="Length of the auto-selected enrollment clip.",
    )
    parser.add_argument(
        "--hop-seconds",
        type=float,
        default=DEFAULT_HOP_SECONDS,
        help="Hop size when searching for a high-energy enrollment window.",
    )
    parser.add_argument(
        "--smoke-seconds",
        type=float,
        default=DEFAULT_SMOKE_SECONDS,
        help="Length of a short mixture clip written for quick smoke tests.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    ffmpeg = Path(imageio_ffmpeg.get_ffmpeg_exe())
    videos = sorted(
        path for path in input_dir.iterdir() if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )
    if not videos:
        raise SystemExit(f"No supported videos found in {input_dir}")

    manifest: list[dict[str, object]] = []
    for video_path in videos:
        stem = video_path.stem
        sample_dir = output_dir / stem
        sample_dir.mkdir(parents=True, exist_ok=True)

        mixture_path = sample_dir / "mixture.wav"
        mixture_smoke_path = sample_dir / "mixture_smoke.wav"
        enrollment_path = sample_dir / "enrollment.wav"

        extract_audio(ffmpeg, video_path, mixture_path, args.sample_rate)
        audio, sample_rate = sf.read(mixture_path, always_2d=True, dtype="float32")
        mono = audio.mean(axis=1)

        enrollment, start_sample = select_enrollment_clip(
            mono,
            sample_rate=sample_rate,
            enrollment_seconds=args.enrollment_seconds,
            hop_seconds=args.hop_seconds,
        )
        mixture_smoke, smoke_start_sample = select_smoke_clip(
            mono,
            sample_rate=sample_rate,
            clip_start_sample=start_sample,
            clip_len_samples=len(enrollment),
            smoke_seconds=args.smoke_seconds,
        )
        sf.write(enrollment_path, enrollment, sample_rate, subtype="PCM_16")
        sf.write(mixture_smoke_path, mixture_smoke, sample_rate, subtype="PCM_16")

        manifest.append(
            {
                "sample_id": stem,
                "source_video": str(video_path),
                "mixture_wav": str(mixture_path),
                "mixture_smoke_wav": str(mixture_smoke_path),
                "enrollment_wav": str(enrollment_path),
                "sample_rate": sample_rate,
                "mixture_seconds": round(len(mono) / sample_rate, 3),
                "mixture_smoke_seconds": round(len(mixture_smoke) / sample_rate, 3),
                "mixture_smoke_start_seconds": round(smoke_start_sample / sample_rate, 3),
                "enrollment_seconds": round(len(enrollment) / sample_rate, 3),
                "enrollment_start_seconds": round(start_sample / sample_rate, 3),
                "note": (
                    "Enrollment was auto-selected from the same mixed video audio. "
                    "Use this only for smoke testing, not quality evaluation."
                ),
            }
        )

    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (output_dir / "README.txt").write_text(
        "\n".join(
            [
                "Prepared ClearerVoice video TSE smoke-test inputs.",
                "",
                "Each sample directory contains:",
                "- mixture.wav: extracted mono 16 kHz audio from the source video",
                "- mixture_smoke.wav: short clip for quick end-to-end tests",
                "- enrollment.wav: automatically selected high-energy clip from the same audio",
                "",
                "The enrollment clips are only for pipeline testing.",
                "For meaningful TSE evaluation, use a separate clean target-speaker enrollment recording.",
                "",
            ]
        )
    )


def extract_audio(ffmpeg: Path, video_path: Path, output_path: Path, sample_rate: int) -> None:
    command = [
        str(ffmpeg),
        "-y",
        "-i",
        str(video_path),
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-vn",
        "-acodec",
        "pcm_s16le",
        str(output_path),
        "-loglevel",
        "error",
    ]
    subprocess.run(command, check=True)


def select_enrollment_clip(
    samples,
    *,
    sample_rate: int,
    enrollment_seconds: float,
    hop_seconds: float,
):
    window = max(1, int(round(enrollment_seconds * sample_rate)))
    hop = max(1, int(round(hop_seconds * sample_rate)))
    if len(samples) <= window:
        return samples, 0

    best_start = 0
    best_score = -math.inf
    for start in range(0, len(samples) - window + 1, hop):
        chunk = samples[start : start + window]
        rms = math.sqrt(float((chunk * chunk).mean()))
        if rms > best_score:
            best_score = rms
            best_start = start

    return samples[best_start : best_start + window], best_start


def select_smoke_clip(
    samples,
    *,
    sample_rate: int,
    clip_start_sample: int,
    clip_len_samples: int,
    smoke_seconds: float,
):
    window = max(clip_len_samples, int(round(smoke_seconds * sample_rate)))
    if len(samples) <= window:
        return samples, 0

    center = clip_start_sample + clip_len_samples // 2
    start = max(0, center - window // 2)
    start = min(start, len(samples) - window)
    end = start + window
    return samples[start:end], start


if __name__ == "__main__":
    main()
