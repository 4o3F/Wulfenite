"""Resample AISHELL-3 from 44.1 kHz to 16 kHz mono in place.

The official AISHELL-3 distribution (openslr #93) is 44.1 kHz, but
Wulfenite training and inference run at 16 kHz. This one-off script
walks the AISHELL-3 tree, decodes each .wav, resamples to 16 kHz
mono, and rewrites the file in place. Existing 16 kHz files are
skipped so the script is idempotent.

Usage:

    cd Wulfenite
    uv run --directory python python -m wulfenite.scripts.resample_aishell3 \\
        --root ../assets/aishell3

Flags:
    --root            AISHELL-3 root directory (accepts both the
                      ``data_aishell3/`` wrapper layout and the
                      flat ``train/wav/...`` layout).
    --splits          Comma-separated list of splits to process.
                      Default: ``train,test``.
    --num-workers     Parallel workers. Default: CPU count.
    --dry-run         Print what would be done without writing.

The script uses ``torchaudio.functional.resample`` with the default
Kaiser-window filter (high quality) and writes 16-bit PCM to keep
the file sizes small and identical in format to AISHELL-1.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
from dataclasses import dataclass
from pathlib import Path

import soundfile as sf
import torch
import torchaudio.functional as AF
from tqdm.auto import tqdm


TARGET_SR = 16000


@dataclass
class ResampleJob:
    path: Path
    source_sr: int


def _scan_files(base: Path, splits: tuple[str, ...]) -> list[Path]:
    """Collect every .wav under ``base/{split}/wav/**`` for the given splits."""
    files: list[Path] = []
    for split in splits:
        split_dir = base / split / "wav"
        if not split_dir.exists():
            continue
        files.extend(sorted(split_dir.rglob("*.wav")))
    return files


def _resample_one(job_path: str) -> tuple[str, str]:
    """Worker function: resample a single file in place.

    Returns ``(path_str, status)`` where status is one of
    ``"skip"`` (already 16 kHz), ``"done"``, or ``"error:<msg>"``.
    Errors are returned rather than raised so the parallel pool
    does not abort on a single bad file.
    """
    path = Path(job_path)
    try:
        info = sf.info(str(path))
    except Exception as e:
        return (str(path), f"error:info:{e}")

    if info.samplerate == TARGET_SR and info.channels == 1:
        return (str(path), "skip")

    try:
        data, sr = sf.read(str(path), dtype="float32", always_2d=False)
    except Exception as e:
        return (str(path), f"error:read:{e}")

    # Downmix to mono if needed.
    if data.ndim == 2:
        data = data.mean(axis=1)

    wav = torch.from_numpy(data).unsqueeze(0)  # [1, T]
    if sr != TARGET_SR:
        wav = AF.resample(wav, orig_freq=sr, new_freq=TARGET_SR)

    resampled = wav.squeeze(0).numpy()

    # Write 16-bit PCM, the same format AISHELL-1 uses.
    try:
        sf.write(str(path), resampled, TARGET_SR, subtype="PCM_16")
    except Exception as e:
        return (str(path), f"error:write:{e}")

    return (str(path), "done")


def _resolve_base(root: Path) -> Path:
    """Accept both ``data_aishell3/`` wrapped and flat layouts."""
    if (root / "data_aishell3").exists():
        return root / "data_aishell3"
    return root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, required=True,
        help="Path to the AISHELL-3 root directory.",
    )
    parser.add_argument(
        "--splits", default="train,test",
        help="Comma-separated splits to process. Default: train,test",
    )
    parser.add_argument(
        "--num-workers", type=int, default=max(1, (os.cpu_count() or 1) - 1),
        help="Parallel worker processes.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="List files that would be touched but do not modify anything.",
    )
    args = parser.parse_args()

    base = _resolve_base(args.root)
    splits = tuple(s.strip() for s in args.splits.split(",") if s.strip())
    files = _scan_files(base, splits)

    if not files:
        raise SystemExit(
            f"No .wav files found under {base} for splits {splits}. "
            "Check --root points at the AISHELL-3 directory and that "
            "the archive was extracted."
        )

    # Pre-scan to count how many actually need work, so the progress
    # bar total is accurate and the user gets a sensible ETA.
    needs_work = 0
    already_ok = 0
    scan_bar = tqdm(
        files, desc="scan headers", unit="file", dynamic_ncols=True,
    )
    for f in scan_bar:
        try:
            info = sf.info(str(f))
        except Exception:
            needs_work += 1
            continue
        if info.samplerate == TARGET_SR and info.channels == 1:
            already_ok += 1
        else:
            needs_work += 1
    scan_bar.close()

    print(f"[scan] {len(files)} file(s) total")
    print(f"[scan] {already_ok} already 16 kHz mono (will skip)")
    print(f"[scan] {needs_work} to resample")

    if args.dry_run:
        print("[dry-run] stopping before any writes")
        return

    if needs_work == 0:
        print("[done] nothing to do")
        return

    n_workers = max(1, min(args.num_workers, needs_work))
    done = 0
    skipped = 0
    errors: list[tuple[str, str]] = []

    paths_as_str = [str(f) for f in files]
    with mp.Pool(n_workers) as pool, tqdm(
        total=len(paths_as_str),
        desc="resample",
        unit="file",
        dynamic_ncols=True,
        smoothing=0.1,
    ) as pbar:
        for path, status in pool.imap_unordered(
            _resample_one, paths_as_str, chunksize=16,
        ):
            if status == "done":
                done += 1
            elif status == "skip":
                skipped += 1
            else:
                errors.append((path, status))
            pbar.update(1)
            pbar.set_postfix(
                done=done, skip=skipped, err=len(errors), refresh=False,
            )

    print(f"[done] resampled={done} skipped={skipped} errors={len(errors)}")
    if errors:
        print("[errors]")
        for path, reason in errors[:20]:
            print(f"  {reason}: {path}")
        if len(errors) > 20:
            print(f"  ... and {len(errors) - 20} more")


if __name__ == "__main__":
    main()
