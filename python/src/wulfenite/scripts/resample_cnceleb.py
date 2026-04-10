"""Convert CN-Celeb FLAC files to 16 kHz mono WAV in place.

CN-Celeb (OpenSLR #82) ships as ``CN-Celeb_flac/data/<speaker_id>/*.flac``
at mixed sample rates. Wulfenite training expects 16 kHz mono WAV, so this
script walks the tree, converts each FLAC to a 16 kHz mono 16-bit PCM WAV
(writing the .wav alongside the original .flac), and optionally deletes the
original FLAC files afterwards.

Already-compliant .wav files are skipped.

Usage:

    uv run --directory python python -m wulfenite.scripts.resample_cnceleb \\
        --root ../assets/CN-Celeb_flac
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
from pathlib import Path

import soundfile as sf
import torch
import torchaudio.functional as AF
from tqdm.auto import tqdm


TARGET_SR = 16000


def _resolve_base(root: Path) -> Path:
    if (root / "CN-Celeb_flac").exists():
        return root / "CN-Celeb_flac" / "data"
    if (root / "cn-celeb_v2").exists():
        return root / "cn-celeb_v2" / "data"
    if (root / "data").exists():
        return root / "data"
    raise SystemExit(
        f"CN-Celeb layout not found under {root}. Expected CN-Celeb_flac/data/ "
        "or data/."
    )


def _scan_files(base: Path) -> list[Path]:
    """Find all audio files (FLAC + WAV) under the data tree."""
    files = sorted(base.rglob("*.flac"))
    files.extend(sorted(base.rglob("*.wav")))
    return files


def _resample_one(job_path: str) -> tuple[str, str]:
    path = Path(job_path)
    # Output is always .wav (convert FLAC → WAV)
    out_path = path.with_suffix(".wav")

    # If a .wav already exists and is compliant, skip
    if out_path.exists() and out_path != path:
        try:
            info = sf.info(str(out_path))
            if info.samplerate == TARGET_SR and info.channels == 1:
                return (str(path), "skip")
        except Exception:
            pass  # re-process

    try:
        info = sf.info(str(path))
    except Exception as e:
        return (str(path), f"error:info:{e}")

    # If source is already a compliant wav, skip
    if path.suffix == ".wav" and info.samplerate == TARGET_SR and info.channels == 1:
        return (str(path), "skip")

    try:
        data, sr = sf.read(str(path), dtype="float32", always_2d=False)
    except Exception as e:
        return (str(path), f"error:read:{e}")

    if data.ndim == 2:
        data = data.mean(axis=1)

    wav = torch.from_numpy(data).unsqueeze(0)
    if sr != TARGET_SR:
        wav = AF.resample(wav, orig_freq=sr, new_freq=TARGET_SR)

    try:
        sf.write(str(out_path), wav.squeeze(0).numpy(), TARGET_SR, subtype="PCM_16")
    except Exception as e:
        return (str(path), f"error:write:{e}")

    return (str(path), "done")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True,
                        help="Path to the CN-Celeb v2 root directory.")
    parser.add_argument("--num-workers", type=int,
                        default=max(1, (os.cpu_count() or 1) - 1),
                        help="Parallel worker processes.")
    parser.add_argument("--dry-run", action="store_true",
                        help="List what would be processed without writing.")
    args = parser.parse_args()

    base = _resolve_base(args.root)
    files = _scan_files(base)
    if not files:
        raise SystemExit(
            f"No audio files found under {base}. Check --root points at CN-Celeb."
        )

    needs_work = 0
    already_ok = 0
    scan_bar = tqdm(files, desc="scan headers", unit="file", dynamic_ncols=True)
    for path in scan_bar:
        try:
            info = sf.info(str(path))
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

    done = 0
    skipped = 0
    errors: list[tuple[str, str]] = []
    n_workers = max(1, min(args.num_workers, needs_work))
    paths_as_str = [str(path) for path in files]

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
            pbar.set_postfix(done=done, skip=skipped, err=len(errors), refresh=False)

    print(f"[done] resampled={done} skipped={skipped} errors={len(errors)}")
    if errors:
        print("[errors]")
        for path, reason in errors[:20]:
            print(f"  {reason}: {path}")
        if len(errors) > 20:
            print(f"  ... and {len(errors) - 20} more")


if __name__ == "__main__":
    main()
