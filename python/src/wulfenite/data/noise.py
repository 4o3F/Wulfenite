"""Generic noise dataset scanner.

A dataset-agnostic recursive scan for 16 kHz mono ``.wav`` files,
used by the mixer to sample additive noise for the final mixture.
Works unchanged with any reasonable noise corpus:

- **MUSAN** (openslr #17, ~3.6 GB for the ``noise/`` subset only).
  Recommended default because it is small, widely used in speech
  research, and distributed as 16 kHz mono wavs.
- **DEMAND** (zenodo 1227121, ~2 GB, 18 real environments).
- **DNS Challenge 4/5** (~80 GB, much larger than needed for our use case).
- **FSD50K**, **ESC-50**, or any custom recording set.

The scanner treats all files under ``root`` as a single flat pool
and returns them as :class:`NoiseEntry` objects with cached
``num_frames``; the mixer samples uniformly at random and draws a
random window of the requested length.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import soundfile as sf


EXPECTED_SAMPLE_RATE = 16000


@dataclass(frozen=True)
class NoiseEntry:
    """One noise recording.

    Attributes:
        path: absolute path to the ``.wav`` file.
        num_frames: cached from the file header at scan time so the
            mixer never re-opens the file just to check length.
    """

    path: Path
    num_frames: int


def scan_noise_dir(
    root: Path | str,
    min_duration_seconds: float = 1.0,
) -> list[NoiseEntry]:
    """Recursively scan a noise directory for 16 kHz mono wav files.

    Args:
        root: path to the directory containing noise ``.wav`` files.
            Subdirectories are walked recursively, so you can point
            this at e.g. ``musan/noise/`` and it will pick up both
            ``free-sound/`` and ``sound-bible/`` subdirs.
        min_duration_seconds: drop files shorter than this. Very
            short clips don't give the mixer enough room for random
            window sampling.

    Returns:
        List of :class:`NoiseEntry`, one per eligible file. Raises
        ``RuntimeError`` if no files were found or the root does
        not exist.
    """
    root = Path(root)
    if not root.exists():
        raise RuntimeError(f"Noise root does not exist: {root}")

    min_frames = int(min_duration_seconds * EXPECTED_SAMPLE_RATE)
    entries: list[NoiseEntry] = []
    for wav in sorted(root.rglob("*.wav")):
        try:
            info = sf.info(str(wav))
        except Exception:
            continue
        if info.samplerate != EXPECTED_SAMPLE_RATE or info.channels != 1:
            continue
        if info.frames < min_frames:
            continue
        entries.append(NoiseEntry(path=wav, num_frames=info.frames))

    if not entries:
        raise RuntimeError(
            f"No 16 kHz mono wav files found under {root}. "
            "Check the path, verify files are 16 kHz mono, and that "
            "recursive wav search finds your corpus layout."
        )
    return entries


def scan_noise_dirs(
    roots: dict[str, Path | str],
    min_duration_seconds: float = 1.0,
) -> dict[str, list[NoiseEntry]]:
    """Scan multiple categorized noise roots.

    Args:
        roots: mapping of category name to directory path.
        min_duration_seconds: minimum duration required per noise file.

    Returns:
        Dict of category name -> list of :class:`NoiseEntry`.

    Raises:
        ValueError: if ``roots`` is empty.
        RuntimeError: if any configured category does not contain valid noise.
    """
    if not roots:
        raise ValueError("roots must not be empty")
    categorized: dict[str, list[NoiseEntry]] = {}
    for category, root in roots.items():
        categorized[category] = scan_noise_dir(
            root,
            min_duration_seconds=min_duration_seconds,
        )
    return categorized
