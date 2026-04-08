"""DNS Challenge 4 noise dataset scanner.

The DNS4 noise set is distributed as a flat-ish directory of 16 kHz
mono ``.wav`` files of various lengths, covering room tone, HVAC,
traffic, keyboard, crowd, etc. There is no speaker structure — each
file is just "some noise of some type for some duration".

For Wulfenite training we treat the whole set as a single pool and
sample uniformly at random, mixing the sampled noise into the final
mixture at a random SNR in a chosen range.
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
        path: absolute path to the .wav file.
        num_frames: cached from the file header at scan time.
    """

    path: Path
    num_frames: int


def scan_dns_noise(
    root: Path | str,
    min_duration_seconds: float = 1.0,
) -> list[NoiseEntry]:
    """Recursively scan a DNS4 noise directory.

    Args:
        root: path to the directory containing noise ``.wav`` files.
            Subdirectories are walked recursively.
        min_duration_seconds: drop files shorter than this; very
            short files don't give enough room for the mixer's
            random window sampling.

    Returns:
        List of :class:`NoiseEntry`, one per eligible file.
        Raises ``RuntimeError`` if no files were found.
    """
    root = Path(root)
    if not root.exists():
        raise RuntimeError(f"DNS noise root does not exist: {root}")

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
            f"No DNS noise wavs found under {root}. "
            "Check the path and that files are 16 kHz mono."
        )
    return entries
