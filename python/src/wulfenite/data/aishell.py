"""AISHELL-1 and AISHELL-3 dataset scanners.

Both datasets ship as directory trees of 16 kHz mono wav files grouped
by speaker. Layouts differ slightly:

AISHELL-1 (openslr #33)::

    data_aishell/
    └── wav/
        ├── train/
        │   ├── S0002/
        │   │   ├── BAC009S0002W0122.wav
        │   │   └── ...
        │   └── ...
        ├── dev/
        │   └── SXXXX/...
        └── test/
            └── SXXXX/...

AISHELL-3 (openslr #93)::

    data_aishell3/
    ├── train/
    │   └── wav/
    │       ├── SSB0005/
    │       │   └── SSB00050001.wav
    │       └── ...
    └── test/
        └── wav/
            └── SSBxxxx/...

The scanners return a dict ``{speaker_id: [Entry, ...]}`` and cache
the audio duration per file at scan time so that the mixer never
needs to call ``sf.info`` during training — this halves the
filesystem syscalls per sample and was a major bottleneck fix on v1.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import soundfile as sf


EXPECTED_SAMPLE_RATE = 16000


@dataclass(frozen=True)
class AudioEntry:
    """One utterance from a speaker.

    Attributes:
        speaker_id: canonical speaker identifier (e.g. ``"S0002"``
            for AISHELL-1, ``"SSB0005"`` for AISHELL-3).
        path: absolute path to the .wav file.
        num_frames: cached from the file header at scan time so the
            mixer does not re-open the file just to check length.
        dataset: short string tag identifying the source dataset,
            useful for filtering or diagnostic logging.
    """

    speaker_id: str
    path: Path
    num_frames: int
    dataset: str


# ---------------------------------------------------------------------------
# Scanner core
# ---------------------------------------------------------------------------


def _scan_split(split_dir: Path, dataset: str,
                speaker_from_dir: bool = True) -> list[AudioEntry]:
    """Walk a ``{split_dir}/{speaker_id}/*.wav`` tree and return entries.

    Silently drops files with unexpected sample rate or channel count —
    both AISHELL-1 and AISHELL-3 are supposed to be 16 kHz mono, so any
    file that is not is either corrupt or was resampled by mistake.
    """
    entries: list[AudioEntry] = []
    if not split_dir.exists():
        return entries
    for spk_dir in sorted(split_dir.iterdir()):
        if not spk_dir.is_dir():
            continue
        spk_id = spk_dir.name if speaker_from_dir else spk_dir.stem
        for wav in sorted(spk_dir.glob("*.wav")):
            try:
                info = sf.info(str(wav))
            except Exception:
                continue
            if info.samplerate != EXPECTED_SAMPLE_RATE or info.channels != 1:
                continue
            entries.append(AudioEntry(
                speaker_id=spk_id,
                path=wav,
                num_frames=info.frames,
                dataset=dataset,
            ))
    return entries


def _group_by_speaker(
    entries: Iterable[AudioEntry],
    min_utts_per_speaker: int = 2,
) -> dict[str, list[AudioEntry]]:
    """Group a flat entry list by speaker, dropping sparse speakers.

    Speakers with fewer than ``min_utts_per_speaker`` utterances are
    dropped because the mixer needs at least two distinct utterances
    per target speaker (one for clean / target, one for enrollment).
    """
    by_spk: dict[str, list[AudioEntry]] = {}
    for e in entries:
        by_spk.setdefault(e.speaker_id, []).append(e)
    return {
        spk: utts for spk, utts in by_spk.items()
        if len(utts) >= min_utts_per_speaker
    }


# ---------------------------------------------------------------------------
# AISHELL-1
# ---------------------------------------------------------------------------


def scan_aishell1(
    root: Path | str,
    splits: tuple[str, ...] = ("train",),
    min_utts_per_speaker: int = 2,
) -> dict[str, list[AudioEntry]]:
    """Scan AISHELL-1 and return ``{speaker_id: [entries]}``.

    Args:
        root: path to the AISHELL-1 root. Accepts either the
            directory that contains ``data_aishell/`` or
            ``data_aishell/`` itself.
        splits: which splits to include. Usually only ``"train"`` is
            needed; ``"dev"`` is useful for a small held-out set.
        min_utts_per_speaker: drop speakers with fewer utterances.

    Returns:
        Dict of speaker id → list of :class:`AudioEntry`.
        Raises ``RuntimeError`` if no entries are found (usually
        means the path is wrong or the archive was not extracted).
    """
    root = Path(root)
    if (root / "data_aishell").exists():
        base = root / "data_aishell"
    elif (root / "wav").exists():
        base = root
    else:
        raise RuntimeError(
            f"AISHELL-1 layout not found under {root}. "
            "Expected data_aishell/wav/{train,dev,test}/"
        )

    entries: list[AudioEntry] = []
    for split in splits:
        split_dir = base / "wav" / split
        entries.extend(_scan_split(split_dir, dataset="aishell1"))

    if not entries:
        raise RuntimeError(
            f"No AISHELL-1 wavs found under {base}/wav/{splits}. "
            "Check that the archive was extracted and the per-speaker "
            "tarballs were unpacked."
        )
    return _group_by_speaker(entries, min_utts_per_speaker)


# ---------------------------------------------------------------------------
# AISHELL-3
# ---------------------------------------------------------------------------


def scan_aishell3(
    root: Path | str,
    splits: tuple[str, ...] = ("train",),
    min_utts_per_speaker: int = 2,
) -> dict[str, list[AudioEntry]]:
    """Scan AISHELL-3 and return ``{speaker_id: [entries]}``.

    Args:
        root: path to the AISHELL-3 root (the directory that
            contains ``train/wav/`` and ``test/wav/``).
        splits: which splits to include.
        min_utts_per_speaker: drop speakers with fewer utterances.
    """
    root = Path(root)
    entries: list[AudioEntry] = []
    for split in splits:
        split_dir = root / split / "wav"
        entries.extend(_scan_split(split_dir, dataset="aishell3"))

    if not entries:
        raise RuntimeError(
            f"No AISHELL-3 wavs found under {root}/{splits}/wav/. "
            "Check the archive path."
        )
    return _group_by_speaker(entries, min_utts_per_speaker)


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------


def merge_speaker_dicts(
    *dicts: dict[str, list[AudioEntry]],
) -> dict[str, list[AudioEntry]]:
    """Merge several speaker dicts into one.

    Because AISHELL-1 and AISHELL-3 use disjoint speaker id prefixes
    (``SXXXX`` vs ``SSBXXXX``) there is no collision risk, but this
    helper is robust against collisions by concatenating utterance
    lists when they do happen.
    """
    merged: dict[str, list[AudioEntry]] = {}
    for d in dicts:
        for spk, utts in d.items():
            merged.setdefault(spk, []).extend(utts)
    return merged
