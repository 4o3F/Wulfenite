"""AISHELL-1, AISHELL-3, and CN-Celeb dataset scanners.

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
                speaker_from_dir: bool = True,
                diagnostics: dict | None = None,
                extensions: tuple[str, ...] = ("*.wav",)) -> list[AudioEntry]:
    """Walk a ``{split_dir}/{speaker_id}/*.wav`` tree and return entries.

    Files with unexpected sample rate or channel count are dropped so
    the mixer sees a homogeneous pool. If a ``diagnostics`` dict is
    passed in, this function updates it with per-rejection counters
    so the caller can build an informative error when the returned
    list is empty:

    - ``seen``: total audio files found
    - ``wrong_sr``: dropped because sample rate != 16000
    - ``wrong_channels``: dropped because channels != 1
    - ``sample_rates``: set of the rejected sample rates, for hints
    - ``unreadable``: couldn't read the file header

    AISHELL-1 ships at 16 kHz mono natively and needs no conversion.
    **AISHELL-3 ships at 44.1 kHz** and must be resampled to 16 kHz
    mono before the scanner will accept it — see
    ``python/src/wulfenite/scripts/resample_aishell3.py``.

    The ``extensions`` parameter controls which file globs are searched
    (default ``("*.wav",)``). Pass ``("*.wav", "*.flac")`` for datasets
    like CN-Celeb that ship in FLAC format.
    """
    entries: list[AudioEntry] = []
    if not split_dir.exists():
        return entries
    for spk_dir in sorted(split_dir.iterdir()):
        if not spk_dir.is_dir():
            continue
        spk_id = spk_dir.name if speaker_from_dir else spk_dir.stem
        audio_files: list[Path] = []
        for ext in extensions:
            audio_files.extend(spk_dir.glob(ext))
        for wav in sorted(audio_files):
            if diagnostics is not None:
                diagnostics["seen"] = diagnostics.get("seen", 0) + 1
            try:
                info = sf.info(str(wav))
            except Exception:
                if diagnostics is not None:
                    diagnostics["unreadable"] = diagnostics.get("unreadable", 0) + 1
                continue
            if info.samplerate != EXPECTED_SAMPLE_RATE:
                if diagnostics is not None:
                    diagnostics["wrong_sr"] = diagnostics.get("wrong_sr", 0) + 1
                    diagnostics.setdefault("sample_rates", set()).add(info.samplerate)
                continue
            if info.channels != 1:
                if diagnostics is not None:
                    diagnostics["wrong_channels"] = diagnostics.get("wrong_channels", 0) + 1
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

    diagnostics: dict = {}
    entries: list[AudioEntry] = []
    scanned_dirs: list[Path] = []
    for split in splits:
        split_dir = base / "wav" / split
        scanned_dirs.append(split_dir)
        entries.extend(
            _scan_split(split_dir, dataset="aishell1", diagnostics=diagnostics)
        )

    if not entries:
        raise RuntimeError(_format_empty_scan_error(
            "AISHELL-1", scanned_dirs, diagnostics,
            extra_hint=(
                "Check that the archive was extracted and the per-speaker "
                ".tar.gz files inside wav/{train,dev,test}/ were unpacked "
                "(see TRAIN.md section 2)."
            ),
        ))
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
        root: path to the AISHELL-3 root. Accepts either the directory
            that contains ``data_aishell3/`` (the tarball's default
            wrapper) or ``data_aishell3/`` itself (after stripping
            the wrapper with ``--strip-components=1``).
        splits: which splits to include.
        min_utts_per_speaker: drop speakers with fewer utterances.

    **Warning**: the official AISHELL-3 distribution is at 44.1 kHz,
    not 16 kHz. This scanner only accepts 16 kHz mono wavs (to match
    AISHELL-1 and the model's sample rate). Run
    ``python -m wulfenite.scripts.resample_aishell3`` once after
    extraction to produce a 16 kHz version in place. The error
    raised when no files are found includes an explicit "found N
    files at sample rate X" summary so this pitfall is easy to spot.
    """
    root = Path(root)
    # Auto-detect the tarball's data_aishell3/ wrapper.
    if (root / "data_aishell3").exists():
        base = root / "data_aishell3"
    else:
        base = root

    diagnostics: dict = {}
    entries: list[AudioEntry] = []
    scanned_dirs: list[Path] = []
    for split in splits:
        split_dir = base / split / "wav"
        scanned_dirs.append(split_dir)
        entries.extend(
            _scan_split(split_dir, dataset="aishell3", diagnostics=diagnostics)
        )

    if not entries:
        raise RuntimeError(_format_empty_scan_error(
            "AISHELL-3", scanned_dirs, diagnostics,
            extra_hint=(
                "AISHELL-3 is distributed at 44.1 kHz; Wulfenite needs 16 kHz "
                "mono. Run `python -m wulfenite.scripts.resample_aishell3 "
                "--root ../assets/aishell3` once to convert in place, then "
                "re-run training. See docs/TRAIN.md section 2."
            ),
        ))
    return _group_by_speaker(entries, min_utts_per_speaker)


# ---------------------------------------------------------------------------
# CN-Celeb
# ---------------------------------------------------------------------------


def scan_cnceleb(
    root: Path | str,
    min_utts_per_speaker: int = 2,
) -> dict[str, list[AudioEntry]]:
    """Scan CN-Celeb v2 and return ``{speaker_id: [entries]}``.

    CN-Celeb is distributed at mixed sample rates. This scanner accepts
    only 16 kHz mono wavs and rejects everything else with the same
    diagnostics used by the AISHELL scanners.
    """
    root = Path(root)
    # Auto-detect common layouts:
    #   CN-Celeb_flac/data/{id00001,...}/*.flac  (OpenSLR #82 default)
    #   cn-celeb_v2/data/{id00001,...}/*.wav      (legacy / resampled)
    #   data/{id00001,...}/*                       (bare extraction)
    if (root / "CN-Celeb_flac").exists():
        base = root / "CN-Celeb_flac" / "data"
    elif (root / "cn-celeb_v2").exists():
        base = root / "cn-celeb_v2" / "data"
    elif (root / "data").exists():
        base = root / "data"
    else:
        raise RuntimeError(
            f"CN-Celeb layout not found under {root}. "
            "Expected CN-Celeb_flac/data/{{id00001,...}}/ or data/{{id00001,...}}/"
        )

    diagnostics: dict = {}
    entries = _scan_split(
        base, dataset="cnceleb", diagnostics=diagnostics,
        extensions=("*.wav", "*.flac"),
    )
    if not entries:
        raise RuntimeError(_format_empty_scan_error(
            "CN-Celeb", [base], diagnostics,
            extra_hint=(
                "CN-Celeb ships as FLAC at mixed sample rates; Wulfenite needs "
                "16 kHz mono WAV. Run `python -m wulfenite.scripts.resample_cnceleb "
                "--root ../assets/CN-Celeb_flac` once, then re-run training."
            ),
        ))
    return _group_by_speaker(entries, min_utts_per_speaker)


# ---------------------------------------------------------------------------
# Error formatting
# ---------------------------------------------------------------------------


def _format_empty_scan_error(
    dataset_name: str,
    scanned_dirs: list[Path],
    diagnostics: dict,
    extra_hint: str = "",
) -> str:
    """Build an informative error message when a scan turns up nothing."""
    lines = [f"No {dataset_name} wavs found."]
    lines.append("Scanned the following directories:")
    for d in scanned_dirs:
        marker = "✓" if d.exists() else "✗ (does not exist)"
        lines.append(f"  {marker} {d}")

    seen = diagnostics.get("seen", 0)
    if seen == 0:
        lines.append("No .wav files were found at all under those paths.")
    else:
        lines.append(f"Found {seen} .wav file(s), but all were rejected:")
        wrong_sr = diagnostics.get("wrong_sr", 0)
        if wrong_sr:
            rates = diagnostics.get("sample_rates", set())
            rates_str = ", ".join(sorted(str(int(r)) for r in rates))
            lines.append(
                f"  - {wrong_sr} with wrong sample rate "
                f"(required 16000 Hz, found: {rates_str} Hz)"
            )
        wrong_ch = diagnostics.get("wrong_channels", 0)
        if wrong_ch:
            lines.append(
                f"  - {wrong_ch} not mono (required 1 channel)"
            )
        unreadable = diagnostics.get("unreadable", 0)
        if unreadable:
            lines.append(
                f"  - {unreadable} could not be opened (corrupt or not wav)"
            )

    if extra_hint:
        lines.append("")
        lines.append(extra_hint)
    return "\n".join(lines)


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
