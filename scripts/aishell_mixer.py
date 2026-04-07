"""On-the-fly AISHELL-1 two-speaker mixer for Phase 0a training.

AISHELL-1 layout after extraction::

    <root>/data_aishell/
        wav/
            train/
                S0002/BAC009S0002W0122.wav
                ...
            dev/
                S0XXX/...
            test/
                S0XXX/...
        transcript/
            aishell_transcript_v0.8.txt

Each speaker has ~300 utterances of 3-8 s, 16 kHz mono. This dataset scans
the directory tree once, groups wavs by speaker id (the parent folder name
such as ``S0002``), and generates 2-speaker mixtures on-the-fly:

    target_utt + interferer_utt -> (mixture, target_clean, target_enrollment)

where ``target_enrollment`` is a DIFFERENT utterance of the same target
speaker (so the model cannot cheat by memorizing the target waveform).

The dataset yields fixed-length chunks of ``segment_seconds`` (default 4 s)
for both the mixture and the enrollment, to keep batching simple and to
bound VRAM usage.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import soundfile as sf
import torch
from torch.utils.data import Dataset

from augmentation import AugmentationConfig, augment_triplet, add_mixture_noise


@dataclass
class AishellEntry:
    speaker_id: str
    path: Path
    num_frames: int  # cached at scan time to avoid re-reading headers


def _scan_split(split_dir: Path) -> list[AishellEntry]:
    """Walk one split directory (train/dev/test) and return all utterances."""
    if not split_dir.exists():
        return []
    entries: list[AishellEntry] = []
    for spk_dir in sorted(split_dir.iterdir()):
        if not spk_dir.is_dir():
            continue
        spk_id = spk_dir.name
        for wav in sorted(spk_dir.glob("*.wav")):
            try:
                info = sf.info(str(wav))
            except Exception:
                continue
            if info.samplerate != 16000 or info.channels != 1:
                continue
            entries.append(AishellEntry(speaker_id=spk_id, path=wav, num_frames=info.frames))
    return entries


def _load_chunk(path: Path, target_len: int, rng: random.Random) -> torch.Tensor:
    """Load a random ``target_len``-sample chunk from a wav file, padding if short."""
    info = sf.info(str(path))
    n = info.frames
    if n >= target_len:
        start = rng.randint(0, n - target_len)
        data, _ = sf.read(str(path), start=start, stop=start + target_len, dtype="float32", always_2d=False)
    else:
        data, _ = sf.read(str(path), dtype="float32", always_2d=False)
        pad = target_len - data.shape[0]
        data = torch.from_numpy(data)
        return torch.nn.functional.pad(data, (0, pad))
    return torch.from_numpy(data)


def _rms(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return torch.sqrt(torch.mean(x * x) + eps)


def _rescale_to_snr(target: torch.Tensor, interferer: torch.Tensor,
                    snr_db: float) -> torch.Tensor:
    """Scale the interferer so that target vs scaled interferer hits ``snr_db``."""
    target_rms = _rms(target)
    interf_rms = _rms(interferer)
    target_factor = 10.0 ** (snr_db / 20.0)
    # We want target_rms / (interferer_rms * k) == target_factor
    k = target_rms / (interf_rms * target_factor + 1e-12)
    return interferer * k


class AishellMixDataset(Dataset):
    """Iterable-style via __len__+__getitem__ over a virtual epoch.

    Each __getitem__ call draws two random speakers, picks two utterances
    from the target (one for clean + one for enrollment) and one from the
    interferer, and produces a mixture at a random SNR.
    """

    def __init__(
        self,
        aishell_root: Path | str,
        split: str = "train",
        samples_per_epoch: int = 20000,
        segment_seconds: float = 4.0,
        enrollment_seconds: float = 4.0,
        snr_range_db: tuple[float, float] = (-5.0, 5.0),
        sample_rate: int = 16000,
        seed: int | None = None,
        augment: bool = False,
        aug_config: AugmentationConfig | None = None,
    ):
        aishell_root = Path(aishell_root)
        split_dir = aishell_root / "data_aishell" / "wav" / split
        if not split_dir.exists():
            # Also accept the root pointing directly at wav/
            alt = aishell_root / "wav" / split
            if alt.exists():
                split_dir = alt
        entries = _scan_split(split_dir)
        if not entries:
            raise RuntimeError(
                f"No AISHELL-1 16 kHz mono wavs found under {split_dir}. "
                "Check that the archive was extracted and the path is correct."
            )

        # Group by speaker
        by_spk: dict[str, list[AishellEntry]] = {}
        for e in entries:
            by_spk.setdefault(e.speaker_id, []).append(e)
        # Drop speakers with <2 utterances (we need 2 for target: clean + enrollment)
        by_spk = {k: v for k, v in by_spk.items() if len(v) >= 2}
        if len(by_spk) < 2:
            raise RuntimeError("Need at least 2 speakers with >=2 utterances each.")

        self.by_spk = by_spk
        self.speaker_ids: list[str] = sorted(by_spk.keys())
        self.samples_per_epoch = samples_per_epoch
        self.segment_len = int(segment_seconds * sample_rate)
        self.enrollment_len = int(enrollment_seconds * sample_rate)
        self.snr_range_db = snr_range_db
        self.sample_rate = sample_rate
        # If seed is None, each __getitem__ call draws from fresh randomness
        # (so consecutive epochs see different mixtures). If seed is set,
        # sampling is deterministic by (base_seed, index), useful for the
        # validation split.
        self._base_seed = seed
        self.augment = augment
        self.aug_config = aug_config or AugmentationConfig()

        print(
            f"[AishellMix] split={split} speakers={len(self.speaker_ids)} "
            f"utterances={sum(len(v) for v in by_spk.values())} "
            f"virtual-epoch={samples_per_epoch} augment={augment}"
        )

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __getitem__(self, index: int):
        if self._base_seed is None:
            rng = random.Random()  # fresh entropy each call
        else:
            rng = random.Random(self._base_seed + index)

        # Two distinct speakers
        target_spk, interf_spk = rng.sample(self.speaker_ids, 2)
        target_utts = self.by_spk[target_spk]
        interf_utts = self.by_spk[interf_spk]

        target_clean_entry, target_enroll_entry = rng.sample(target_utts, 2)
        interf_entry = rng.choice(interf_utts)

        target_clean = _load_chunk(target_clean_entry.path, self.segment_len, rng)
        interf = _load_chunk(interf_entry.path, self.segment_len, rng)
        enrollment = _load_chunk(target_enroll_entry.path, self.enrollment_len, rng)

        # Normalize each source to unit-ish RMS
        tgt_rms = _rms(target_clean)
        target_clean = target_clean / (tgt_rms + 1e-8) * 0.1
        interf_rms = _rms(interf)
        interf = interf / (interf_rms + 1e-8) * 0.1
        enrollment = enrollment / (_rms(enrollment) + 1e-8) * 0.1

        # Acoustic augmentation BEFORE mixing, so target and interferer sit in
        # different virtual spatial positions of the same room. The
        # REVERBERATED target becomes the training "clean target" — the model
        # is not asked to dereverb, only to separate.
        if self.augment:
            target_clean, interf, enrollment = augment_triplet(
                target_clean, interf, enrollment, self.aug_config, rng
            )
            # Re-normalize after convolution can change RMS considerably
            target_clean = target_clean / (_rms(target_clean) + 1e-8) * 0.1
            interf = interf / (_rms(interf) + 1e-8) * 0.1
            enrollment = enrollment / (_rms(enrollment) + 1e-8) * 0.1

        snr_db = rng.uniform(*self.snr_range_db)
        interf_scaled = _rescale_to_snr(target_clean, interf, snr_db)

        mixture = target_clean + interf_scaled

        # Light broadband noise on the mixture only (room tone / mic self noise)
        if self.augment:
            mixture = add_mixture_noise(mixture, self.aug_config, rng)

        # Guard against clipping by rescaling the whole triplet if needed
        peak = mixture.abs().max().item()
        if peak > 0.99:
            scale = 0.99 / peak
            mixture = mixture * scale
            target_clean = target_clean * scale

        return {
            "mixture": mixture,            # [T_mix]
            "target": target_clean,        # [T_mix]  <- reverberated target
            "enrollment": enrollment,      # [T_enr]  <- reverberated enrollment
            "snr_db": torch.tensor(snr_db, dtype=torch.float32),
        }


def collate_mix_batch(batch: Sequence[dict]) -> dict:
    return {
        "mixture": torch.stack([b["mixture"] for b in batch], dim=0),
        "target": torch.stack([b["target"] for b in batch], dim=0),
        "enrollment": torch.stack([b["enrollment"] for b in batch], dim=0),
        "snr_db": torch.stack([b["snr_db"] for b in batch], dim=0),
    }
