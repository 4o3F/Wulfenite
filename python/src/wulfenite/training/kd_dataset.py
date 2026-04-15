"""Dataset helpers for TinyECAPA contrastive knowledge distillation."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
import random
from typing import TypedDict

import soundfile as sf
import torch
from torch.utils.data import Dataset

from wulfenite.data import AudioEntry


class TinyECAPAKDBatch(TypedDict):
    student_waveform: torch.Tensor
    teacher_waveform: torch.Tensor
    speaker_id: str


def _eligible_speakers(
    speakers: dict[str, list[AudioEntry]],
) -> dict[str, list[AudioEntry]]:
    return {speaker_id: entries for speaker_id, entries in speakers.items() if len(entries) >= 2}


class TinyECAPAKDDataset(Dataset[TinyECAPAKDBatch]):
    """Sample same-speaker waveform pairs for TinyECAPA KD."""

    def __init__(
        self,
        speakers: dict[str, list[AudioEntry]],
        *,
        excerpt_length: int = 48000,
        epoch_size: int = 1000,
        sample_rate: int = 16000,
        audio_cache_size: int = 128,
        seed: int = 0,
    ) -> None:
        eligible_speakers = _eligible_speakers(speakers)
        if not eligible_speakers:
            raise ValueError("TinyECAPAKDDataset requires speakers with >=2 utterances")
        self.speakers = eligible_speakers
        self.speaker_ids = sorted(eligible_speakers)
        self.excerpt_length = excerpt_length
        self.epoch_size = epoch_size
        self.sample_rate = sample_rate
        self.audio_cache_size = audio_cache_size
        self.seed = seed
        self._epoch = 0
        self._audio_cache: OrderedDict[Path, torch.Tensor] = OrderedDict()

    def __len__(self) -> int:
        return self.epoch_size

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError(f"epoch must be >= 0, got {epoch}")
        self._epoch = epoch

    def _rng(self, index: int) -> random.Random:
        return random.Random(hash((self.seed, self._epoch, index)) & 0xFFFFFFFF)

    def _cache_audio(self, path: Path, audio: torch.Tensor) -> None:
        if self.audio_cache_size <= 0:
            return
        self._audio_cache[path] = audio
        self._audio_cache.move_to_end(path)
        while len(self._audio_cache) > self.audio_cache_size:
            self._audio_cache.popitem(last=False)

    def _load_audio(self, path: Path) -> torch.Tensor:
        cached = self._audio_cache.get(path)
        if cached is not None:
            self._audio_cache.move_to_end(path)
            return cached.clone()
        audio, sample_rate = sf.read(str(path), dtype="float32")
        if sample_rate != self.sample_rate:
            raise RuntimeError(
                f"Expected {self.sample_rate} Hz audio, got {sample_rate} Hz at {path}"
            )
        if getattr(audio, "ndim", 1) != 1:
            raise RuntimeError(f"Expected mono audio at {path}")
        tensor = torch.from_numpy(audio)
        self._cache_audio(path, tensor)
        return tensor.clone()

    def _sample_speaker_segment(
        self,
        entries: list[AudioEntry],
        length: int,
        rng: random.Random,
        *,
        anchor_entry: AudioEntry | None = None,
        avoid_paths: set[Path] | None = None,
    ) -> torch.Tensor:
        blocked_paths = avoid_paths or set()
        available_entries = [entry for entry in entries if entry.path not in blocked_paths]
        if not available_entries:
            raise ValueError("No KD entries available after exclusions")
        selected_entries: list[AudioEntry] = []
        total_frames = 0
        if anchor_entry is not None and anchor_entry.path not in blocked_paths:
            selected_entries.append(anchor_entry)
            total_frames += anchor_entry.num_frames
        chosen_paths = {entry.path for entry in selected_entries}
        while total_frames < length:
            candidates = [entry for entry in available_entries if entry.path not in chosen_paths]
            if not candidates:
                candidates = available_entries
            chosen = rng.choice(candidates)
            selected_entries.append(chosen)
            chosen_paths.add(chosen.path)
            total_frames += chosen.num_frames
        merged = torch.cat([self._load_audio(entry.path) for entry in selected_entries], dim=0)
        if merged.numel() >= length:
            max_start = merged.numel() - length
            start = rng.randint(0, max_start) if max_start > 0 else 0
            return merged[start : start + length].clone()
        return torch.nn.functional.pad(merged, (0, length - merged.numel()))

    def __getitem__(self, index: int) -> TinyECAPAKDBatch:
        rng = self._rng(index)
        speaker_id = rng.choice(self.speaker_ids)
        teacher_entry, student_entry = rng.sample(self.speakers[speaker_id], 2)
        teacher_waveform = self._sample_speaker_segment(
            self.speakers[speaker_id],
            self.excerpt_length,
            rng,
            anchor_entry=teacher_entry,
        )
        student_waveform = self._sample_speaker_segment(
            self.speakers[speaker_id],
            self.excerpt_length,
            rng,
            anchor_entry=student_entry,
            avoid_paths={teacher_entry.path},
        )
        return {
            "student_waveform": student_waveform,
            "teacher_waveform": teacher_waveform,
            "speaker_id": speaker_id,
        }


def split_speakers_for_kd(
    speakers: dict[str, list[AudioEntry]],
    *,
    val_fraction: float = 0.05,
    seed: int = 0,
) -> tuple[dict[str, list[AudioEntry]], dict[str, list[AudioEntry]]]:
    """Split speakers into disjoint train/validation partitions."""
    if not 0.0 <= val_fraction <= 1.0:
        raise ValueError(f"val_fraction must be in [0, 1], got {val_fraction}")
    eligible_speakers = _eligible_speakers(speakers)
    speaker_ids = sorted(eligible_speakers)
    rng = random.Random(seed)
    rng.shuffle(speaker_ids)
    val_count = int(round(len(speaker_ids) * val_fraction))
    if val_fraction > 0.0 and val_count == 0 and len(speaker_ids) > 1:
        val_count = 1
    if val_count >= len(speaker_ids) and len(speaker_ids) > 1:
        val_count = len(speaker_ids) - 1
    val_ids = set(speaker_ids[:val_count])
    train_speakers = {
        speaker_id: entries for speaker_id, entries in eligible_speakers.items()
        if speaker_id not in val_ids
    }
    val_speakers = {
        speaker_id: entries for speaker_id, entries in eligible_speakers.items()
        if speaker_id in val_ids
    }
    return train_speakers, val_speakers


__all__ = ["TinyECAPAKDBatch", "TinyECAPAKDDataset", "split_speakers_for_kd"]
