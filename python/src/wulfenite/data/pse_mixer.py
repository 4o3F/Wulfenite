"""On-the-fly PSE scene mixer."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
import random

import soundfile as sf
import torch
from torch.utils.data import Dataset

from .aishell import AudioEntry
from .augmentation import (
    ReverbConfig,
    apply_bandwidth_limit,
    apply_random_gain,
    apply_rir,
    scale_noise_to_snr,
    synth_room_rir,
)
from .noise import NoiseEntry


def _filter_target_speakers(
    speakers: dict[str, list[AudioEntry]],
) -> dict[str, list[AudioEntry]]:
    return {spk: utts for spk, utts in speakers.items() if len(utts) >= 2}


def _filter_interferer_speakers(
    speakers: dict[str, list[AudioEntry]],
) -> dict[str, list[AudioEntry]]:
    return {spk: utts for spk, utts in speakers.items() if utts}


class PSEMixer(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]]):
    """Sample target/enrollment pairs with interference and additive noise."""

    def __init__(
        self,
        speakers: dict[str, list[AudioEntry]],
        noises: list[NoiseEntry] | None = None,
        *,
        interferer_speakers: dict[str, list[AudioEntry]] | None = None,
        epoch_size: int = 1000,
        segment_length: int = 160000,
        enrollment_length: int = 240000,
        sample_rate: int = 16000,
        reverb_config: ReverbConfig | None = None,
        reverb_probability: float = 0.3,
        gain_probability: float = 0.3,
        bandwidth_limit_probability: float = 0.2,
        audio_cache_size: int = 128,
        seed: int = 0,
    ) -> None:
        if not speakers:
            raise ValueError("speakers must not be empty")
        target_speakers = _filter_target_speakers(speakers)
        if not target_speakers:
            raise ValueError("at least one speaker with >=2 utterances is required")
        effective_interferers = interferer_speakers if interferer_speakers is not None else speakers
        self.target_speakers = target_speakers
        self.target_speaker_ids = sorted(target_speakers)
        self.interferer_speakers = _filter_interferer_speakers(effective_interferers)
        self.interferer_speaker_ids = sorted(self.interferer_speakers)
        self.noises = noises or []
        self.epoch_size = epoch_size
        self.segment_length = segment_length
        self.enrollment_length = enrollment_length
        self.sample_rate = sample_rate
        self.reverb_config = reverb_config if reverb_config is not None else ReverbConfig(sample_rate=sample_rate)
        self.reverb_probability = reverb_probability
        self.gain_probability = gain_probability
        self.bandwidth_limit_probability = bandwidth_limit_probability
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
        segment, _used_paths = self._sample_speaker_segment_tracked(
            entries,
            length,
            rng,
            anchor_entry=anchor_entry,
            avoid_paths=avoid_paths,
        )
        return segment

    def _sample_speaker_segment_tracked(
        self,
        entries: list[AudioEntry],
        length: int,
        rng: random.Random,
        *,
        anchor_entry: AudioEntry | None = None,
        avoid_paths: set[Path] | None = None,
    ) -> tuple[torch.Tensor, set[Path]]:
        blocked_paths = avoid_paths or set()
        available_entries = [entry for entry in entries if entry.path not in blocked_paths]
        if not available_entries:
            raise ValueError("No speaker entries available after exclusions")
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
            if len(available_entries) == 1 and total_frames >= length:
                break
        merged = torch.cat([self._load_audio(entry.path) for entry in selected_entries], dim=0)
        if merged.numel() >= length:
            max_start = merged.numel() - length
            start = rng.randint(0, max_start) if max_start > 0 else 0
            segment = merged[start : start + length].clone()
        else:
            segment = torch.nn.functional.pad(merged, (0, length - merged.numel()))
        return segment, {entry.path for entry in selected_entries}

    def _sample_noise(
        self,
        length: int,
        rng: random.Random,
    ) -> torch.Tensor | None:
        if not self.noises:
            return None
        entry = rng.choice(self.noises)
        audio = self._load_audio(entry.path)
        if audio.numel() >= length:
            max_start = audio.numel() - length
            start = rng.randint(0, max_start) if max_start > 0 else 0
            return audio[start : start + length].clone()
        reps = (length + audio.numel() - 1) // max(audio.numel(), 1)
        return audio.repeat(reps)[:length].clone()

    def _sample_target_pair(
        self,
        rng: random.Random,
    ) -> tuple[str, AudioEntry, AudioEntry]:
        speaker_id = rng.choice(self.target_speaker_ids)
        utts = rng.sample(self.target_speakers[speaker_id], 2)
        return speaker_id, utts[0], utts[1]

    def _sample_interferer(
        self,
        target_speaker: str,
        rng: random.Random,
    ) -> tuple[str, AudioEntry] | None:
        candidates = [spk for spk in self.interferer_speaker_ids if spk != target_speaker]
        if not candidates:
            return None
        speaker_id = rng.choice(candidates)
        return speaker_id, rng.choice(self.interferer_speakers[speaker_id])

    def _maybe_reverb(self, signal: torch.Tensor, rng: random.Random) -> torch.Tensor:
        if rng.random() >= self.reverb_probability:
            return signal
        rir = synth_room_rir(self.reverb_config, rng)
        return apply_rir(signal, rir)

    def _draw_scene_components(self, rng: random.Random) -> tuple[bool, bool]:
        include_noise = False
        include_interference = False
        draw = rng.random()
        if draw < 0.20:
            include_noise = True
        elif draw < 0.50:
            include_interference = True
        else:
            include_noise = True
            include_interference = True
        return include_noise, include_interference

    def _sample_snr_db(self, rng: random.Random) -> float:
        return max(-5.0, min(35.0, rng.gauss(15.0, 10.0)))

    def _sample_sir_db(self, rng: random.Random) -> float:
        return max(-5.0, min(25.0, rng.gauss(10.0, 7.5)))

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
        rng = self._rng(index)
        speaker_id, target_entry, enroll_entry = self._sample_target_pair(rng)
        # Reserve enrollment anchor so target sampling never consumes it.
        # This guarantees enrollment always has at least one disjoint file.
        target, target_paths = self._sample_speaker_segment_tracked(
            self.target_speakers[speaker_id],
            self.segment_length,
            rng,
            anchor_entry=target_entry,
            avoid_paths={enroll_entry.path},
        )
        enrollment = self._sample_speaker_segment(
            self.target_speakers[speaker_id],
            self.enrollment_length,
            rng,
            anchor_entry=enroll_entry,
            avoid_paths=target_paths,
        )
        target = self._maybe_reverb(target, rng)
        mixture = target.clone()
        include_noise, include_interference = self._draw_scene_components(rng)

        if include_interference:
            interferer_sample = self._sample_interferer(speaker_id, rng)
            if interferer_sample is not None:
                interferer_speaker_id, interferer_entry = interferer_sample
                interferer = self._sample_speaker_segment(
                    self.interferer_speakers[interferer_speaker_id],
                    self.segment_length,
                    rng,
                    anchor_entry=interferer_entry,
                )
                interferer = self._maybe_reverb(interferer, rng)
                sir_db = self._sample_sir_db(rng)
                mixture = mixture + scale_noise_to_snr(target, interferer, sir_db, rng=rng)

        if include_noise:
            noise = self._sample_noise(self.segment_length, rng)
            if noise is not None:
                snr_db = self._sample_snr_db(rng)
                mixture = mixture + scale_noise_to_snr(target, noise, snr_db, rng=rng)

        if rng.random() < self.gain_probability:
            mixture = apply_random_gain(mixture, rng=rng)
        if rng.random() < self.bandwidth_limit_probability:
            mixture = apply_bandwidth_limit(mixture, sample_rate=self.sample_rate, rng=rng)

        peak = max(
            float(mixture.abs().max()),
            float(target.abs().max()),
            float(enrollment.abs().max()),
            1.0,
        )
        if peak > 1.0:
            scale = 1.0 / peak
            mixture = mixture * scale
            target = target * scale
            enrollment = enrollment * scale
        return mixture, target, enrollment, speaker_id


__all__ = ["PSEMixer"]
