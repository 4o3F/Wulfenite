"""On-the-fly PSE scene mixer."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
import random
from typing import Literal

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


SceneName = Literal["target_only_degraded", "noise", "interference", "both"]

_DEFAULT_SCENE_WEIGHTS: dict[SceneName, float] = {
    "target_only_degraded": 0.0,
    "noise": 0.20,
    "interference": 0.30,
    "both": 0.50,
}
_ROOM_FAMILY_NAMES = ("small", "medium", "large")


def _filter_target_speakers(
    speakers: dict[str, list[AudioEntry]],
) -> dict[str, list[AudioEntry]]:
    return {spk: utts for spk, utts in speakers.items() if len(utts) >= 2}


def _filter_interferer_speakers(
    speakers: dict[str, list[AudioEntry]],
) -> dict[str, list[AudioEntry]]:
    return {spk: utts for spk, utts in speakers.items() if utts}


def _weighted_choice(
    rng: random.Random,
    weights: dict[str, float],
) -> str:
    """Draw one key from a positive-weight map."""
    total = 0.0
    for weight in weights.values():
        if weight < 0.0:
            raise ValueError(f"weights must be non-negative, got {weight}")
        total += weight
    if total <= 0.0:
        raise ValueError("weights must contain at least one positive entry")
    draw = rng.uniform(0.0, total)
    cumulative = 0.0
    last_key = None
    for key, weight in weights.items():
        if weight <= 0.0:
            continue
        cumulative += weight
        last_key = key
        if draw <= cumulative:
            return key
    assert last_key is not None
    return last_key


def _flatten_dataset_speakers(
    datasets: dict[str, dict[str, list[AudioEntry]]],
) -> dict[str, list[AudioEntry]]:
    merged: dict[str, list[AudioEntry]] = {}
    for speakers in datasets.values():
        for speaker_id, utterances in speakers.items():
            merged.setdefault(speaker_id, []).extend(utterances)
    return merged


def _normalize_weight_map(
    keys: list[str],
    weights: dict[str, float] | None,
    *,
    default_equal: bool = True,
) -> dict[str, float]:
    if not keys:
        return {}
    if weights is None:
        if default_equal:
            return {key: 1.0 for key in keys}
        raise ValueError("weights must be provided when default_equal=False")
    unknown_keys = sorted(set(weights) - set(keys))
    if unknown_keys:
        raise ValueError(f"Unknown weight keys: {', '.join(unknown_keys)}")
    normalized = {key: float(weights.get(key, 0.0)) for key in keys}
    if not any(weight > 0.0 for weight in normalized.values()):
        raise ValueError("weights must contain at least one positive entry")
    return normalized


def _subset_weights(
    weights: dict[str, float],
    allowed_keys: list[str],
) -> dict[str, float]:
    subset = {
        key: weights[key]
        for key in allowed_keys
        if key in weights and weights[key] > 0.0
    }
    if not subset and allowed_keys:
        subset = {key: 1.0 for key in allowed_keys}
    if not subset:
        raise ValueError("No positive weights available for the requested subset")
    return subset


def _normalize_buckets(
    buckets: tuple[tuple[float, float, float], ...] | None,
) -> tuple[tuple[float, float, float], ...] | None:
    if buckets is None:
        return None
    if not buckets:
        raise ValueError("buckets must not be empty")
    normalized: list[tuple[float, float, float]] = []
    for weight, min_db, max_db in buckets:
        item = (float(weight), float(min_db), float(max_db))
        if item[0] < 0.0:
            raise ValueError(f"bucket weight must be non-negative, got {item[0]}")
        if item[1] > item[2]:
            raise ValueError(f"bucket min_db must be <= max_db, got {item[1:3]}")
        normalized.append(item)
    if not any(weight > 0.0 for weight, _min_db, _max_db in normalized):
        raise ValueError("buckets must contain at least one positive-weight entry")
    return tuple(normalized)


def _sample_bucketed_value(
    rng: random.Random,
    buckets: tuple[tuple[float, float, float], ...],
) -> float:
    weights = {
        str(index): bucket[0]
        for index, bucket in enumerate(buckets)
        if bucket[0] > 0.0
    }
    choice = int(_weighted_choice(rng, weights))
    _weight, min_db, max_db = buckets[choice]
    return rng.uniform(min_db, max_db)


class PSEMixer(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]]):
    """Sample target/enrollment pairs with interference and additive noise."""

    def __init__(
        self,
        speakers: dict[str, list[AudioEntry]] | None = None,
        noises: list[NoiseEntry] | dict[str, list[NoiseEntry]] | None = None,
        *,
        datasets: dict[str, dict[str, list[AudioEntry]]] | None = None,
        interferer_speakers: dict[str, list[AudioEntry]] | None = None,
        interferer_datasets: dict[str, dict[str, list[AudioEntry]]] | None = None,
        dataset_weights: dict[str, float] | None = None,
        interferer_same_dataset_probability: float = 1.0,
        noise_category_weights: dict[str, float] | None = None,
        scene_weights: dict[str, float] | None = None,
        snr_buckets: tuple[tuple[float, float, float], ...] | None = None,
        sir_buckets: tuple[tuple[float, float, float], ...] | None = None,
        epoch_size: int = 1000,
        segment_length: int = 160000,
        enrollment_length: int = 240000,
        sample_rate: int = 16000,
        reverb_config: ReverbConfig | None = None,
        reverb_probability: float = 0.3,
        reverb_room_weights: dict[str, float] | None = None,
        gain_probability: float = 0.3,
        gain_range_db: tuple[float, float] = (-6.0, 6.0),
        bandwidth_limit_probability: float = 0.2,
        bandwidth_cutoff_range_hz: tuple[float, float] = (4000.0, 7000.0),
        mixing_rms_mode: Literal["full", "active"] = "full",
        activity_frame_ms: float = 32.0,
        activity_threshold_db: float = -40.0,
        audio_cache_size: int = 128,
        seed: int = 0,
    ) -> None:
        if datasets is None and speakers is None:
            raise ValueError("speakers must not be empty")
        self.target_datasets = self._normalize_target_datasets(
            speakers=speakers,
            datasets=datasets,
        )
        self.target_dataset_ids = sorted(self.target_datasets)
        if not self.target_dataset_ids:
            raise ValueError("at least one speaker with >=2 utterances is required")

        self.interferer_datasets = self._normalize_interferer_datasets(
            interferer_speakers=interferer_speakers,
            interferer_datasets=interferer_datasets,
            datasets=datasets,
        )
        self.interferer_dataset_ids = sorted(self.interferer_datasets)
        self.target_speakers = _flatten_dataset_speakers(self.target_datasets)
        self.target_speaker_ids = sorted(self.target_speakers)
        self.interferer_speakers = _flatten_dataset_speakers(self.interferer_datasets)
        self.interferer_speaker_ids = sorted(self.interferer_speakers)
        self.target_speaker_ids_by_dataset = {
            dataset_id: sorted(speakers)
            for dataset_id, speakers in self.target_datasets.items()
        }
        self.interferer_speaker_ids_by_dataset = {
            dataset_id: sorted(speakers)
            for dataset_id, speakers in self.interferer_datasets.items()
        }

        self.dataset_weights = _normalize_weight_map(
            self.target_dataset_ids,
            dataset_weights,
        )
        self.interferer_dataset_weights = {
            dataset_id: self.dataset_weights.get(dataset_id, 1.0)
            for dataset_id in self.interferer_dataset_ids
        }
        if not 0.0 <= interferer_same_dataset_probability <= 1.0:
            raise ValueError(
                "interferer_same_dataset_probability must be in [0, 1], got "
                f"{interferer_same_dataset_probability}"
            )
        self.interferer_same_dataset_probability = interferer_same_dataset_probability

        self.noise_pools = self._normalize_noise_pools(noises)
        self.noise_categories = sorted(self.noise_pools)
        self.noise_category_weights = (
            _normalize_weight_map(self.noise_categories, noise_category_weights)
            if self.noise_categories
            else {}
        )
        self.noises = [
            entry
            for category in self.noise_categories
            for entry in self.noise_pools[category]
        ]

        if scene_weights is None:
            self.scene_weights = dict(_DEFAULT_SCENE_WEIGHTS)
        else:
            self.scene_weights = self._normalize_scene_weights(scene_weights)
        self.snr_buckets = _normalize_buckets(snr_buckets)
        self.sir_buckets = _normalize_buckets(sir_buckets)
        self.epoch_size = epoch_size
        self.segment_length = segment_length
        self.enrollment_length = enrollment_length
        self.sample_rate = sample_rate
        self.reverb_config = reverb_config if reverb_config is not None else ReverbConfig(sample_rate=sample_rate)
        self.reverb_probability = reverb_probability
        self.reverb_configs_by_family = {
            family: ReverbConfig.from_preset(family, sample_rate=sample_rate)
            for family in _ROOM_FAMILY_NAMES
        }
        self.reverb_room_weights = (
            _normalize_weight_map(
                list(_ROOM_FAMILY_NAMES),
                reverb_room_weights,
            )
            if reverb_room_weights is not None
            else None
        )
        self.gain_probability = gain_probability
        self.gain_range_db = gain_range_db
        self.bandwidth_limit_probability = bandwidth_limit_probability
        self.bandwidth_cutoff_range_hz = bandwidth_cutoff_range_hz
        if mixing_rms_mode not in ("full", "active"):
            raise ValueError(f"Unsupported mixing_rms_mode: {mixing_rms_mode}")
        if activity_frame_ms <= 0.0:
            raise ValueError(f"activity_frame_ms must be positive, got {activity_frame_ms}")
        self.mixing_rms_mode = mixing_rms_mode
        self.activity_frame_ms = activity_frame_ms
        self.activity_frame_samples = max(1, int(round(activity_frame_ms * sample_rate / 1000.0)))
        self.activity_threshold_db = activity_threshold_db
        self.audio_cache_size = audio_cache_size
        self.seed = seed
        self._epoch = 0
        self._audio_cache: OrderedDict[Path, torch.Tensor] = OrderedDict()

    @staticmethod
    def _normalize_target_datasets(
        *,
        speakers: dict[str, list[AudioEntry]] | None,
        datasets: dict[str, dict[str, list[AudioEntry]]] | None,
    ) -> dict[str, dict[str, list[AudioEntry]]]:
        source = datasets if datasets is not None else {"default": speakers or {}}
        normalized: dict[str, dict[str, list[AudioEntry]]] = {}
        for dataset_id, pool in source.items():
            filtered = _filter_target_speakers(pool)
            if filtered:
                normalized[dataset_id] = filtered
        return normalized

    def _normalize_interferer_datasets(
        self,
        *,
        interferer_speakers: dict[str, list[AudioEntry]] | None,
        interferer_datasets: dict[str, dict[str, list[AudioEntry]]] | None,
        datasets: dict[str, dict[str, list[AudioEntry]]] | None,
    ) -> dict[str, dict[str, list[AudioEntry]]]:
        if interferer_datasets is not None:
            source = interferer_datasets
        elif interferer_speakers is not None:
            filtered = _filter_interferer_speakers(interferer_speakers)
            if datasets is None:
                source = {"default": filtered}
            else:
                source = {
                    dataset_id: filtered
                    for dataset_id in self.target_dataset_ids
                }
        else:
            source = self.target_datasets
        normalized: dict[str, dict[str, list[AudioEntry]]] = {}
        for dataset_id, pool in source.items():
            filtered = _filter_interferer_speakers(pool)
            if filtered:
                normalized[dataset_id] = filtered
        return normalized

    @staticmethod
    def _normalize_noise_pools(
        noises: list[NoiseEntry] | dict[str, list[NoiseEntry]] | None,
    ) -> dict[str, list[NoiseEntry]]:
        if noises is None:
            return {}
        if isinstance(noises, list):
            return {"default": list(noises)} if noises else {}
        return {
            category: list(entries)
            for category, entries in noises.items()
            if entries
        }

    @staticmethod
    def _normalize_scene_weights(
        scene_weights: dict[str, float],
    ) -> dict[str, float]:
        normalized = {
            "target_only_degraded": 0.0,
            "noise": 0.0,
            "interference": 0.0,
            "both": 0.0,
        }
        unknown_keys = sorted(set(scene_weights) - set(normalized))
        if unknown_keys:
            raise ValueError(f"Unknown scene weight keys: {', '.join(unknown_keys)}")
        for key, value in scene_weights.items():
            normalized[key] = float(value)
        if not any(weight > 0.0 for weight in normalized.values()):
            raise ValueError("scene_weights must contain at least one positive entry")
        return normalized

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
        if not self.noise_categories:
            return None
        category = _weighted_choice(rng, self.noise_category_weights)
        entry = rng.choice(self.noise_pools[category])
        audio = self._load_audio(entry.path)
        if audio.numel() >= length:
            max_start = audio.numel() - length
            start = rng.randint(0, max_start) if max_start > 0 else 0
            return audio[start : start + length].clone()
        reps = (length + audio.numel() - 1) // max(audio.numel(), 1)
        return audio.repeat(reps)[:length].clone()

    def _sample_target_dataset(self, rng: random.Random) -> str:
        return _weighted_choice(rng, self.dataset_weights)

    def _sample_target_pair(
        self,
        rng: random.Random,
    ) -> tuple[str, str, AudioEntry, AudioEntry]:
        dataset_id = self._sample_target_dataset(rng)
        speaker_id = rng.choice(self.target_speaker_ids_by_dataset[dataset_id])
        utts = rng.sample(self.target_datasets[dataset_id][speaker_id], 2)
        return dataset_id, speaker_id, utts[0], utts[1]

    def _sample_interferer(
        self,
        target_dataset: str,
        target_speaker: str,
        rng: random.Random,
    ) -> tuple[str, str, AudioEntry] | None:
        if not self.interferer_dataset_ids:
            return None
        same_dataset_ids = [target_dataset] if target_dataset in self.interferer_dataset_ids else []
        cross_dataset_ids = [dataset_id for dataset_id in self.interferer_dataset_ids if dataset_id != target_dataset]

        if same_dataset_ids and (
            not cross_dataset_ids
            or rng.random() < self.interferer_same_dataset_probability
        ):
            candidate_dataset_ids = same_dataset_ids
        else:
            candidate_dataset_ids = cross_dataset_ids or same_dataset_ids

        available_dataset_ids: list[str] = []
        for dataset_id in candidate_dataset_ids:
            candidates = [
                speaker_id
                for speaker_id in self.interferer_speaker_ids_by_dataset[dataset_id]
                if not (dataset_id == target_dataset and speaker_id == target_speaker)
            ]
            if candidates:
                available_dataset_ids.append(dataset_id)
        if not available_dataset_ids:
            for dataset_id in self.interferer_dataset_ids:
                candidates = [
                    speaker_id
                    for speaker_id in self.interferer_speaker_ids_by_dataset[dataset_id]
                    if not (dataset_id == target_dataset and speaker_id == target_speaker)
                ]
                if candidates:
                    available_dataset_ids.append(dataset_id)
        if not available_dataset_ids:
            return None
        dataset_id = _weighted_choice(
            rng,
            _subset_weights(self.interferer_dataset_weights, available_dataset_ids),
        )
        candidates = [
            speaker_id
            for speaker_id in self.interferer_speaker_ids_by_dataset[dataset_id]
            if not (dataset_id == target_dataset and speaker_id == target_speaker)
        ]
        if not candidates:
            return None
        speaker_id = rng.choice(candidates)
        entry = rng.choice(self.interferer_datasets[dataset_id][speaker_id])
        return dataset_id, speaker_id, entry

    def _sample_scene_reverb_config(
        self,
        rng: random.Random,
    ) -> ReverbConfig | None:
        if rng.random() >= self.reverb_probability:
            return None
        if self.reverb_room_weights is None:
            return self.reverb_config
        family = _weighted_choice(rng, self.reverb_room_weights)
        return self.reverb_configs_by_family[family]

    def _maybe_reverb(
        self,
        signal: torch.Tensor,
        rng: random.Random,
        cfg: ReverbConfig | None = None,
    ) -> torch.Tensor:
        effective_cfg = cfg if cfg is not None else self._sample_scene_reverb_config(rng)
        if effective_cfg is None:
            return signal
        rir = synth_room_rir(effective_cfg, rng)
        return apply_rir(signal, rir)

    def _draw_scene_components(self, rng: random.Random) -> tuple[bool, bool]:
        scene = _weighted_choice(rng, self.scene_weights)
        if scene == "target_only_degraded":
            return False, False
        if scene == "noise":
            return True, False
        if scene == "interference":
            return False, True
        return True, True

    def _sample_snr_db(self, rng: random.Random) -> float:
        if self.snr_buckets is None:
            return max(-5.0, min(35.0, rng.gauss(15.0, 10.0)))
        return _sample_bucketed_value(rng, self.snr_buckets)

    def _sample_sir_db(self, rng: random.Random) -> float:
        if self.sir_buckets is None:
            return max(-5.0, min(25.0, rng.gauss(10.0, 7.5)))
        return _sample_bucketed_value(rng, self.sir_buckets)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
        rng = self._rng(index)
        target_dataset_id, speaker_id, target_entry, enroll_entry = self._sample_target_pair(rng)
        target_speakers = self.target_datasets[target_dataset_id][speaker_id]
        # Reserve enrollment anchor so target sampling never consumes it.
        # This guarantees enrollment always has at least one disjoint file.
        target, target_paths = self._sample_speaker_segment_tracked(
            target_speakers,
            self.segment_length,
            rng,
            anchor_entry=target_entry,
            avoid_paths={enroll_entry.path},
        )
        enrollment = self._sample_speaker_segment(
            target_speakers,
            self.enrollment_length,
            rng,
            anchor_entry=enroll_entry,
            avoid_paths=target_paths,
        )
        scene_reverb_config = self._sample_scene_reverb_config(rng)
        target = self._maybe_reverb(target, rng, scene_reverb_config)
        mixture = target.clone()
        include_noise, include_interference = self._draw_scene_components(rng)

        if include_interference:
            interferer_sample = self._sample_interferer(target_dataset_id, speaker_id, rng)
            if interferer_sample is not None:
                interferer_dataset_id, interferer_speaker_id, interferer_entry = interferer_sample
                interferer = self._sample_speaker_segment(
                    self.interferer_datasets[interferer_dataset_id][interferer_speaker_id],
                    self.segment_length,
                    rng,
                    anchor_entry=interferer_entry,
                )
                interferer = self._maybe_reverb(interferer, rng, scene_reverb_config)
                sir_db = self._sample_sir_db(rng)
                mixture = mixture + scale_noise_to_snr(
                    target,
                    interferer,
                    sir_db,
                    rng=rng,
                    rms_mode=self.mixing_rms_mode,
                    activity_frame_samples=self.activity_frame_samples,
                    activity_threshold_db=self.activity_threshold_db,
                )

        if include_noise:
            noise = self._sample_noise(self.segment_length, rng)
            if noise is not None:
                snr_db = self._sample_snr_db(rng)
                mixture = mixture + scale_noise_to_snr(
                    target,
                    noise,
                    snr_db,
                    rng=rng,
                    rms_mode=self.mixing_rms_mode,
                    activity_frame_samples=self.activity_frame_samples,
                    activity_threshold_db=self.activity_threshold_db,
                )

        if rng.random() < self.gain_probability:
            mixture = apply_random_gain(
                mixture,
                gain_range_db=self.gain_range_db,
                rng=rng,
            )
        if rng.random() < self.bandwidth_limit_probability:
            mixture = apply_bandwidth_limit(
                mixture,
                sample_rate=self.sample_rate,
                cutoff_range_hz=self.bandwidth_cutoff_range_hz,
                rng=rng,
            )

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


__all__ = ["PSEMixer", "_weighted_choice"]
