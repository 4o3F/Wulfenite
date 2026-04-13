"""Unified long-scene composer for routing-aware TSE training.

The clip composer now generates one coherent long scene that contains
all routing-critical patterns by construction:

1. target only
2. other only
3. target return after absence
4. overlap
5. other only after overlap
6. background only
7. target return

The renderer preserves the clean per-role source tracks so the mixer
can derive multiple enrollment-conditioned training views from the same
mixture.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Sequence

import torch
import torch.nn.functional as F

from .aishell import AudioEntry
from .augmentation import (
    add_gaussian_noise,
    add_noise_at_snr,
    apply_rir,
    synth_room_rir,
)
from .noise import NoiseEntry

if TYPE_CHECKING:
    from .mixer import MixerConfig

from .mixer import (
    _load_chunk,
    _load_noise_chunk,
    _rms,
)


__all__ = [
    "ClipFamily",
    "EventType",
    "EventSlot",
    "ClipPlan",
    "ComposerConfig",
    "SceneBundle",
    "SceneView",
    "SpeakerCast",
    "ClipPlanner",
    "ClipRenderer",
]


class ClipFamily(str, Enum):
    """Scene family emitted by the unified long-scene planner."""

    UNIFIED_LONG_SCENE = "unified_long_scene"


class EventType(str, Enum):
    """Conversation event types on the frame grid."""

    TARGET_ONLY = "target_only"
    NONTARGET_ONLY = "nontarget_only"
    OVERLAP = "overlap"
    BACKGROUND_ONLY = "background_only"


@dataclass(frozen=True)
class EventSlot:
    """One contiguous scene segment on the frame timeline."""

    index: int
    event_type: EventType
    start_frame: int
    end_frame: int
    active_roles: tuple[str, ...]
    anchor_name: str | None = None

    @property
    def duration_frames(self) -> int:
        return self.end_frame - self.start_frame


@dataclass(frozen=True)
class ClipPlan:
    """Unified long-scene plan consumed by :class:`ClipRenderer`."""

    family: ClipFamily
    slots: tuple[EventSlot, ...]
    total_frames: int
    stride_samples: int
    segment_samples: int
    use_third_speaker: bool
    snr_db: float
    noise_snr_db: float
    active_frames_by_role: dict[str, torch.Tensor]
    overlap_frames: torch.Tensor
    background_frames: torch.Tensor


@dataclass
class ComposerConfig:
    """Configuration bundle for the unified long-scene composer."""

    sample_rate: int = 16000
    segment_seconds: float = 8.0
    stride_samples: int = 160
    label_hop_samples: int = 160

    target_only_min_frames: int = 80
    nontarget_only_min_frames: int = 80
    overlap_min_frames: int = 40
    background_min_frames: int = 30
    absence_before_return_min_frames: int = 100
    crossfade_samples: int = 80

    optional_third_speaker_prob: float = 0.35
    gain_drift_db_range: tuple[float, float] = (-1.5, 1.5)
    snr_range_db: tuple[float, float] = (-5.0, 5.0)
    noise_snr_range_db: tuple[float, float] = (10.0, 25.0)

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError(f"sample_rate must be positive; got {self.sample_rate}")
        if self.segment_seconds <= 0.0:
            raise ValueError(
                f"segment_seconds must be positive; got {self.segment_seconds}"
            )
        self.segment_samples = int(self.segment_seconds * self.sample_rate)
        if self.segment_samples <= 0:
            raise ValueError(
                f"segment_seconds {self.segment_seconds} gives empty segment"
            )
        if self.segment_samples % self.stride_samples != 0:
            raise ValueError(
                "segment_seconds must produce a sample count divisible by stride; "
                f"got {self.segment_samples} for stride {self.stride_samples}"
            )
        self.total_frames = self.segment_samples // self.stride_samples
        if self.label_hop_samples != self.stride_samples:
            raise ValueError(
                "label_hop_samples must match stride_samples; got "
                f"{self.label_hop_samples} vs {self.stride_samples}"
            )
        for name, value in (
            ("target_only_min_frames", self.target_only_min_frames),
            ("nontarget_only_min_frames", self.nontarget_only_min_frames),
            ("overlap_min_frames", self.overlap_min_frames),
            ("background_min_frames", self.background_min_frames),
            (
                "absence_before_return_min_frames",
                self.absence_before_return_min_frames,
            ),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive; got {value}")
        if self.crossfade_samples < 0:
            raise ValueError(
                f"crossfade_samples must be non-negative; got {self.crossfade_samples}"
            )
        if self.crossfade_samples >= self.stride_samples:
            raise ValueError(
                "crossfade_samples must be smaller than stride_samples; got "
                f"{self.crossfade_samples} vs {self.stride_samples}"
            )
        min_required_frames = (
            self.target_only_min_frames * 3
            + max(
                self.nontarget_only_min_frames,
                self.absence_before_return_min_frames,
            )
            + self.nontarget_only_min_frames
            + self.overlap_min_frames
            + self.background_min_frames
        )
        if min_required_frames > self.total_frames:
            raise ValueError(
                "scene minimum exceeds clip length; got "
                f"{min_required_frames} > {self.total_frames}"
            )


@dataclass(frozen=True)
class SpeakerCast:
    """Concrete speaker assignment for one rendered scene."""

    source_speaker_ids: dict[str, str]
    source_entries: dict[str, AudioEntry]
    enrollment_entries: dict[str, AudioEntry]
    outsider_speaker_id: str
    outsider_enrollment_entry: AudioEntry


@dataclass(frozen=True)
class SceneBundle:
    """Rendered scene with preserved per-role source tracks."""

    scene_id: int
    mixture: torch.Tensor
    source_tracks: dict[str, torch.Tensor]
    enrollment_pool: dict[str, tuple[torch.Tensor, ...]]
    active_frames_by_role: dict[str, torch.Tensor]
    overlap_frames: torch.Tensor
    background_frames: torch.Tensor
    metadata: dict


@dataclass(frozen=True)
class SceneView:
    """One enrollment-conditioned supervised view derived from a scene."""

    scene_id: int
    view_role: str
    enrollment: torch.Tensor
    target: torch.Tensor
    target_present: float
    target_speaker_idx: int
    target_active_frames: torch.Tensor
    nontarget_active_frames: torch.Tensor
    overlap_frames: torch.Tensor
    background_frames: torch.Tensor
    snr_db: float

    def to_sample(self) -> dict:
        return {
            "scene_id": torch.tensor(self.scene_id, dtype=torch.long),
            "view_role": self.view_role,
            "mixture": None,
            "target": self.target,
            "enrollment": self.enrollment,
            "target_present": torch.tensor(
                self.target_present, dtype=torch.float32,
            ),
            "target_speaker_idx": torch.tensor(
                self.target_speaker_idx, dtype=torch.long,
            ),
            "snr_db": torch.tensor(self.snr_db, dtype=torch.float32),
            "target_active_frames": self.target_active_frames.clone(),
            "nontarget_active_frames": self.nontarget_active_frames.clone(),
            "overlap_frames": self.overlap_frames.clone(),
            "background_frames": self.background_frames.clone(),
        }


@dataclass(frozen=True)
class _AnchorSpec:
    event_type: EventType
    min_frames: int
    active_roles: tuple[str, ...]
    anchor_name: str


class ClipPlanner:
    """Generate one coherent long scene with mandatory routing patterns."""

    def __init__(
        self,
        cfg: ComposerConfig,
        *,
        allow_third_speaker: bool = True,
    ) -> None:
        self.cfg = cfg
        self.allow_third_speaker = allow_third_speaker

    def plan(
        self,
        rng: random.Random | None = None,
    ) -> ClipPlan:
        rng = rng or random.Random()
        use_third_speaker = (
            self.allow_third_speaker
            and rng.random() < self.cfg.optional_third_speaker_prob
        )
        slot_specs = self._build_slot_specs(use_third_speaker)
        durations = self._allocate_frames(slot_specs, rng)
        slots = self._materialize_slots(slot_specs, durations)
        active_frames_by_role = self._labels_from_slots(slots)
        overlap_frames = self._compute_overlap(active_frames_by_role)
        background_frames = self._compute_background(
            active_frames_by_role, self.cfg.total_frames,
        )
        self._assert_constraints(
            slots=slots,
            active_frames_by_role=active_frames_by_role,
            overlap_frames=overlap_frames,
            background_frames=background_frames,
        )
        return ClipPlan(
            family=ClipFamily.UNIFIED_LONG_SCENE,
            slots=tuple(slots),
            total_frames=self.cfg.total_frames,
            stride_samples=self.cfg.stride_samples,
            segment_samples=self.cfg.segment_samples,
            use_third_speaker=use_third_speaker,
            snr_db=rng.uniform(*self.cfg.snr_range_db),
            noise_snr_db=rng.uniform(*self.cfg.noise_snr_range_db),
            active_frames_by_role=active_frames_by_role,
            overlap_frames=overlap_frames,
            background_frames=background_frames,
        )

    def _build_slot_specs(
        self,
        use_third_speaker: bool,
    ) -> tuple[_AnchorSpec, ...]:
        post_overlap_role = "C" if use_third_speaker else "B"
        return (
            _AnchorSpec(
                EventType.TARGET_ONLY,
                self.cfg.target_only_min_frames,
                ("A",),
                "target_only_intro",
            ),
            _AnchorSpec(
                EventType.NONTARGET_ONLY,
                max(
                    self.cfg.nontarget_only_min_frames,
                    self.cfg.absence_before_return_min_frames,
                ),
                ("B",),
                "other_only_intro",
            ),
            _AnchorSpec(
                EventType.TARGET_ONLY,
                self.cfg.target_only_min_frames,
                ("A",),
                "target_return_after_absence",
            ),
            _AnchorSpec(
                EventType.OVERLAP,
                self.cfg.overlap_min_frames,
                ("A", "B"),
                "overlap_main",
            ),
            _AnchorSpec(
                EventType.NONTARGET_ONLY,
                self.cfg.nontarget_only_min_frames,
                (post_overlap_role,),
                "other_only_after_overlap",
            ),
            _AnchorSpec(
                EventType.BACKGROUND_ONLY,
                self.cfg.background_min_frames,
                (),
                "background_only_reset",
            ),
            _AnchorSpec(
                EventType.TARGET_ONLY,
                self.cfg.target_only_min_frames,
                ("A",),
                "target_return_final",
            ),
        )

    def _allocate_frames(
        self,
        specs: Sequence[_AnchorSpec],
        rng: random.Random,
    ) -> list[int]:
        durations = [spec.min_frames for spec in specs]
        remaining = self.cfg.total_frames - sum(durations)
        while remaining > 0:
            idx = rng.randrange(len(durations))
            durations[idx] += 1
            remaining -= 1
        return durations

    def _materialize_slots(
        self,
        specs: Sequence[_AnchorSpec],
        durations: Sequence[int],
    ) -> list[EventSlot]:
        slots: list[EventSlot] = []
        cursor = 0
        for index, (spec, duration) in enumerate(zip(specs, durations)):
            slots.append(
                EventSlot(
                    index=index,
                    event_type=spec.event_type,
                    start_frame=cursor,
                    end_frame=cursor + duration,
                    active_roles=spec.active_roles,
                    anchor_name=spec.anchor_name,
                )
            )
            cursor += duration
        return slots

    def _labels_from_slots(
        self,
        slots: Sequence[EventSlot],
    ) -> dict[str, torch.Tensor]:
        roles = {"A", "B"}
        if any("C" in slot.active_roles for slot in slots):
            roles.add("C")
        labels = {
            role: torch.zeros(self.cfg.total_frames, dtype=torch.bool)
            for role in roles
        }
        for slot in slots:
            sl = slice(slot.start_frame, slot.end_frame)
            for role in slot.active_roles:
                labels[role][sl] = True
        return labels

    def _compute_overlap(
        self,
        active_frames_by_role: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        stacked = torch.stack(
            [mask.to(torch.int32) for mask in active_frames_by_role.values()], dim=0
        )
        return stacked.sum(dim=0) >= 2

    def _compute_background(
        self,
        active_frames_by_role: dict[str, torch.Tensor],
        total_frames: int,
    ) -> torch.Tensor:
        background = torch.ones(total_frames, dtype=torch.bool)
        for mask in active_frames_by_role.values():
            background = background & ~mask.bool()
        return background

    def _assert_constraints(
        self,
        *,
        slots: Sequence[EventSlot],
        active_frames_by_role: dict[str, torch.Tensor],
        overlap_frames: torch.Tensor,
        background_frames: torch.Tensor,
    ) -> None:
        if not slots:
            raise AssertionError("planner produced no slots")
        if slots[0].start_frame != 0 or slots[-1].end_frame != self.cfg.total_frames:
            raise AssertionError("planner must cover the full scene")
        if not active_frames_by_role["A"].any():
            raise AssertionError("target role A must be active")
        if not active_frames_by_role["B"].any():
            raise AssertionError("speaker B must be active for paired routing")
        if not overlap_frames.any():
            raise AssertionError("planner must contain overlap")
        if not background_frames.any():
            raise AssertionError("planner must contain background-only frames")

        target_only = active_frames_by_role["A"].clone()
        for role, active in active_frames_by_role.items():
            if role != "A":
                target_only = target_only & ~active
        if int(target_only.sum().item()) < self.cfg.target_only_min_frames:
            raise AssertionError("target-only minimum not satisfied")

        nontarget_union = torch.zeros_like(background_frames)
        for role, active in active_frames_by_role.items():
            if role != "A":
                nontarget_union = nontarget_union | active
        nontarget_only = nontarget_union & ~active_frames_by_role["A"]
        if int(nontarget_only.sum().item()) < self.cfg.nontarget_only_min_frames:
            raise AssertionError("nontarget-only minimum not satisfied")
        if int(overlap_frames.sum().item()) < self.cfg.overlap_min_frames:
            raise AssertionError("overlap minimum not satisfied")
        if int(background_frames.sum().item()) < self.cfg.background_min_frames:
            raise AssertionError("background-only minimum not satisfied")

        intro_end = slots[1].end_frame
        return_start = slots[2].start_frame
        if return_start - slots[0].end_frame < self.cfg.absence_before_return_min_frames:
            raise AssertionError("target return after absence minimum not satisfied")
        transitions = [
            (slots[i].event_type, slots[i + 1].event_type)
            for i in range(len(slots) - 1)
        ]
        required = {
            (EventType.TARGET_ONLY, EventType.NONTARGET_ONLY),
            (EventType.NONTARGET_ONLY, EventType.TARGET_ONLY),
            (EventType.TARGET_ONLY, EventType.OVERLAP),
        }
        if not required.issubset(set(transitions)):
            raise AssertionError(f"missing required transitions: {transitions}")
        if (
            (EventType.NONTARGET_ONLY, EventType.OVERLAP) not in transitions
            and (EventType.OVERLAP, EventType.NONTARGET_ONLY) not in transitions
        ):
            raise AssertionError("scene must contain other_only <-> overlap transition")
        del intro_end


class ClipRenderer:
    """Render unified long scenes with preserved role tracks."""

    _enrollment_modes = ("dense", "random", "silence_heavy", "noisy", "reverbed")

    def __init__(
        self,
        mixer_cfg: MixerConfig,
        *,
        noise_pool: Sequence[NoiseEntry] | None = None,
        rir_pool: Sequence[torch.Tensor] | None = None,
    ) -> None:
        self.cfg = mixer_cfg
        self.composer = mixer_cfg.composer
        self._enrollment_len = int(
            mixer_cfg.enrollment_seconds * mixer_cfg.sample_rate
        )
        self.noise_pool = list(noise_pool) if noise_pool else []
        self._has_noise_pool = bool(self.noise_pool) and mixer_cfg.use_noise_pool
        self._rir_pool = list(rir_pool) if rir_pool else []

    def render_bundle(
        self,
        scene_id: int,
        plan: ClipPlan,
        cast: SpeakerCast,
        speaker_to_idx: dict[str, int],
        rng: random.Random,
    ) -> SceneBundle:
        role_stems = self._load_role_stems(plan, cast, rng)
        role_tracks = self._stitch_role_tracks(plan, role_stems)
        role_envelopes = self._build_role_envelopes(plan)
        role_tracks = {
            role: track * role_envelopes.get(role, 1.0)
            for role, track in role_tracks.items()
        }
        role_tracks = {
            role: self._normalize(track) for role, track in role_tracks.items()
        }
        role_tracks = self._apply_role_augmentation(role_tracks, rng)
        role_tracks = self._match_scene_snr(plan, role_tracks)
        clean_tracks = {role: track.clone() for role, track in role_tracks.items()}

        mixture = torch.zeros(plan.segment_samples, dtype=torch.float32)
        for track in clean_tracks.values():
            mixture = mixture + track
        mixture = self._maybe_add_noise(mixture, plan.noise_snr_db, rng)

        peak = float(mixture.abs().max())
        if peak > self.cfg.peak_clip:
            scale = self.cfg.peak_clip / peak
            mixture = mixture * scale
            clean_tracks = {
                role: track * scale for role, track in clean_tracks.items()
            }

        enrollment_pool = self._build_enrollment_pool(cast, rng)
        metadata = {
            "family": plan.family.value,
            "snr_db": float(plan.snr_db),
            "source_speaker_ids": dict(cast.source_speaker_ids),
            "outsider_speaker_id": cast.outsider_speaker_id,
            "speaker_indices": {
                role: speaker_to_idx[speaker_id]
                for role, speaker_id in cast.source_speaker_ids.items()
            },
            "outsider_speaker_idx": speaker_to_idx[cast.outsider_speaker_id],
            "enrollment_modes": self._enrollment_modes,
        }
        return SceneBundle(
            scene_id=scene_id,
            mixture=mixture,
            source_tracks=clean_tracks,
            enrollment_pool=enrollment_pool,
            active_frames_by_role={
                role: mask.clone() for role, mask in plan.active_frames_by_role.items()
            },
            overlap_frames=plan.overlap_frames.clone(),
            background_frames=plan.background_frames.clone(),
            metadata=metadata,
        )

    def _load_role_stems(
        self,
        plan: ClipPlan,
        cast: SpeakerCast,
        rng: random.Random,
    ) -> dict[str, torch.Tensor]:
        return {
            role: _load_chunk(entry, plan.segment_samples, rng).to(torch.float32)
            for role, entry in cast.source_entries.items()
        }

    def _build_role_envelopes(
        self,
        plan: ClipPlan,
    ) -> dict[str, torch.Tensor]:
        roles = set(plan.active_frames_by_role.keys())
        envelopes = {
            role: torch.zeros(plan.segment_samples, dtype=torch.float32)
            for role in roles
        }
        for slot in plan.slots:
            start = slot.start_frame * plan.stride_samples
            end = slot.end_frame * plan.stride_samples
            for role in slot.active_roles:
                envelopes[role][start:end] = 1.0

        fade = self.composer.crossfade_samples
        if fade <= 0:
            return envelopes

        fade_in = 0.5 - 0.5 * torch.cos(torch.linspace(0.0, math.pi, steps=fade))
        fade_out = torch.flip(fade_in, dims=[0])
        for i in range(len(plan.slots) - 1):
            prev_slot = plan.slots[i]
            next_slot = plan.slots[i + 1]
            boundary = prev_slot.end_frame * plan.stride_samples
            for role in roles:
                prev_active = role in prev_slot.active_roles
                next_active = role in next_slot.active_roles
                if prev_active and not next_active:
                    length = min(
                        fade,
                        boundary - prev_slot.start_frame * plan.stride_samples,
                    )
                    if length > 0:
                        envelopes[role][boundary - length:boundary] *= fade_out[-length:]
                elif next_active and not prev_active:
                    length = min(
                        fade,
                        next_slot.end_frame * plan.stride_samples - boundary,
                    )
                    if length > 0:
                        envelopes[role][boundary:boundary + length] *= fade_in[:length]
        return envelopes

    def _stitch_role_tracks(
        self,
        plan: ClipPlan,
        stems: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        tracks = {
            role: torch.zeros(plan.segment_samples, dtype=stem.dtype)
            for role, stem in stems.items()
        }
        cursors = {role: 0 for role in stems}
        for slot in plan.slots:
            start = slot.start_frame * plan.stride_samples
            end = slot.end_frame * plan.stride_samples
            length = end - start
            for role in slot.active_roles:
                stem = stems[role]
                cursor = cursors[role]
                segment = stem[cursor:cursor + length]
                if segment.numel() < length:
                    segment = F.pad(segment, (0, length - segment.numel()))
                tracks[role][start:end] = segment
                cursors[role] += length
        return tracks

    def _apply_role_augmentation(
        self,
        role_tracks: dict[str, torch.Tensor],
        rng: random.Random,
    ) -> dict[str, torch.Tensor]:
        tracks = {
            role: self._normalize(track.clone()) for role, track in role_tracks.items()
        }
        if self.cfg.apply_reverb and rng.random() < self.cfg.reverb_prob:
            tracks = {
                role: self._normalize(apply_rir(track, self._sample_rir(rng)))
                for role, track in tracks.items()
            }
        return {
            role: track * self._sample_gain_drift(track.numel(), rng)
            for role, track in tracks.items()
        }

    def _match_scene_snr(
        self,
        plan: ClipPlan,
        role_tracks: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        if "A" not in role_tracks or "B" not in role_tracks:
            return role_tracks
        nontarget = torch.zeros_like(role_tracks["A"])
        for role, track in role_tracks.items():
            if role != "A":
                nontarget = nontarget + track
        overlap_mask = plan.overlap_frames.repeat_interleave(plan.stride_samples)[
            :plan.segment_samples
        ]
        if bool(overlap_mask.any().item()):
            target_overlap = role_tracks["A"][overlap_mask]
            nontarget_overlap = nontarget[overlap_mask]
            tgt_rms = float(torch.sqrt(torch.mean(target_overlap * target_overlap) + 1e-8))
            nt_rms = float(
                torch.sqrt(torch.mean(nontarget_overlap * nontarget_overlap) + 1e-8)
            )
            if nt_rms > 1e-8:
                scale = tgt_rms / (nt_rms * (10.0 ** (plan.snr_db / 20.0)) + 1e-8)
                for role in tuple(role_tracks.keys()):
                    if role != "A":
                        role_tracks[role] = role_tracks[role] * scale
        return role_tracks

    def _build_enrollment_pool(
        self,
        cast: SpeakerCast,
        rng: random.Random,
    ) -> dict[str, tuple[torch.Tensor, ...]]:
        pool: dict[str, tuple[torch.Tensor, ...]] = {}
        entries = {
            **cast.enrollment_entries,
            "OUTSIDER": cast.outsider_enrollment_entry,
        }
        for role, entry in entries.items():
            pool[role] = tuple(
                self._build_enrollment_candidate(entry, mode, rng)
                for mode in self._enrollment_modes
            )
        return pool

    def _build_enrollment_candidate(
        self,
        entry: AudioEntry,
        mode: str,
        rng: random.Random,
    ) -> torch.Tensor:
        if mode == "dense":
            audio = self._load_ranked_enrollment(entry, rng, pick="loudest")
        elif mode == "silence_heavy":
            audio = self._load_ranked_enrollment(entry, rng, pick="quietest")
        else:
            audio = _load_chunk(entry, self._enrollment_len, rng)
        audio = self._normalize(audio.to(torch.float32))

        if mode == "noisy":
            if self._has_noise_pool:
                noise_entry = rng.choice(self.noise_pool)
                noise = _load_noise_chunk(noise_entry, audio.numel(), rng)
                audio = add_noise_at_snr(audio, noise, rng.uniform(6.0, 18.0))
            else:
                audio = add_gaussian_noise(audio, rng.uniform(6.0, 18.0))
            audio = self._normalize(audio)
        elif mode == "reverbed":
            audio = self._normalize(apply_rir(audio, self._sample_rir(rng)))
        return audio

    def _load_ranked_enrollment(
        self,
        entry: AudioEntry,
        rng: random.Random,
        *,
        pick: str,
        n_trials: int = 8,
    ) -> torch.Tensor:
        candidates = [
            _load_chunk(entry, self._enrollment_len, rng).to(torch.float32)
            for _ in range(max(1, n_trials))
        ]
        key = max if pick == "loudest" else min
        return key(candidates, key=lambda chunk: float(_rms(chunk)))

    def _sample_rir(self, rng: random.Random) -> torch.Tensor:
        if self._rir_pool:
            return rng.choice(self._rir_pool)
        return synth_room_rir(self.cfg.reverb, rng)

    def _maybe_add_noise(
        self,
        mixture: torch.Tensor,
        noise_snr_db: float,
        rng: random.Random,
    ) -> torch.Tensor:
        if not self.cfg.apply_noise or rng.random() >= self.cfg.noise_prob:
            return mixture
        if self._has_noise_pool:
            noise_entry = rng.choice(self.noise_pool)
            noise = _load_noise_chunk(noise_entry, mixture.shape[-1], rng)
            return add_noise_at_snr(mixture, noise, noise_snr_db)
        return add_gaussian_noise(mixture, noise_snr_db)

    def _sample_gain_drift(
        self,
        n_samples: int,
        rng: random.Random,
    ) -> torch.Tensor:
        lo_db, hi_db = self.composer.gain_drift_db_range
        if lo_db == 0.0 and hi_db == 0.0:
            return torch.ones(n_samples, dtype=torch.float32)
        amplitude_db = rng.uniform(lo_db, hi_db)
        phase = rng.uniform(0.0, 2.0 * math.pi)
        cycles = rng.uniform(0.5, 1.5)
        t = torch.linspace(0.0, 1.0, steps=n_samples)
        drift_db = amplitude_db * torch.sin(2.0 * math.pi * cycles * t + phase)
        return torch.pow(10.0, drift_db / 20.0)

    def _normalize(self, signal: torch.Tensor) -> torch.Tensor:
        return signal / (_rms(signal) + 1e-8) * self.cfg.rms_target
