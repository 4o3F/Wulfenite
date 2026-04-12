"""Clip composer for conversational TSE training samples.

The legacy mixer uses a small set of branch-specific recipes. This
module replaces those branches with a two-stage pipeline:

1. :class:`ClipPlanner` builds a frame-grid conversation plan.
2. :class:`ClipRenderer` renders the plan into waveforms + labels.

The plan is deterministic once a family and RNG seed are fixed. All
constraints are satisfied by construction; there is no rejection loop.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Sequence, TypeVar

import torch
import torch.nn.functional as F

from .aishell import AudioEntry
from .augmentation import (
    ReverbConfig,
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
    _rescale_to_snr,
    _rms,
)


__all__ = [
    "ClipFamily",
    "EventType",
    "EventSlot",
    "ClipPlan",
    "ComposerConfig",
    "SpeakerCast",
    "ClipPlanner",
    "ClipRenderer",
]

_ChoiceT = TypeVar("_ChoiceT")


class ClipFamily(str, Enum):
    """High-level clip families emitted by the planner."""

    MULTI_TURN_TARGET_PRESENT = "multi_turn_target_present"
    OVERLAP_HEAVY = "overlap_heavy"
    HARD_NEGATIVE_ABSENT = "hard_negative_absent"


class EventType(str, Enum):
    """Local conversation event type on the encoder frame grid."""

    TARGET_ONLY = "target_only"
    NONTARGET_ONLY = "nontarget_only"
    OVERLAP = "overlap"
    BACKGROUND_ONLY = "background_only"


@dataclass(frozen=True)
class EventSlot:
    """One contiguous conversational event on the frame timeline."""

    index: int
    event_type: EventType
    start_frame: int
    end_frame: int
    active_roles: tuple[str, ...]
    anchor_name: str | None = None
    nontarget_role: str | None = None

    @property
    def duration_frames(self) -> int:
        return self.end_frame - self.start_frame


@dataclass(frozen=True)
class ClipPlan:
    """Planner output consumed by :class:`ClipRenderer`."""

    family: ClipFamily
    slots: tuple[EventSlot, ...]
    total_frames: int
    stride_samples: int
    segment_samples: int
    use_third_speaker: bool
    target_present: bool
    snr_db: float
    noise_snr_db: float
    target_active_frames: torch.Tensor
    nontarget_active_frames: torch.Tensor
    overlap_frames: torch.Tensor


@dataclass
class ComposerConfig:
    """Configuration bundle for the clip composer."""

    sample_rate: int = 16000
    segment_seconds: float = 4.0
    stride_samples: int = 160
    label_hop_samples: int = 160

    family_weights: dict[ClipFamily, float] = field(
        default_factory=lambda: {
            ClipFamily.MULTI_TURN_TARGET_PRESENT: 0.60,
            ClipFamily.OVERLAP_HEAVY: 0.25,
            ClipFamily.HARD_NEGATIVE_ABSENT: 0.15,
        }
    )
    min_events: int = 4
    max_events: int = 8
    min_event_frames: int = 30
    max_event_frames: int = 120
    crossfade_samples: int = 80

    min_target_only_frames: int = 50
    min_nontarget_only_frames: int = 50
    min_overlap_frames: int = 30
    min_absence_before_return_frames: int = 60

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
                f"got {self.segment_samples} samples for stride {self.stride_samples}"
            )
        self.total_frames = self.segment_samples // self.stride_samples
        if self.label_hop_samples != self.stride_samples:
            raise ValueError(
                "label_hop_samples must match stride_samples; got "
                f"{self.label_hop_samples} vs {self.stride_samples}"
            )
        if self.min_events <= 0 or self.max_events < self.min_events:
            raise ValueError(
                f"event bounds must satisfy 0 < min <= max; got "
                f"{self.min_events}, {self.max_events}"
            )
        if self.min_event_frames <= 0 or self.max_event_frames < self.min_event_frames:
            raise ValueError(
                f"event frame bounds must satisfy 0 < min <= max; got "
                f"{self.min_event_frames}, {self.max_event_frames}"
            )
        if self.min_events * self.min_event_frames > self.total_frames:
            raise ValueError(
                "composer min_events * min_event_frames exceeds clip length; got "
                f"{self.min_events * self.min_event_frames} > {self.total_frames}"
            )
        if self.crossfade_samples < 0:
            raise ValueError(
                f"crossfade_samples must be non-negative; got {self.crossfade_samples}"
            )
        if self.crossfade_samples >= self.stride_samples:
            raise ValueError(
                "crossfade_samples must be smaller than stride_samples; got "
                f"{self.crossfade_samples} vs {self.stride_samples}"
            )


@dataclass(frozen=True)
class SpeakerCast:
    """Concrete speaker / utterance assignment for one clip plan."""

    target_speaker_id: str
    interferer_speaker_ids: tuple[str, ...]
    target_entry: AudioEntry
    enrollment_entry: AudioEntry
    interferer_entries: dict[str, AudioEntry]


@dataclass(frozen=True)
class _TemplateAnchor:
    event_type: EventType
    min_frames: int
    max_frames: int
    anchor_name: str


@dataclass(frozen=True)
class _ClipTemplate:
    family: ClipFamily
    anchors: tuple[_TemplateAnchor, ...]
    filler_choices_by_gap: tuple[tuple[EventType, ...], ...]


def _weighted_choice(
    weights: dict[_ChoiceT, float],
    rng: random.Random,
) -> _ChoiceT:
    total = float(sum(max(0.0, w) for w in weights.values()))
    if total <= 0.0:
        raise ValueError(f"weights must have positive mass; got {weights}")
    threshold = rng.uniform(0.0, total)
    acc = 0.0
    last_key = next(iter(weights))
    for key, weight in weights.items():
        acc += max(0.0, float(weight))
        if threshold <= acc:
            return key
        last_key = key
    return last_key


def _random_partition(total: int, n_parts: int, rng: random.Random) -> list[int]:
    parts = [0] * n_parts
    for _ in range(total):
        parts[rng.randrange(n_parts)] += 1
    return parts


def _max_true_run(mask: torch.Tensor) -> int:
    best = 0
    cur = 0
    for value in mask.tolist():
        if value:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def _has_target_return_after_absence(
    target_active: torch.Tensor,
    *,
    min_absence_frames: int,
) -> bool:
    saw_target = False
    absence = 0
    for value in target_active.tolist():
        if value:
            if saw_target and absence >= min_absence_frames:
                return True
            saw_target = True
            absence = 0
        elif saw_target:
            absence += 1
    return False


class ClipPlanner:
    """Generate a clip plan with hard constraints satisfied by construction."""

    def __init__(
        self,
        cfg: ComposerConfig,
        *,
        allow_third_speaker: bool = True,
    ) -> None:
        self.cfg = cfg
        self.allow_third_speaker = allow_third_speaker
        self._templates = self._build_templates(cfg)

    def plan(
        self,
        family: ClipFamily | None = None,
        rng: random.Random | None = None,
    ) -> ClipPlan:
        rng = rng or random.Random()
        family = family or _weighted_choice(self.cfg.family_weights, rng)
        use_third_speaker = (
            self.allow_third_speaker
            and rng.random() < self.cfg.optional_third_speaker_prob
        )
        if family is ClipFamily.HARD_NEGATIVE_ABSENT and not self.allow_third_speaker:
            use_third_speaker = False

        template = rng.choice(self._templates[family])
        anchor_count = len(template.anchors)
        min_events = max(self.cfg.min_events, anchor_count)
        max_events = max(min_events, self.cfg.max_events)
        event_count = rng.randint(min_events, max_events)
        anchors = self._expand_template(template, event_count, rng)
        durations = self._allocate_frames(anchors, rng)
        slots = self._materialize_slots(
            family=family,
            anchors=anchors,
            durations=durations,
            use_third_speaker=use_third_speaker,
            rng=rng,
        )
        target_active, nontarget_active, overlap = self._labels_from_slots(slots)
        self._assert_constraints(
            family=family,
            slots=slots,
            target_active=target_active,
            nontarget_active=nontarget_active,
            overlap=overlap,
            use_third_speaker=use_third_speaker,
        )
        snr_db = (
            0.0
            if family is ClipFamily.HARD_NEGATIVE_ABSENT
            else rng.uniform(*self.cfg.snr_range_db)
        )
        noise_snr_db = rng.uniform(*self.cfg.noise_snr_range_db)
        return ClipPlan(
            family=family,
            slots=tuple(slots),
            total_frames=self.cfg.total_frames,
            stride_samples=self.cfg.stride_samples,
            segment_samples=self.cfg.segment_samples,
            use_third_speaker=use_third_speaker,
            target_present=family is not ClipFamily.HARD_NEGATIVE_ABSENT,
            snr_db=snr_db,
            noise_snr_db=noise_snr_db,
            target_active_frames=target_active,
            nontarget_active_frames=nontarget_active,
            overlap_frames=overlap,
        )

    def _build_templates(
        self,
        cfg: ComposerConfig,
    ) -> dict[ClipFamily, list[_ClipTemplate]]:
        return {
            ClipFamily.MULTI_TURN_TARGET_PRESENT: [
                _ClipTemplate(
                    family=ClipFamily.MULTI_TURN_TARGET_PRESENT,
                    anchors=(
                        _TemplateAnchor(
                            EventType.TARGET_ONLY,
                            cfg.min_target_only_frames,
                            cfg.max_event_frames,
                            "target_only_anchor",
                        ),
                        _TemplateAnchor(
                            EventType.NONTARGET_ONLY,
                            cfg.min_absence_before_return_frames,
                            cfg.max_event_frames,
                            "absence_anchor",
                        ),
                        _TemplateAnchor(
                            EventType.OVERLAP,
                            cfg.min_overlap_frames,
                            cfg.max_event_frames,
                            "overlap_anchor",
                        ),
                        _TemplateAnchor(
                            EventType.TARGET_ONLY,
                            cfg.min_target_only_frames,
                            cfg.max_event_frames,
                            "target_return",
                        ),
                    ),
                    filler_choices_by_gap=(
                        (EventType.BACKGROUND_ONLY, EventType.NONTARGET_ONLY),
                        (EventType.BACKGROUND_ONLY,),
                        (EventType.NONTARGET_ONLY, EventType.BACKGROUND_ONLY),
                        (EventType.OVERLAP, EventType.BACKGROUND_ONLY),
                        (EventType.NONTARGET_ONLY, EventType.BACKGROUND_ONLY),
                    ),
                ),
            ],
            ClipFamily.OVERLAP_HEAVY: [
                _ClipTemplate(
                    family=ClipFamily.OVERLAP_HEAVY,
                    anchors=(
                        _TemplateAnchor(
                            EventType.TARGET_ONLY,
                            cfg.min_target_only_frames,
                            cfg.max_event_frames,
                            "target_only_anchor",
                        ),
                        _TemplateAnchor(
                            EventType.OVERLAP,
                            cfg.min_overlap_frames,
                            cfg.max_event_frames,
                            "overlap_anchor_1",
                        ),
                        _TemplateAnchor(
                            EventType.NONTARGET_ONLY,
                            cfg.min_absence_before_return_frames,
                            cfg.max_event_frames,
                            "absence_anchor",
                        ),
                        _TemplateAnchor(
                            EventType.OVERLAP,
                            cfg.min_overlap_frames,
                            cfg.max_event_frames,
                            "overlap_anchor_2",
                        ),
                        _TemplateAnchor(
                            EventType.TARGET_ONLY,
                            cfg.min_target_only_frames,
                            cfg.max_event_frames,
                            "target_return",
                        ),
                    ),
                    filler_choices_by_gap=(
                        (EventType.OVERLAP, EventType.NONTARGET_ONLY),
                        (EventType.OVERLAP,),
                        (EventType.OVERLAP, EventType.NONTARGET_ONLY),
                        (EventType.OVERLAP, EventType.BACKGROUND_ONLY),
                        (EventType.OVERLAP, EventType.TARGET_ONLY),
                        (EventType.OVERLAP, EventType.BACKGROUND_ONLY),
                    ),
                ),
            ],
            ClipFamily.HARD_NEGATIVE_ABSENT: [
                _ClipTemplate(
                    family=ClipFamily.HARD_NEGATIVE_ABSENT,
                    anchors=(
                        _TemplateAnchor(
                            EventType.NONTARGET_ONLY,
                            cfg.min_nontarget_only_frames,
                            cfg.max_event_frames,
                            "nontarget_anchor",
                        ),
                        _TemplateAnchor(
                            EventType.OVERLAP,
                            cfg.min_overlap_frames,
                            cfg.max_event_frames,
                            "bc_overlap_anchor",
                        ),
                        _TemplateAnchor(
                            EventType.BACKGROUND_ONLY,
                            cfg.min_event_frames,
                            cfg.max_event_frames,
                            "background_anchor",
                        ),
                        _TemplateAnchor(
                            EventType.NONTARGET_ONLY,
                            cfg.min_event_frames,
                            cfg.max_event_frames,
                            "tail_nontarget",
                        ),
                    ),
                    filler_choices_by_gap=(
                        (EventType.NONTARGET_ONLY, EventType.BACKGROUND_ONLY),
                        (EventType.NONTARGET_ONLY, EventType.BACKGROUND_ONLY),
                        (EventType.BACKGROUND_ONLY, EventType.NONTARGET_ONLY),
                        (EventType.NONTARGET_ONLY, EventType.BACKGROUND_ONLY),
                        (EventType.NONTARGET_ONLY,),
                    ),
                ),
            ],
        }

    def _expand_template(
        self,
        template: _ClipTemplate,
        event_count: int,
        rng: random.Random,
    ) -> list[_TemplateAnchor]:
        filler_count = event_count - len(template.anchors)
        filler_by_gap = _random_partition(
            filler_count, len(template.filler_choices_by_gap), rng
        )
        anchors: list[_TemplateAnchor] = []
        for gap_index, allowed_types in enumerate(template.filler_choices_by_gap):
            for _ in range(filler_by_gap[gap_index]):
                filler_weights = {
                    event_type: 1.0 + (
                        1.5 if event_type is EventType.OVERLAP else 0.0
                    )
                    for event_type in allowed_types
                }
                filler_type = _weighted_choice(filler_weights, rng)
                anchors.append(
                    _TemplateAnchor(
                        filler_type,
                        self.cfg.min_event_frames,
                        self.cfg.max_event_frames,
                        "filler",
                    )
                )
            if gap_index < len(template.anchors):
                anchors.append(template.anchors[gap_index])
        return anchors

    def _allocate_frames(
        self,
        anchors: Sequence[_TemplateAnchor],
        rng: random.Random,
    ) -> list[int]:
        durations = [anchor.min_frames for anchor in anchors]
        max_caps = [anchor.max_frames for anchor in anchors]
        if sum(durations) > self.cfg.total_frames:
            raise RuntimeError("template minimum exceeds clip budget")
        if sum(max_caps) < self.cfg.total_frames:
            raise RuntimeError("template maximum cannot fill clip budget")
        remaining = self.cfg.total_frames - sum(durations)
        while remaining > 0:
            candidates = [
                idx for idx, (dur, cap) in enumerate(zip(durations, max_caps))
                if dur < cap
            ]
            idx = rng.choice(candidates)
            durations[idx] += 1
            remaining -= 1
        return durations

    def _materialize_slots(
        self,
        *,
        family: ClipFamily,
        anchors: Sequence[_TemplateAnchor],
        durations: Sequence[int],
        use_third_speaker: bool,
        rng: random.Random,
    ) -> list[EventSlot]:
        slots: list[EventSlot] = []
        frame_cursor = 0
        last_nontarget_role = "B"
        for index, (anchor, duration) in enumerate(zip(anchors, durations)):
            active_roles, nontarget_role = self._roles_for_event(
                family=family,
                event_type=anchor.event_type,
                use_third_speaker=use_third_speaker,
                last_nontarget_role=last_nontarget_role,
                rng=rng,
            )
            if nontarget_role is not None:
                last_nontarget_role = nontarget_role
            slots.append(
                EventSlot(
                    index=index,
                    event_type=anchor.event_type,
                    start_frame=frame_cursor,
                    end_frame=frame_cursor + duration,
                    active_roles=active_roles,
                    anchor_name=None if anchor.anchor_name == "filler" else anchor.anchor_name,
                    nontarget_role=nontarget_role,
                )
            )
            frame_cursor += duration
        return slots

    def _roles_for_event(
        self,
        *,
        family: ClipFamily,
        event_type: EventType,
        use_third_speaker: bool,
        last_nontarget_role: str,
        rng: random.Random,
    ) -> tuple[tuple[str, ...], str | None]:
        if family is ClipFamily.HARD_NEGATIVE_ABSENT:
            if event_type is EventType.NONTARGET_ONLY:
                role = "B" if not use_third_speaker else rng.choice(("B", "C"))
                return (role,), role
            if event_type is EventType.OVERLAP:
                if use_third_speaker:
                    return ("B", "C"), "C"
                return ("B",), "B"
            return (), None

        if event_type is EventType.TARGET_ONLY:
            return ("A",), None
        if event_type is EventType.NONTARGET_ONLY:
            role = "B" if not use_third_speaker else rng.choice(("B", "C"))
            return (role,), role
        if event_type is EventType.OVERLAP:
            role = "B"
            if use_third_speaker:
                role = "C" if last_nontarget_role == "B" else "B"
            return ("A", role), role
        return (), None

    def _labels_from_slots(
        self,
        slots: Sequence[EventSlot],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        target_active = torch.zeros(self.cfg.total_frames, dtype=torch.bool)
        nontarget_active = torch.zeros(self.cfg.total_frames, dtype=torch.bool)
        overlap = torch.zeros(self.cfg.total_frames, dtype=torch.bool)
        for slot in slots:
            sl = slice(slot.start_frame, slot.end_frame)
            target_active[sl] = "A" in slot.active_roles
            nontarget_active[sl] = any(
                role in slot.active_roles for role in ("B", "C")
            )
            overlap[sl] = len(slot.active_roles) >= 2
        return target_active, nontarget_active, overlap

    def _assert_constraints(
        self,
        *,
        family: ClipFamily,
        slots: Sequence[EventSlot],
        target_active: torch.Tensor,
        nontarget_active: torch.Tensor,
        overlap: torch.Tensor,
        use_third_speaker: bool,
    ) -> None:
        if not slots:
            raise AssertionError("planner produced no event slots")
        if slots[0].start_frame != 0:
            raise AssertionError("planner must start at frame 0")
        if slots[-1].end_frame != self.cfg.total_frames:
            raise AssertionError("planner must cover the full clip")
        for slot in slots:
            if slot.duration_frames < self.cfg.min_event_frames:
                raise AssertionError("slot shorter than min_event_frames")
            if slot.duration_frames > self.cfg.max_event_frames:
                raise AssertionError("slot longer than max_event_frames")
        if family is ClipFamily.HARD_NEGATIVE_ABSENT:
            if target_active.any():
                raise AssertionError("hard-negative plan may not activate target")
            if not nontarget_active.any():
                raise AssertionError("hard-negative plan must activate nontarget")
            if use_third_speaker and not overlap.any():
                raise AssertionError(
                    "hard-negative plan with third speaker must contain overlap"
                )
            return
        target_only = target_active & ~nontarget_active
        nontarget_only = nontarget_active & ~target_active
        if _max_true_run(target_only) < self.cfg.min_target_only_frames:
            raise AssertionError("target_only constraint not satisfied")
        if _max_true_run(nontarget_only) < self.cfg.min_nontarget_only_frames:
            raise AssertionError("nontarget_only constraint not satisfied")
        if _max_true_run(overlap) < self.cfg.min_overlap_frames:
            raise AssertionError("overlap constraint not satisfied")
        if not _has_target_return_after_absence(
            target_active,
            min_absence_frames=self.cfg.min_absence_before_return_frames,
        ):
            raise AssertionError("target return after absence constraint not satisfied")


class ClipRenderer:
    """Render a :class:`ClipPlan` into waveforms and framewise labels."""

    def __init__(
        self,
        mixer_cfg: MixerConfig,
        *,
        noise_pool: Sequence[NoiseEntry] | None = None,
        rir_pool: Sequence[torch.Tensor] | None = None,
    ) -> None:
        self.cfg = mixer_cfg
        self.composer = mixer_cfg.composer
        self._enrollment_len = int(mixer_cfg.enrollment_seconds * mixer_cfg.sample_rate)
        self.noise_pool = list(noise_pool) if noise_pool else []
        self._has_noise_pool = bool(self.noise_pool) and mixer_cfg.use_noise_pool
        self._rir_pool = list(rir_pool) if rir_pool else []

    def render(
        self,
        plan: ClipPlan,
        cast: SpeakerCast,
        speaker_to_idx: dict[str, int],
        rng: random.Random,
    ) -> dict:
        role_stems = self._load_role_stems(plan, cast, rng)
        role_tracks = self._stitch_role_tracks(plan, role_stems)
        role_tracks = {
            role: self._normalize(track) for role, track in role_tracks.items()
        }
        enrollment = self._normalize(
            _load_chunk(cast.enrollment_entry, self._enrollment_len, rng)
        )

        if self.cfg.apply_reverb and rng.random() < self.cfg.reverb_prob:
            clip_rir = self._sample_rir(rng)
            role_tracks = {
                role: self._normalize(apply_rir(track, clip_rir))
                for role, track in role_tracks.items()
            }
            if self.cfg.reverb_enrollment:
                enrollment = self._normalize(
                    apply_rir(enrollment, self._sample_rir(rng))
                )

        gain_envelopes = self._build_gain_envelopes(plan, rng)
        role_tracks = {
            role: role_tracks[role] * gain_envelopes.get(role, 1.0)
            for role in role_tracks
        }

        target_track = role_tracks.get(
            "A", torch.zeros(plan.segment_samples, dtype=torch.float32)
        )
        nontarget_track = torch.zeros_like(target_track)
        for role, track in role_tracks.items():
            if role != "A":
                nontarget_track = nontarget_track + track

        if plan.target_present and nontarget_track.abs().max() > 0:
            nontarget_track = _rescale_to_snr(
                target_track, nontarget_track, plan.snr_db
            )
        else:
            target_track = torch.zeros_like(target_track)

        mixture = target_track + nontarget_track
        mixture = self._maybe_add_noise(mixture, plan.noise_snr_db, rng)
        mixture, target_track = self._clip_guard(mixture, target_track)

        return {
            "mixture": mixture,
            "target": target_track,
            "enrollment": enrollment,
            "target_present": torch.tensor(
                1.0 if plan.target_present else 0.0, dtype=torch.float32,
            ),
            "target_speaker_idx": torch.tensor(
                speaker_to_idx[cast.target_speaker_id], dtype=torch.long,
            ),
            "snr_db": torch.tensor(
                plan.snr_db if plan.target_present else 0.0, dtype=torch.float32,
            ),
            "target_active_frames": plan.target_active_frames.clone(),
            "nontarget_active_frames": plan.nontarget_active_frames.clone(),
            "overlap_frames": plan.overlap_frames.clone(),
        }

    def _load_role_stems(
        self,
        plan: ClipPlan,
        cast: SpeakerCast,
        rng: random.Random,
    ) -> dict[str, torch.Tensor]:
        stems = {
            "A": _load_chunk(cast.target_entry, plan.segment_samples, rng),
            "B": _load_chunk(cast.interferer_entries["B"], plan.segment_samples, rng),
        }
        if plan.use_third_speaker and "C" in cast.interferer_entries:
            stems["C"] = _load_chunk(
                cast.interferer_entries["C"], plan.segment_samples, rng,
            )
        return {
            role: stem.to(dtype=torch.float32) for role, stem in stems.items()
        }

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

    def _build_gain_envelopes(
        self,
        plan: ClipPlan,
        rng: random.Random,
    ) -> dict[str, torch.Tensor]:
        roles = {"A", "B"}
        if plan.use_third_speaker:
            roles.add("C")
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
        if fade > 0:
            fade_in = 0.5 - 0.5 * torch.cos(
                torch.linspace(0.0, math.pi, steps=fade)
            )
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

        drift = self._sample_gain_drift(plan.segment_samples, rng)
        for role in roles:
            envelopes[role] = envelopes[role] * drift
        return envelopes

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

    def _normalize(self, signal: torch.Tensor) -> torch.Tensor:
        return signal / (_rms(signal) + 1e-8) * self.cfg.rms_target

    def _clip_guard(
        self,
        mixture: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        peak = float(mixture.abs().max())
        if peak > self.cfg.peak_clip:
            scale = self.cfg.peak_clip / peak
            mixture = mixture * scale
            target = target * scale
        return mixture, target
