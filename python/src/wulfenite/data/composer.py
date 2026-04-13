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
from dataclasses import dataclass, field
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
    template_name: str | None = None
    overlap_density: str | None = None
    overlap_ratio: float | None = None


OVERLAP_DENSITY_ORDER = ("sparse", "medium", "dense")
_ROLE_POST_OV = "post_ov"
_ROLE_OV2 = "ov2"
TEMPLATES: dict[str, dict[str, object]] = {
    "baseline_mid_overlap": {
        "density": "medium",
        "slots": (
            (EventType.TARGET_ONLY, ("A",), "target_only_intro"),
            (EventType.NONTARGET_ONLY, ("B",), "nontarget_only_intro"),
            (EventType.TARGET_ONLY, ("A",), "target_return_after_absence"),
            (EventType.OVERLAP, ("A", "B"), "overlap_main"),
            (
                EventType.NONTARGET_ONLY,
                (_ROLE_POST_OV,),
                "nontarget_only_after_overlap",
            ),
            (EventType.BACKGROUND_ONLY, (), "background_only_reset"),
            (EventType.TARGET_ONLY, ("A",), "target_return_final"),
        ),
    },
    "early_overlap_double": {
        "density": "dense",
        "slots": (
            (EventType.TARGET_ONLY, ("A",), "target_only_intro"),
            (EventType.OVERLAP, ("A", "B"), "overlap_main"),
            (EventType.NONTARGET_ONLY, ("B",), "nontarget_only_intro"),
            (EventType.TARGET_ONLY, ("A",), "target_return_after_absence"),
            (EventType.OVERLAP, ("A", _ROLE_OV2), "overlap_secondary"),
            (EventType.BACKGROUND_ONLY, (), "background_only_reset"),
            (EventType.TARGET_ONLY, ("A",), "target_return_final"),
        ),
    },
    "late_overlap_long_absence": {
        "density": "sparse",
        "slots": (
            (EventType.TARGET_ONLY, ("A",), "target_only_intro"),
            (EventType.NONTARGET_ONLY, ("B",), "nontarget_only_intro"),
            (EventType.BACKGROUND_ONLY, (), "background_only_reset"),
            (
                EventType.NONTARGET_ONLY,
                (_ROLE_POST_OV,),
                "nontarget_only_after_overlap",
            ),
            (EventType.TARGET_ONLY, ("A",), "target_return_after_absence"),
            (EventType.OVERLAP, ("A", "B"), "overlap_main"),
            (EventType.TARGET_ONLY, ("A",), "target_return_final"),
        ),
    },
    "background_mid_double_overlap": {
        "density": "dense",
        "slots": (
            (EventType.TARGET_ONLY, ("A",), "target_only_intro"),
            (EventType.OVERLAP, ("A", "B"), "overlap_main"),
            (EventType.NONTARGET_ONLY, ("B",), "nontarget_only_intro"),
            (EventType.BACKGROUND_ONLY, (), "background_only_reset"),
            (EventType.TARGET_ONLY, ("A",), "target_return_after_absence"),
            (EventType.OVERLAP, ("A", _ROLE_OV2), "overlap_secondary"),
            (EventType.TARGET_ONLY, ("A",), "target_return_final"),
        ),
    },
    "nontarget_heavy": {
        "density": "sparse",
        "slots": (
            (EventType.NONTARGET_ONLY, ("B",), "nontarget_only_intro"),
            (EventType.TARGET_ONLY, ("A",), "target_only_intro"),
            (
                EventType.NONTARGET_ONLY,
                (_ROLE_POST_OV,),
                "nontarget_only_after_overlap",
            ),
            (EventType.TARGET_ONLY, ("A",), "target_return_after_absence"),
            (EventType.BACKGROUND_ONLY, (), "background_only_reset"),
            (EventType.OVERLAP, ("A", "B"), "overlap_main"),
            (EventType.TARGET_ONLY, ("A",), "target_return_final"),
        ),
    },
}
TEMPLATE_NAMES_BY_DENSITY = {
    density: tuple(
        name
        for name, spec in TEMPLATES.items()
        if spec["density"] == density
    )
    for density in OVERLAP_DENSITY_ORDER
}


def _default_overlap_density_weights() -> dict[str, float]:
    return {
        "sparse": 0.20,
        "medium": 0.55,
        "dense": 0.25,
    }


def _default_overlap_ratio_ranges() -> dict[str, tuple[float, float]]:
    return {
        "sparse": (0.15, 0.25),
        "medium": (0.25, 0.40),
        "dense": (0.40, 0.55),
    }


def _resolve_template_roles(
    active_roles: tuple[str, ...],
    *,
    use_third_speaker: bool,
) -> tuple[str, ...]:
    resolved: list[str] = []
    overlap_role = "C" if use_third_speaker else "B"
    for role in active_roles:
        if role in {_ROLE_POST_OV, _ROLE_OV2}:
            resolved.append(overlap_role)
        else:
            resolved.append(role)
    return tuple(resolved)


def _slot_min_frames_for_cfg(
    cfg: "ComposerConfig",
    *,
    event_type: EventType,
    anchor_name: str,
) -> int:
    if event_type is EventType.TARGET_ONLY:
        return cfg.target_only_min_frames
    if event_type is EventType.NONTARGET_ONLY and anchor_name in {
        "nontarget_only_intro",
        "nontarget_only_after_overlap",
    }:
        return max(
            cfg.nontarget_only_min_frames,
            cfg.absence_before_return_min_frames,
        )
    if event_type is EventType.NONTARGET_ONLY:
        return cfg.nontarget_only_min_frames
    if event_type is EventType.OVERLAP:
        return cfg.overlap_min_frames
    if event_type is EventType.BACKGROUND_ONLY:
        return cfg.background_min_frames
    raise ValueError(f"unsupported event type {event_type!r}")


def _find_slot(slots: Sequence[EventSlot], anchor_name: str) -> EventSlot:
    return next(slot for slot in slots if slot.anchor_name == anchor_name)


def _distribute_extra_frames(
    durations: list[int],
    *,
    indices: Sequence[int],
    extra_frames: int,
    rng: random.Random,
) -> None:
    if extra_frames <= 0:
        return
    if not indices:
        raise ValueError("cannot distribute frames across an empty slot set")
    for _ in range(extra_frames):
        durations[rng.choice(tuple(indices))] += 1


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
    global_gain_range_db: tuple[float, float] = (-9.0, 9.0)
    snr_range_db: tuple[float, float] = (-5.0, 5.0)
    noise_snr_range_db: tuple[float, float] = (0.0, 25.0)
    overlap_density_weights: dict[str, float] = field(
        default_factory=_default_overlap_density_weights,
    )
    overlap_ratio_ranges: dict[str, tuple[float, float]] = field(
        default_factory=_default_overlap_ratio_ranges,
    )
    overlap_snr_center_range_db: tuple[float, float] = (-2.0, 4.0)
    overlap_snr_tail_range_db: tuple[float, float] = (-6.0, 8.0)
    overlap_snr_center_prob: float = 0.7

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
        for name, value in (
            ("gain_drift_db_range", self.gain_drift_db_range),
            ("global_gain_range_db", self.global_gain_range_db),
            ("snr_range_db", self.snr_range_db),
            ("noise_snr_range_db", self.noise_snr_range_db),
            ("overlap_snr_center_range_db", self.overlap_snr_center_range_db),
            ("overlap_snr_tail_range_db", self.overlap_snr_tail_range_db),
        ):
            lo, hi = value
            if lo > hi:
                raise ValueError(f"{name} must satisfy lo <= hi; got {value}")
        if not 0.0 <= self.overlap_snr_center_prob <= 1.0:
            raise ValueError(
                "overlap_snr_center_prob must be in [0, 1]; got "
                f"{self.overlap_snr_center_prob}"
            )
        weight_keys = set(self.overlap_density_weights.keys())
        if weight_keys != set(OVERLAP_DENSITY_ORDER):
            raise ValueError(
                "overlap_density_weights must define sparse/medium/dense; got "
                f"{sorted(weight_keys)}"
            )
        if sum(self.overlap_density_weights.values()) <= 0.0:
            raise ValueError("overlap_density_weights must sum to a positive value")
        ratio_keys = set(self.overlap_ratio_ranges.keys())
        if ratio_keys != set(OVERLAP_DENSITY_ORDER):
            raise ValueError(
                "overlap_ratio_ranges must define sparse/medium/dense; got "
                f"{sorted(ratio_keys)}"
            )
        for density, ratio_range in self.overlap_ratio_ranges.items():
            lo, hi = ratio_range
            if lo < 0.0 or hi > 1.0 or lo > hi:
                raise ValueError(
                    "overlap_ratio_ranges values must satisfy 0 <= lo <= hi <= 1; "
                    f"got {density}={ratio_range}"
                )
        min_required_frames = max(
            sum(
                _slot_min_frames_for_cfg(
                    self,
                    event_type=event_type,
                    anchor_name=anchor_name,
                )
                for event_type, _, anchor_name in spec["slots"]
            )
            for spec in TEMPLATES.values()
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


@dataclass(frozen=True)
class _TemplateSelection:
    template_name: str
    density: str
    specs: tuple[_AnchorSpec, ...]


def _required_samples_by_role(plan: ClipPlan) -> dict[str, int]:
    """Return the total source samples needed per role for ``plan``."""
    required = {
        role: 0 for role in plan.active_frames_by_role
    }
    for slot in plan.slots:
        length = (slot.end_frame - slot.start_frame) * plan.stride_samples
        for role in slot.active_roles:
            required[role] = required.get(role, 0) + length
    return required


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
        template = self._build_slot_specs(use_third_speaker, rng)
        durations, overlap_ratio = self._allocate_frames(
            template.specs,
            density=template.density,
            rng=rng,
        )
        slots = self._materialize_slots(template.specs, durations)
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
            snr_db=self._sample_overlap_snr_db(rng),
            noise_snr_db=rng.uniform(*self.cfg.noise_snr_range_db),
            active_frames_by_role=active_frames_by_role,
            overlap_frames=overlap_frames,
            background_frames=background_frames,
            template_name=template.template_name,
            overlap_density=template.density,
            overlap_ratio=overlap_ratio,
        )

    def _build_slot_specs(
        self,
        use_third_speaker: bool,
        rng: random.Random,
    ) -> _TemplateSelection:
        density = rng.choices(
            OVERLAP_DENSITY_ORDER,
            weights=[
                self.cfg.overlap_density_weights[density]
                for density in OVERLAP_DENSITY_ORDER
            ],
            k=1,
        )[0]
        template_name = rng.choice(TEMPLATE_NAMES_BY_DENSITY[density])
        template = TEMPLATES[template_name]
        slots = tuple(
            _AnchorSpec(
                event_type=event_type,
                min_frames=_slot_min_frames_for_cfg(
                    self.cfg,
                    event_type=event_type,
                    anchor_name=anchor_name,
                ),
                active_roles=_resolve_template_roles(
                    active_roles,
                    use_third_speaker=use_third_speaker,
                ),
                anchor_name=anchor_name,
            )
            for event_type, active_roles, anchor_name in template["slots"]
        )
        return _TemplateSelection(
            template_name=template_name,
            density=density,
            specs=slots,
        )

    def _allocate_frames(
        self,
        specs: Sequence[_AnchorSpec],
        *,
        density: str,
        rng: random.Random,
    ) -> tuple[list[int], float]:
        durations = [spec.min_frames for spec in specs]
        overlap_indices = [
            i for i, spec in enumerate(specs) if spec.event_type is EventType.OVERLAP
        ]
        non_overlap_indices = [
            i for i, spec in enumerate(specs) if spec.event_type is not EventType.OVERLAP
        ]
        overlap_min_frames = sum(durations[i] for i in overlap_indices)
        non_overlap_min_frames = sum(durations[i] for i in non_overlap_indices)
        ratio_lo, ratio_hi = self.cfg.overlap_ratio_ranges[density]
        feasible_lo = max(
            ratio_lo,
            overlap_min_frames / self.cfg.total_frames,
        )
        feasible_hi = min(
            ratio_hi,
            (self.cfg.total_frames - non_overlap_min_frames) / self.cfg.total_frames,
        )
        if feasible_hi < feasible_lo:
            raise ValueError(
                "configured overlap ratio range is incompatible with scene minima; "
                f"density={density} feasible=({feasible_lo:.3f}, {feasible_hi:.3f})"
            )
        target_overlap_ratio = rng.uniform(feasible_lo, feasible_hi)
        target_overlap_frames = int(round(target_overlap_ratio * self.cfg.total_frames))
        target_overlap_frames = max(target_overlap_frames, overlap_min_frames)
        max_overlap_frames = self.cfg.total_frames - non_overlap_min_frames
        target_overlap_frames = min(target_overlap_frames, max_overlap_frames)
        _distribute_extra_frames(
            durations,
            indices=overlap_indices,
            extra_frames=target_overlap_frames - overlap_min_frames,
            rng=rng,
        )
        remaining = self.cfg.total_frames - sum(durations)
        _distribute_extra_frames(
            durations,
            indices=non_overlap_indices,
            extra_frames=remaining,
            rng=rng,
        )
        actual_overlap_ratio = (
            sum(durations[i] for i in overlap_indices) / self.cfg.total_frames
        )
        return durations, actual_overlap_ratio

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
        if any(
            slots[i].end_frame != slots[i + 1].start_frame
            for i in range(len(slots) - 1)
        ):
            raise AssertionError("planner must not leave gaps or overlaps between slots")
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

        return_slot = _find_slot(slots, "target_return_after_absence")
        previous_target_end = max(
            slot.end_frame
            for slot in slots
            if slot.end_frame <= return_slot.start_frame
            and "A" in slot.active_roles
            and slot.anchor_name != "target_return_after_absence"
        )
        if (
            return_slot.start_frame - previous_target_end
            < self.cfg.absence_before_return_min_frames
        ):
            raise AssertionError("target return after absence minimum not satisfied")
        transitions = [
            (slots[i].event_type, slots[i + 1].event_type)
            for i in range(len(slots) - 1)
        ]

        def _has_order(before: EventType, after: EventType) -> bool:
            before_seen = False
            for slot in slots:
                if slot.event_type is before:
                    before_seen = True
                elif before_seen and slot.event_type is after:
                    return True
            return False

        required_orders = (
            (EventType.TARGET_ONLY, EventType.NONTARGET_ONLY),
            (EventType.NONTARGET_ONLY, EventType.TARGET_ONLY),
            (EventType.TARGET_ONLY, EventType.OVERLAP),
        )
        if not all(_has_order(before, after) for before, after in required_orders):
            raise AssertionError(f"missing required event ordering: {transitions}")
        if (
            not _has_order(EventType.NONTARGET_ONLY, EventType.OVERLAP)
            and not _has_order(EventType.OVERLAP, EventType.NONTARGET_ONLY)
        ):
            raise AssertionError("scene must contain other_only <-> overlap transition")

    def _sample_overlap_snr_db(self, rng: random.Random) -> float:
        center_lo, center_hi = self.cfg.overlap_snr_center_range_db
        tail_lo, tail_hi = self.cfg.overlap_snr_tail_range_db
        if rng.random() < self.cfg.overlap_snr_center_prob:
            return rng.uniform(center_lo, center_hi)
        lower_tail_valid = tail_lo < center_lo
        upper_tail_valid = tail_hi > center_hi
        if lower_tail_valid and upper_tail_valid:
            if rng.random() < 0.5:
                return rng.uniform(tail_lo, center_lo)
            return rng.uniform(center_hi, tail_hi)
        if lower_tail_valid:
            return rng.uniform(tail_lo, center_lo)
        if upper_tail_valid:
            return rng.uniform(center_hi, tail_hi)
        return rng.uniform(center_lo, center_hi)


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
        role_tracks = self._apply_role_augmentation(role_tracks, rng)
        role_tracks = self._match_scene_snr(plan, role_tracks)
        scene_gain = self._sample_global_gain(rng)
        role_tracks = {
            role: track * scene_gain for role, track in role_tracks.items()
        }
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

        active_frames_by_role, overlap_frames, background_frames = (
            self._labels_from_tracks(clean_tracks, frame_size=plan.stride_samples)
        )

        enrollment_pool = self._build_enrollment_pool(cast, rng)
        metadata = {
            "family": plan.family.value,
            "snr_db": float(plan.snr_db),
            "template_name": plan.template_name,
            "overlap_density": plan.overlap_density,
            "overlap_ratio": plan.overlap_ratio,
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
            active_frames_by_role=active_frames_by_role,
            overlap_frames=overlap_frames,
            background_frames=background_frames,
            metadata=metadata,
        )

    def _load_role_stems(
        self,
        plan: ClipPlan,
        cast: SpeakerCast,
        rng: random.Random,
    ) -> dict[str, torch.Tensor]:
        required_samples_by_role = _required_samples_by_role(plan)
        return {
            role: self._normalize(
                self._load_ranked_chunk(
                    entry,
                    required_samples_by_role.get(role, 0),
                    rng,
                    pick="loudest",
                )
            )
            for role, entry in cast.source_entries.items()
        }

    def _labels_from_tracks(
        self,
        role_tracks: dict[str, torch.Tensor],
        *,
        frame_size: int,
        activity_threshold: float = 1e-4,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        if not role_tracks:
            raise ValueError("role_tracks must not be empty")
        n_frames = next(iter(role_tracks.values())).shape[-1] // frame_size
        active_frames_by_role: dict[str, torch.Tensor] = {}
        for role, track in role_tracks.items():
            usable = n_frames * frame_size
            active_frames_by_role[role] = (
                track[:usable]
                .reshape(n_frames, frame_size)
                .abs()
                .amax(dim=-1)
                .gt(activity_threshold)
            )

        stacked = torch.stack(
            [mask.to(torch.int32) for mask in active_frames_by_role.values()], dim=0
        )
        overlap_frames = stacked.sum(dim=0) >= 2
        background_frames = stacked.sum(dim=0) == 0
        return active_frames_by_role, overlap_frames, background_frames

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
        tracks = {role: track.clone() for role, track in role_tracks.items()}
        if self.cfg.apply_reverb and rng.random() < self.cfg.reverb_prob:
            tracks = {
                role: apply_rir(track, self._sample_rir(rng))
                for role, track in tracks.items()
            }
        return {
            role: track * self._sample_temporal_drift(track.numel(), rng)
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
            audio = self._load_ranked_chunk(
                entry,
                self._enrollment_len,
                rng,
                pick="loudest",
            )
        elif mode == "silence_heavy":
            audio = self._load_ranked_chunk(
                entry,
                self._enrollment_len,
                rng,
                pick="quietest",
            )
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

    def _load_ranked_chunk(
        self,
        entry: AudioEntry,
        target_len: int,
        rng: random.Random,
        *,
        pick: str,
        n_trials: int = 8,
    ) -> torch.Tensor:
        if target_len <= 0:
            return torch.zeros(0, dtype=torch.float32)

        if entry.num_frames <= 0:
            return torch.zeros(target_len, dtype=torch.float32)

        key = max if pick == "loudest" else min
        chunks: list[torch.Tensor] = []
        remaining = target_len
        while remaining > 0:
            chunk_len = min(remaining, entry.num_frames)
            candidates = [
                _load_chunk(entry, chunk_len, rng).to(torch.float32)
                for _ in range(max(1, n_trials))
            ]
            chunks.append(key(candidates, key=lambda chunk: float(_rms(chunk))))
            remaining -= chunk_len
        return torch.cat(chunks, dim=0)[:target_len]

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

    def _sample_global_gain(self, rng: random.Random) -> float:
        lo_db, hi_db = self.composer.global_gain_range_db
        gain_db = rng.uniform(lo_db, hi_db)
        return 10.0 ** (gain_db / 20.0)

    def _sample_temporal_drift(
        self,
        n_samples: int,
        rng: random.Random,
    ) -> torch.Tensor:
        lo_db, hi_db = self.composer.gain_drift_db_range
        if n_samples <= 0 or (lo_db == 0.0 and hi_db == 0.0):
            return torch.ones(n_samples, dtype=torch.float32)
        amplitude_db = rng.uniform(lo_db, hi_db)
        phase = rng.uniform(0.0, 2.0 * math.pi)
        cycles = rng.uniform(0.5, 1.5)
        t = torch.linspace(0.0, 1.0, steps=n_samples)
        drift_db = amplitude_db * torch.sin(2.0 * math.pi * cycles * t + phase)
        return torch.pow(10.0, drift_db / 20.0)

    def _normalize(self, signal: torch.Tensor) -> torch.Tensor:
        return signal / (_rms(signal) + 1e-8) * self.cfg.rms_target
