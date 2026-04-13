"""Unit tests for the unified long-scene composer."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
import math
import random
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

from wulfenite.data import MixerConfig, scan_aishell1
from wulfenite.data.composer import (
    ClipFamily,
    ClipPlan,
    ClipPlanner,
    ClipRenderer,
    EventSlot,
    EventType,
    SpeakerCast,
    TEMPLATES,
    _AnchorSpec,
    _resolve_template_roles,
    _slot_min_frames_for_cfg,
)


SR = 16000


def _write_sine(path: Path, seconds: float, freq: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.arange(int(seconds * SR)) / SR
    audio = (0.25 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    sf.write(str(path), audio, SR)


def _build_speakers(
    root: Path,
    *,
    num_speakers: int = 5,
    utts_per_speaker: int = 4,
) -> dict:
    split_dir = root / "data_aishell" / "wav" / "train"
    for s in range(num_speakers):
        spk_id = f"S{s:04d}"
        for u in range(utts_per_speaker):
            _write_sine(
                split_dir / spk_id / f"BAC009{spk_id}W{u:04d}.wav",
                seconds=12.0,
                freq=220.0 + 45.0 * s + 5.0 * u,
            )
    return scan_aishell1(root)


def _renderer_and_cast(
    tmp_path: Path,
) -> tuple[ClipPlanner, ClipRenderer, SpeakerCast, dict[str, int]]:
    speakers = _build_speakers(tmp_path / "aishell")
    cfg = MixerConfig(
        composition_mode="clip_composer",
        segment_seconds=8.0,
        enrollment_seconds=4.0,
        apply_reverb=False,
        apply_noise=False,
    )
    planner = ClipPlanner(cfg.composer, allow_third_speaker=True)
    renderer = ClipRenderer(cfg, noise_pool=None, rir_pool=[])
    cast = SpeakerCast(
        source_speaker_ids={"A": "S0000", "B": "S0001", "C": "S0002"},
        source_entries={
            "A": speakers["S0000"][0],
            "B": speakers["S0001"][0],
            "C": speakers["S0002"][0],
        },
        enrollment_entries={
            "A": speakers["S0000"][1],
            "B": speakers["S0001"][1],
            "C": speakers["S0002"][1],
        },
        outsider_speaker_id="S0003",
        outsider_enrollment_entry=speakers["S0003"][0],
    )
    speaker_to_idx = {sid: i for i, sid in enumerate(sorted(speakers))}
    return planner, renderer, cast, speaker_to_idx


def _template_specs(
    planner: ClipPlanner,
    template_name: str,
    *,
    use_third_speaker: bool,
) -> tuple[_AnchorSpec, ...]:
    template = TEMPLATES[template_name]
    return tuple(
        _AnchorSpec(
            event_type=event_type,
            min_frames=_slot_min_frames_for_cfg(
                planner.cfg,
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


def test_planner_unified_scene_constraints() -> None:
    planner = ClipPlanner(MixerConfig().composer, allow_third_speaker=True)
    for seed in range(32):
        plan = planner.plan(rng=random.Random(seed))
        event_sequence = [slot.event_type for slot in plan.slots]
        nontarget = torch.zeros_like(plan.active_frames_by_role["A"])
        for role, active in plan.active_frames_by_role.items():
            if role != "A":
                nontarget = nontarget | active.bool()
        target_only = plan.active_frames_by_role["A"] & ~nontarget
        assert plan.family is ClipFamily.UNIFIED_LONG_SCENE
        assert len(plan.slots) == 7
        assert plan.template_name in TEMPLATES
        assert plan.overlap_density == TEMPLATES[plan.template_name]["density"]
        assert plan.overlap_ratio is not None
        assert plan.slots[0].start_frame == 0
        assert plan.slots[-1].end_frame == plan.total_frames
        assert target_only.any()
        assert plan.active_frames_by_role["B"].any()
        assert plan.overlap_frames.any()
        assert plan.background_frames.any()
        assert EventType.NONTARGET_ONLY in event_sequence
        assert EventType.OVERLAP in event_sequence
        assert min(
            i for i, event_type in enumerate(event_sequence)
            if event_type is EventType.TARGET_ONLY
        ) < max(
            i for i, event_type in enumerate(event_sequence)
            if event_type is EventType.NONTARGET_ONLY
        )
        assert min(
            i for i, event_type in enumerate(event_sequence)
            if event_type is EventType.NONTARGET_ONLY
        ) < max(
            i for i, event_type in enumerate(event_sequence)
            if event_type is EventType.TARGET_ONLY
        )
        assert min(
            i for i, event_type in enumerate(event_sequence)
            if event_type is EventType.TARGET_ONLY
        ) < max(
            i for i, event_type in enumerate(event_sequence)
            if event_type is EventType.OVERLAP
        )


def test_planner_template_family_coverage() -> None:
    planner = ClipPlanner(MixerConfig().composer, allow_third_speaker=True)

    for i, template_name in enumerate(TEMPLATES):
        specs = _template_specs(
            planner,
            template_name,
            use_third_speaker=True,
        )
        durations, overlap_ratio = planner._allocate_frames(
            specs,
            density=TEMPLATES[template_name]["density"],
            rng=random.Random(100 + i),
        )
        slots = planner._materialize_slots(specs, durations)
        active_frames_by_role = planner._labels_from_slots(slots)
        overlap_frames = planner._compute_overlap(active_frames_by_role)
        background_frames = planner._compute_background(
            active_frames_by_role,
            planner.cfg.total_frames,
        )

        planner._assert_constraints(
            slots=slots,
            active_frames_by_role=active_frames_by_role,
            overlap_frames=overlap_frames,
            background_frames=background_frames,
        )

        assert len(slots) == 7
        assert 0.0 < overlap_ratio < 1.0


def test_planner_anchor_validation_handles_noncanonical_slot_order() -> None:
    planner = ClipPlanner(MixerConfig().composer, allow_third_speaker=False)
    specs = _template_specs(
        planner,
        "nontarget_heavy",
        use_third_speaker=False,
    )
    durations, _ = planner._allocate_frames(
        specs,
        density="sparse",
        rng=random.Random(17),
    )
    slots = planner._materialize_slots(specs, durations)
    active_frames_by_role = planner._labels_from_slots(slots)
    overlap_frames = planner._compute_overlap(active_frames_by_role)
    background_frames = planner._compute_background(
        active_frames_by_role,
        planner.cfg.total_frames,
    )

    planner._assert_constraints(
        slots=slots,
        active_frames_by_role=active_frames_by_role,
        overlap_frames=overlap_frames,
        background_frames=background_frames,
    )

    assert slots[0].event_type is EventType.NONTARGET_ONLY
    assert next(
        slot for slot in slots if slot.anchor_name == "target_only_intro"
    ).index == 1


def test_planner_density_selection_matches_weights() -> None:
    planner = ClipPlanner(MixerConfig().composer, allow_third_speaker=True)
    counts: Counter[str] = Counter()
    n_plans = 1200

    for seed in range(n_plans):
        plan = planner.plan(rng=random.Random(seed))
        assert plan.overlap_density is not None
        counts[plan.overlap_density] += 1

    for density, expected in planner.cfg.overlap_density_weights.items():
        observed = counts[density] / n_plans
        assert observed == pytest.approx(expected, abs=0.05)


@pytest.mark.parametrize("density", ("sparse", "medium", "dense"))
def test_planner_overlap_ratio_stays_within_density_range(density: str) -> None:
    cfg = replace(
        MixerConfig().composer,
        overlap_density_weights={
            "sparse": 1.0 if density == "sparse" else 0.0,
            "medium": 1.0 if density == "medium" else 0.0,
            "dense": 1.0 if density == "dense" else 0.0,
        },
    )
    planner = ClipPlanner(cfg, allow_third_speaker=True)
    lo, hi = cfg.overlap_ratio_ranges[density]

    for seed in range(32):
        plan = planner.plan(rng=random.Random(seed))
        assert plan.overlap_density == density
        assert plan.overlap_ratio is not None
        assert lo - 1e-6 <= plan.overlap_ratio <= hi + 1e-6


def test_renderer_bundle_preserves_roles_and_enrollment_pools(
    tmp_path: Path,
) -> None:
    planner, renderer, cast, speaker_to_idx = _renderer_and_cast(tmp_path)
    plan = planner.plan(rng=random.Random(2))
    bundle = renderer.render_bundle(11, plan, cast, speaker_to_idx, random.Random(3))
    expected_len = int(8.0 * SR)
    expected_frames = expected_len // plan.stride_samples

    assert bundle.scene_id == 11
    assert bundle.mixture.shape == (expected_len,)
    assert bundle.source_tracks["A"].shape == (expected_len,)
    assert bundle.source_tracks["B"].shape == (expected_len,)
    assert bundle.enrollment_pool["A"][0].shape == (int(4.0 * SR),)
    assert bundle.enrollment_pool["B"][0].shape == (int(4.0 * SR),)
    assert bundle.enrollment_pool["OUTSIDER"][0].shape == (int(4.0 * SR),)
    assert bundle.active_frames_by_role["A"].shape == (expected_frames,)
    assert bundle.background_frames.shape == (expected_frames,)
    assert bundle.overlap_frames.shape == (expected_frames,)


def test_renderer_loads_only_required_role_audio(tmp_path: Path) -> None:
    planner, renderer, cast, _ = _renderer_and_cast(tmp_path)
    plan = planner.plan(rng=random.Random(12))
    stems = renderer._load_role_stems(plan, cast, random.Random(13))

    for role, stem in stems.items():
        expected_len = sum(
            (slot.end_frame - slot.start_frame) * plan.stride_samples
            for slot in plan.slots
            if role in slot.active_roles
        )
        assert stem.shape == (expected_len,)


def test_renderer_label_consistency(tmp_path: Path) -> None:
    planner, renderer, cast, speaker_to_idx = _renderer_and_cast(tmp_path)
    plan = planner.plan(rng=random.Random(4))
    bundle = renderer.render_bundle(0, plan, cast, speaker_to_idx, random.Random(5))
    stride = plan.stride_samples
    for frame in range(plan.total_frames):
        start = frame * stride
        end = start + stride
        a_energy = float(bundle.source_tracks["A"][start:end].abs().sum())
        if bool(bundle.active_frames_by_role["A"][frame]):
            assert a_energy > 0.0
        else:
            assert a_energy == pytest.approx(0.0, abs=1e-6)


def test_renderer_labels_follow_rendered_audio_not_plan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    planner, renderer, cast, speaker_to_idx = _renderer_and_cast(tmp_path)
    plan = planner.plan(rng=random.Random(14))

    def fake_role_stems(
        plan_: ClipPlan,
        cast_: SpeakerCast,
        rng_: random.Random,
    ) -> dict[str, torch.Tensor]:
        stems: dict[str, torch.Tensor] = {}
        for role in cast_.source_entries:
            expected_len = sum(
                (slot.end_frame - slot.start_frame) * plan_.stride_samples
                for slot in plan_.slots
                if role in slot.active_roles
            )
            fill = 0.0 if role == "B" else 0.25
            stems[role] = torch.full(
                (expected_len,),
                fill,
                dtype=torch.float32,
            )
        return stems

    monkeypatch.setattr(renderer, "_load_role_stems", fake_role_stems)
    bundle = renderer.render_bundle(0, plan, cast, speaker_to_idx, random.Random(15))

    assert bundle.active_frames_by_role["A"].any()
    assert not bundle.active_frames_by_role["B"].any()
    assert not bundle.overlap_frames.any()
    assert bundle.background_frames.any()


def test_renderer_crossfade_no_clicks(tmp_path: Path) -> None:
    planner, renderer, cast, speaker_to_idx = _renderer_and_cast(tmp_path)
    plan = planner.plan(rng=random.Random(6))
    bundle = renderer.render_bundle(0, plan, cast, speaker_to_idx, random.Random(7))
    peak_delta = float((bundle.mixture[1:] - bundle.mixture[:-1]).abs().max())
    assert peak_delta < 0.6


def test_renderer_finite_values(tmp_path: Path) -> None:
    planner, renderer, cast, speaker_to_idx = _renderer_and_cast(tmp_path)
    plan = planner.plan(rng=random.Random(8))
    bundle = renderer.render_bundle(0, plan, cast, speaker_to_idx, random.Random(9))
    assert torch.isfinite(bundle.mixture).all()
    for track in bundle.source_tracks.values():
        assert torch.isfinite(track).all()
    for pool in bundle.enrollment_pool.values():
        for candidate in pool:
            assert torch.isfinite(candidate).all()


def test_renderer_normalizes_stems_before_sparse_stitching(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    planner, renderer, cast, speaker_to_idx = _renderer_and_cast(tmp_path)
    plan = planner.plan(rng=random.Random(18))

    def fake_load_ranked_chunk(
        entry: object,
        target_len: int,
        rng: random.Random,
        *,
        pick: str,
        n_trials: int = 8,
    ) -> torch.Tensor:
        del entry, rng, pick, n_trials
        return torch.full((target_len,), 0.5, dtype=torch.float32)

    monkeypatch.setattr(renderer, "_load_ranked_chunk", fake_load_ranked_chunk)
    monkeypatch.setattr(
        renderer,
        "_sample_temporal_drift",
        lambda n_samples, rng: torch.ones(n_samples, dtype=torch.float32),
    )
    monkeypatch.setattr(renderer, "_sample_global_gain", lambda rng: 1.0)
    bundle = renderer.render_bundle(0, plan, cast, speaker_to_idx, random.Random(19))

    active_mask = bundle.active_frames_by_role["A"].repeat_interleave(plan.stride_samples)[
        :plan.segment_samples
    ]
    active = bundle.source_tracks["A"][active_mask]
    full = bundle.source_tracks["A"]
    active_rms = float(torch.sqrt(torch.mean(active * active) + 1e-12))
    full_rms = float(torch.sqrt(torch.mean(full * full) + 1e-12))

    assert active_rms == pytest.approx(renderer.cfg.rms_target, rel=0.25, abs=1e-3)
    assert full_rms < active_rms


def test_renderer_overlap_snr_accuracy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, renderer, cast, speaker_to_idx = _renderer_and_cast(tmp_path)
    total_frames = renderer.composer.total_frames
    overlap_start = total_frames // 4
    overlap_end = (3 * total_frames) // 4
    active_frames_by_role = {
        "A": torch.ones(total_frames, dtype=torch.bool),
        "B": torch.zeros(total_frames, dtype=torch.bool),
    }
    active_frames_by_role["B"][overlap_start:overlap_end] = True
    background_frames = torch.zeros(total_frames, dtype=torch.bool)
    overlap_frames = active_frames_by_role["A"] & active_frames_by_role["B"]
    plan = ClipPlan(
        family=ClipFamily.UNIFIED_LONG_SCENE,
        slots=(
            EventSlot(
                index=0,
                event_type=EventType.TARGET_ONLY,
                start_frame=0,
                end_frame=overlap_start,
                active_roles=("A",),
                anchor_name="target_only_intro",
            ),
            EventSlot(
                index=1,
                event_type=EventType.OVERLAP,
                start_frame=overlap_start,
                end_frame=overlap_end,
                active_roles=("A", "B"),
                anchor_name="overlap_main",
            ),
            EventSlot(
                index=2,
                event_type=EventType.TARGET_ONLY,
                start_frame=overlap_end,
                end_frame=total_frames,
                active_roles=("A",),
                anchor_name="target_return_final",
            ),
        ),
        total_frames=total_frames,
        stride_samples=renderer.composer.stride_samples,
        segment_samples=renderer.composer.segment_samples,
        use_third_speaker=False,
        snr_db=6.0,
        noise_snr_db=20.0,
        active_frames_by_role=active_frames_by_role,
        overlap_frames=overlap_frames,
        background_frames=background_frames,
    )

    monkeypatch.setattr(
        renderer,
        "_load_role_stems",
        lambda plan_, cast_, rng_: {
            "A": torch.full((plan_.segment_samples,), 0.25, dtype=torch.float32),
            "B": torch.full((plan_.segment_samples,), 0.50, dtype=torch.float32),
        },
    )

    bundle = renderer.render_bundle(0, plan, cast, speaker_to_idx, random.Random(10))
    overlap_mask = overlap_frames.repeat_interleave(plan.stride_samples)[
        :plan.segment_samples
    ]
    target_overlap = bundle.source_tracks["A"][overlap_mask]
    nontarget_overlap = bundle.source_tracks["B"][overlap_mask]
    target_rms = float(torch.sqrt(torch.mean(target_overlap * target_overlap)))
    nontarget_rms = float(
        torch.sqrt(torch.mean(nontarget_overlap * nontarget_overlap))
    )
    measured_snr_db = 20.0 * math.log10(target_rms / (nontarget_rms + 1e-12))

    assert measured_snr_db == pytest.approx(plan.snr_db, abs=1.0)
