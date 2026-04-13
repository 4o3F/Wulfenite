"""Unit tests for the unified long-scene composer."""

from __future__ import annotations

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


def test_planner_unified_scene_constraints() -> None:
    planner = ClipPlanner(MixerConfig().composer, allow_third_speaker=True)
    for seed in range(32):
        plan = planner.plan(rng=random.Random(seed))
        transitions = [
            (plan.slots[i].event_type, plan.slots[i + 1].event_type)
            for i in range(len(plan.slots) - 1)
        ]
        nontarget = torch.zeros_like(plan.active_frames_by_role["A"])
        for role, active in plan.active_frames_by_role.items():
            if role != "A":
                nontarget = nontarget | active.bool()
        target_only = plan.active_frames_by_role["A"] & ~nontarget
        assert plan.family is ClipFamily.UNIFIED_LONG_SCENE
        assert len(plan.slots) == 7
        assert plan.slots[0].start_frame == 0
        assert plan.slots[-1].end_frame == plan.total_frames
        assert target_only.any()
        assert plan.active_frames_by_role["B"].any()
        assert plan.overlap_frames.any()
        assert plan.background_frames.any()
        assert (EventType.TARGET_ONLY, EventType.NONTARGET_ONLY) in transitions
        assert (EventType.NONTARGET_ONLY, EventType.TARGET_ONLY) in transitions
        assert (EventType.TARGET_ONLY, EventType.OVERLAP) in transitions


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
