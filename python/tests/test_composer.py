"""Unit tests for the clip composer."""

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


def _build_speakers(root: Path, *, num_speakers: int = 4, utts_per_speaker: int = 4) -> dict:
    split_dir = root / "data_aishell" / "wav" / "train"
    for s in range(num_speakers):
        spk_id = f"S{s:04d}"
        for u in range(utts_per_speaker):
            _write_sine(
                split_dir / spk_id / f"BAC009{spk_id}W{u:04d}.wav",
                seconds=6.0,
                freq=220.0 + 45.0 * s + 5.0 * u,
            )
    return scan_aishell1(root)


def _renderer_and_cast(tmp_path: Path) -> tuple[ClipPlanner, ClipRenderer, SpeakerCast]:
    speakers = _build_speakers(tmp_path / "aishell")
    cfg = MixerConfig(
        composition_mode="clip_composer",
        segment_seconds=4.0,
        enrollment_seconds=4.0,
        apply_reverb=False,
        apply_noise=False,
    )
    planner = ClipPlanner(cfg.composer, allow_third_speaker=True)
    renderer = ClipRenderer(cfg, noise_pool=None, rir_pool=[])
    target_id = "S0000"
    cast = SpeakerCast(
        target_speaker_id=target_id,
        interferer_speaker_ids=("S0001", "S0002"),
        target_entry=speakers[target_id][0],
        enrollment_entry=speakers[target_id][1],
        interferer_entries={
            "B": speakers["S0001"][0],
            "C": speakers["S0002"][0],
        },
    )
    return planner, renderer, cast


def test_planner_multiturn_constraints() -> None:
    planner = ClipPlanner(MixerConfig().composer, allow_third_speaker=True)
    for seed in range(64):
        plan = planner.plan(ClipFamily.MULTI_TURN_TARGET_PRESENT, random.Random(seed))
        target_only = plan.target_active_frames & ~plan.nontarget_active_frames
        nontarget_only = plan.nontarget_active_frames & ~plan.target_active_frames
        assert len(plan.slots) >= 4
        assert len(plan.slots) <= 8
        assert target_only.any()
        assert nontarget_only.any()
        assert plan.overlap_frames.any()
        assert any(slot.anchor_name == "target_return" for slot in plan.slots)


def test_planner_hard_negative_no_target() -> None:
    planner = ClipPlanner(MixerConfig().composer, allow_third_speaker=True)
    plan = planner.plan(ClipFamily.HARD_NEGATIVE_ABSENT, random.Random(0))
    assert not plan.target_present
    assert not plan.target_active_frames.any()
    assert plan.nontarget_active_frames.any()


def test_planner_family_distribution() -> None:
    planner = ClipPlanner(MixerConfig().composer, allow_third_speaker=True)
    counts = {family: 0 for family in ClipFamily}
    draws = 2000
    for seed in range(draws):
        plan = planner.plan(None, random.Random(seed))
        counts[plan.family] += 1
    assert counts[ClipFamily.MULTI_TURN_TARGET_PRESENT] / draws == pytest.approx(
        0.60, abs=0.05,
    )
    assert counts[ClipFamily.OVERLAP_HEAVY] / draws == pytest.approx(
        0.25, abs=0.05,
    )
    assert counts[ClipFamily.HARD_NEGATIVE_ABSENT] / draws == pytest.approx(
        0.15, abs=0.05,
    )


def test_planner_stride_alignment() -> None:
    planner = ClipPlanner(MixerConfig().composer, allow_third_speaker=True)
    plan = planner.plan(None, random.Random(1))
    assert plan.slots[0].start_frame == 0
    assert plan.slots[-1].end_frame == plan.total_frames
    assert all(slot.start_frame < slot.end_frame for slot in plan.slots)


def test_planner_event_count_bounds() -> None:
    planner = ClipPlanner(MixerConfig().composer, allow_third_speaker=True)
    for seed in range(32):
        plan = planner.plan(None, random.Random(seed))
        assert 4 <= len(plan.slots) <= 8


def test_renderer_label_consistency(tmp_path: Path) -> None:
    planner, renderer, cast = _renderer_and_cast(tmp_path)
    plan = planner.plan(ClipFamily.MULTI_TURN_TARGET_PRESENT, random.Random(2))
    sample = renderer.render(
        plan,
        cast,
        {"S0000": 0, "S0001": 1, "S0002": 2},
        random.Random(3),
    )
    stride = plan.stride_samples
    for frame in range(plan.total_frames):
        start = frame * stride
        end = start + stride
        target_energy = float(sample["target"][start:end].abs().sum())
        if bool(sample["target_active_frames"][frame]):
            assert target_energy > 0.0
        else:
            assert target_energy == pytest.approx(0.0, abs=1e-6)


def test_renderer_crossfade_no_clicks(tmp_path: Path) -> None:
    planner, renderer, cast = _renderer_and_cast(tmp_path)
    plan = planner.plan(ClipFamily.OVERLAP_HEAVY, random.Random(4))
    sample = renderer.render(
        plan,
        cast,
        {"S0000": 0, "S0001": 1, "S0002": 2},
        random.Random(5),
    )
    peak_delta = float((sample["mixture"][1:] - sample["mixture"][:-1]).abs().max())
    assert peak_delta < 0.5


def test_renderer_finite_values(tmp_path: Path) -> None:
    planner, renderer, cast = _renderer_and_cast(tmp_path)
    plan = planner.plan(None, random.Random(6))
    sample = renderer.render(
        plan,
        cast,
        {"S0000": 0, "S0001": 1, "S0002": 2},
        random.Random(7),
    )
    assert torch.isfinite(sample["mixture"]).all()
    assert torch.isfinite(sample["target"]).all()
    assert torch.isfinite(sample["enrollment"]).all()


def test_renderer_overlap_snr_accuracy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, renderer, cast = _renderer_and_cast(tmp_path)
    renderer.composer.crossfade_samples = 0
    total_frames = renderer.composer.total_frames
    overlap_start = total_frames // 4
    overlap_end = (3 * total_frames) // 4
    target_active = torch.ones(total_frames, dtype=torch.bool)
    nontarget_active = torch.zeros(total_frames, dtype=torch.bool)
    nontarget_active[overlap_start:overlap_end] = True
    overlap_frames = target_active & nontarget_active
    plan = ClipPlan(
        family=ClipFamily.MULTI_TURN_TARGET_PRESENT,
        slots=(
            EventSlot(
                index=0,
                event_type=EventType.TARGET_ONLY,
                start_frame=0,
                end_frame=overlap_start,
                active_roles=("A",),
            ),
            EventSlot(
                index=1,
                event_type=EventType.OVERLAP,
                start_frame=overlap_start,
                end_frame=overlap_end,
                active_roles=("A", "B"),
                nontarget_role="B",
            ),
            EventSlot(
                index=2,
                event_type=EventType.TARGET_ONLY,
                start_frame=overlap_end,
                end_frame=total_frames,
                active_roles=("A",),
            ),
        ),
        total_frames=total_frames,
        stride_samples=renderer.composer.stride_samples,
        segment_samples=renderer.composer.segment_samples,
        use_third_speaker=False,
        target_present=True,
        snr_db=6.0,
        noise_snr_db=20.0,
        target_active_frames=target_active,
        nontarget_active_frames=nontarget_active,
        overlap_frames=overlap_frames,
    )

    monkeypatch.setattr(
        renderer,
        "_load_role_stems",
        lambda plan_, cast_, rng_: {
            "A": torch.full((plan_.segment_samples,), 0.25, dtype=torch.float32),
            "B": torch.full((plan_.segment_samples,), 0.50, dtype=torch.float32),
        },
    )

    sample = renderer.render(
        plan,
        cast,
        {"S0000": 0, "S0001": 1, "S0002": 2},
        random.Random(8),
    )
    overlap_mask = plan.overlap_frames.repeat_interleave(plan.stride_samples)[
        :plan.segment_samples
    ]
    target_overlap = sample["target"][overlap_mask]
    nontarget_overlap = (sample["mixture"] - sample["target"])[overlap_mask]
    target_rms = float(torch.sqrt(torch.mean(target_overlap * target_overlap)))
    nontarget_rms = float(
        torch.sqrt(torch.mean(nontarget_overlap * nontarget_overlap))
    )
    measured_snr_db = 20.0 * math.log10(target_rms / (nontarget_rms + 1e-12))

    assert measured_snr_db == pytest.approx(plan.snr_db, abs=1.0)
