"""Unit tests for the data pipeline.

Tests use synthetic wav fixtures written to a pytest ``tmp_path`` so
they do not depend on the real AISHELL / DNS4 datasets being present.
Fixtures follow the same directory layout the scanners expect, so
the tests exercise the actual scan paths.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

from wulfenite.data import (
    AudioEntry,
    MixerConfig,
    NoiseEntry,
    WulfeniteMixer,
    apply_rir,
    add_noise_at_snr,
    collate_mixer_batch,
    merge_speaker_dicts,
    scan_aishell1,
    scan_aishell3,
    scan_noise_dir,
    synth_room_rir,
)
from wulfenite.data.augmentation import ReverbConfig


SR = 16000


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _write_sine_wav(path: Path, seconds: float, freq: float) -> None:
    """Write a tiny sine-wave mono 16 kHz wav at ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.arange(int(seconds * SR)) / SR
    audio = (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    sf.write(str(path), audio, SR)


def _build_fake_aishell1(root: Path, num_speakers: int = 3,
                        utts_per_speaker: int = 4,
                        seconds: float = 2.0) -> Path:
    """Write a synthetic AISHELL-1-shaped tree under ``root``."""
    split_dir = root / "data_aishell" / "wav" / "train"
    for s in range(num_speakers):
        spk_id = f"S{s:04d}"
        for u in range(utts_per_speaker):
            freq = 200 + 50 * s + 5 * u  # distinct per utterance
            _write_sine_wav(
                split_dir / spk_id / f"BAC009{spk_id}W{u:04d}.wav",
                seconds, freq,
            )
    return root


def _build_fake_aishell3(root: Path, num_speakers: int = 3,
                        utts_per_speaker: int = 4,
                        seconds: float = 2.0) -> Path:
    """Write a synthetic AISHELL-3-shaped tree under ``root``."""
    split_dir = root / "train" / "wav"
    for s in range(num_speakers):
        spk_id = f"SSB{s:04d}"
        for u in range(utts_per_speaker):
            freq = 300 + 60 * s + 7 * u
            _write_sine_wav(
                split_dir / spk_id / f"{spk_id}{u:04d}.wav",
                seconds, freq,
            )
    return root


def _build_fake_noise_dir(root: Path, num_files: int = 4,
                         seconds: float = 3.0) -> Path:
    """Write a flat directory of noise wavs under ``root``."""
    root.mkdir(parents=True, exist_ok=True)
    for i in range(num_files):
        t = np.arange(int(seconds * SR)) / SR
        # White-ish noise — deterministic per file so tests stay stable
        rng = np.random.default_rng(i + 42)
        audio = (0.05 * rng.standard_normal(t.shape)).astype(np.float32)
        sf.write(str(root / f"noise_{i:02d}.wav"), audio, SR)
    return root


# ---------------------------------------------------------------------------
# Scanner tests
# ---------------------------------------------------------------------------


def test_scan_aishell1(tmp_path: Path) -> None:
    root = _build_fake_aishell1(tmp_path / "aishell1", num_speakers=3,
                                 utts_per_speaker=4)
    speakers = scan_aishell1(root)
    assert len(speakers) == 3
    for spk_id, utts in speakers.items():
        assert spk_id.startswith("S")
        assert len(utts) == 4
        assert all(u.dataset == "aishell1" for u in utts)
        assert all(u.num_frames > 0 for u in utts)


def test_scan_aishell1_drops_sparse_speakers(tmp_path: Path) -> None:
    """Speakers with only one utterance must be dropped."""
    root = tmp_path / "sparse"
    split_dir = root / "data_aishell" / "wav" / "train"
    _write_sine_wav(split_dir / "S0001" / "BAC009S0001W0001.wav", 2.0, 220)
    _write_sine_wav(split_dir / "S0002" / "BAC009S0002W0001.wav", 2.0, 440)
    _write_sine_wav(split_dir / "S0002" / "BAC009S0002W0002.wav", 2.0, 450)
    speakers = scan_aishell1(root)
    assert "S0001" not in speakers
    assert "S0002" in speakers


def test_scan_aishell3(tmp_path: Path) -> None:
    root = _build_fake_aishell3(tmp_path / "aishell3")
    speakers = scan_aishell3(root)
    assert len(speakers) == 3
    assert all(k.startswith("SSB") for k in speakers.keys())


def test_merge_speaker_dicts(tmp_path: Path) -> None:
    a1 = _build_fake_aishell1(tmp_path / "a1", num_speakers=2)
    a3 = _build_fake_aishell3(tmp_path / "a3", num_speakers=2)
    merged = merge_speaker_dicts(scan_aishell1(a1), scan_aishell3(a3))
    assert len(merged) == 4  # disjoint S vs SSB prefixes
    assert any(k.startswith("S") for k in merged)
    assert any(k.startswith("SSB") for k in merged)


def test_scan_noise_dir(tmp_path: Path) -> None:
    root = _build_fake_noise_dir(tmp_path / "noise", num_files=5, seconds=3.0)
    noises = scan_noise_dir(root)
    assert len(noises) == 5
    assert all(n.num_frames > 0 for n in noises)


def test_scan_noise_dir_drops_short_files(tmp_path: Path) -> None:
    root = tmp_path / "noise_short"
    root.mkdir()
    # Short file (500 ms) should be dropped under the 1 s default.
    _write_sine_wav(root / "short.wav", 0.5, 100)
    _write_sine_wav(root / "long.wav", 2.0, 200)
    noises = scan_noise_dir(root, min_duration_seconds=1.0)
    assert len(noises) == 1
    assert noises[0].path.name == "long.wav"


# ---------------------------------------------------------------------------
# Augmentation tests
# ---------------------------------------------------------------------------


def test_synth_rir_shape_and_peak() -> None:
    import random as pyrandom
    rng = pyrandom.Random(0)
    cfg = ReverbConfig()
    rir = synth_room_rir(cfg, rng)
    assert rir.ndim == 1
    assert rir.numel() > 0
    # Peak-normalized to 1 during synth
    assert float(rir.abs().max()) == pytest.approx(1.0, abs=1e-5)


def test_apply_rir_preserves_length() -> None:
    import random as pyrandom
    signal = torch.randn(1000)
    rir = synth_room_rir(ReverbConfig(), pyrandom.Random(0))
    y = apply_rir(signal, rir)
    assert y.shape == signal.shape
    assert torch.isfinite(y).all()


def test_add_noise_at_snr_matches_request() -> None:
    """Verify the noise scaling actually produces the requested SNR."""
    torch.manual_seed(0)
    signal = torch.randn(16000)
    noise = torch.randn(16000)
    mixed = add_noise_at_snr(signal, noise, snr_db=10.0)
    # Recover the noise component: mixed - signal (approximately).
    recovered_noise = mixed - signal
    sig_rms = float(torch.sqrt((signal ** 2).mean()))
    noise_rms = float(torch.sqrt((recovered_noise ** 2).mean()))
    measured_snr_db = 20.0 * math.log10(sig_rms / (noise_rms + 1e-12))
    assert abs(measured_snr_db - 10.0) < 0.1


# ---------------------------------------------------------------------------
# Mixer tests
# ---------------------------------------------------------------------------


def _build_mixer(tmp_path: Path, with_noise: bool = True) -> WulfeniteMixer:
    a1 = _build_fake_aishell1(tmp_path / "a1", num_speakers=4,
                              utts_per_speaker=4)
    a3 = _build_fake_aishell3(tmp_path / "a3", num_speakers=4,
                              utts_per_speaker=4)
    speakers = merge_speaker_dicts(scan_aishell1(a1), scan_aishell3(a3))
    noise_pool = None
    if with_noise:
        noise_root = tmp_path / "noise"
        noise_root.mkdir()
        _build_fake_noise_dir(noise_root, num_files=3, seconds=3.0)
        noise_pool = scan_noise_dir(noise_root)
    cfg = MixerConfig(
        segment_seconds=1.0,
        enrollment_seconds=1.0,
        target_present_prob=0.75,
    )
    return WulfeniteMixer(
        speakers=speakers,
        noise_pool=noise_pool,
        config=cfg,
        samples_per_epoch=20,
        seed=7,
    )


def test_mixer_sample_shapes(tmp_path: Path) -> None:
    mixer = _build_mixer(tmp_path)
    sample = mixer[0]
    expected_len = int(1.0 * SR)
    assert sample["mixture"].shape == (expected_len,)
    assert sample["target"].shape == (expected_len,)
    assert sample["enrollment"].shape == (expected_len,)
    assert sample["target_present"].shape == ()
    assert sample["target_speaker_idx"].shape == ()
    assert sample["target_present"].item() in (0.0, 1.0)


def test_mixer_emits_target_speaker_idx(tmp_path: Path) -> None:
    mixer = _build_mixer(tmp_path)
    sample = mixer[0]
    assert "target_speaker_idx" in sample
    assert sample["target_speaker_idx"].dtype == torch.long
    assert 0 <= int(sample["target_speaker_idx"].item()) < len(mixer.speaker_ids)


def test_mixer_present_and_absent_branches_fire(tmp_path: Path) -> None:
    """Over many samples we should see both branches at roughly the configured ratio."""
    mixer = _build_mixer(tmp_path)
    n_present = 0
    n_absent = 0
    for i in range(200):
        s = mixer[i]
        if s["target_present"].item() == 1.0:
            n_present += 1
        else:
            n_absent += 1
    # Configured target_present_prob = 0.75; allow a lot of slack
    # because the seed chain only has 200 samples.
    assert n_present > 100
    assert n_absent > 20


def test_mixer_absent_target_is_zero(tmp_path: Path) -> None:
    """For every target-absent sample, the target tensor must be all zeros."""
    mixer = _build_mixer(tmp_path)
    seen_absent = False
    for i in range(100):
        s = mixer[i]
        if s["target_present"].item() == 0.0:
            seen_absent = True
            assert torch.all(s["target"] == 0), \
                "absent sample has nonzero target tensor"
    assert seen_absent, "No absent samples in 100 draws; seed or prob too skewed"


def test_mixer_finite_values(tmp_path: Path) -> None:
    mixer = _build_mixer(tmp_path)
    for i in range(20):
        s = mixer[i]
        assert torch.isfinite(s["mixture"]).all()
        assert torch.isfinite(s["target"]).all()
        assert torch.isfinite(s["enrollment"]).all()


def test_mixer_collate(tmp_path: Path) -> None:
    mixer = _build_mixer(tmp_path)
    batch = [mixer[i] for i in range(4)]
    collated = collate_mixer_batch(batch)
    assert collated["mixture"].shape == (4, int(1.0 * SR))
    assert collated["target"].shape == (4, int(1.0 * SR))
    assert collated["enrollment"].shape == (4, int(1.0 * SR))
    assert collated["target_present"].shape == (4,)
    assert collated["target_speaker_idx"].shape == (4,)


def test_mixer_without_noise_pool(tmp_path: Path) -> None:
    """Mixer should fall back to synthetic Gaussian noise when no DNS is provided."""
    mixer = _build_mixer(tmp_path, with_noise=False)
    s = mixer[0]
    assert torch.isfinite(s["mixture"]).all()
