"""Unit tests for the data pipeline.

Tests use synthetic wav fixtures written to a pytest ``tmp_path`` so
they do not depend on the real AISHELL / DNS4 datasets being present.
Fixtures follow the same directory layout the scanners expect, so
the tests exercise the actual scan paths.
"""

from __future__ import annotations

import math
import random
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
    scan_cnceleb,
    scan_magicdata,
    scan_noise_dir,
    synth_room_rir,
)
from wulfenite.data.augmentation import ReverbConfig
from wulfenite.models import compute_fbank_batch


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


def _write_sine_wav_at_sr(
    path: Path,
    seconds: float,
    freq: float,
    sample_rate: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.arange(int(seconds * sample_rate)) / sample_rate
    audio = (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    sf.write(str(path), audio, sample_rate)


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


def _build_fake_cnceleb(root: Path, num_speakers: int = 3,
                        utts_per_speaker: int = 4,
                        seconds: float = 2.0) -> Path:
    split_dir = root / "cn-celeb_v2" / "data"
    for s in range(num_speakers):
        spk_id = f"id{s + 1:05d}"
        for u in range(utts_per_speaker):
            freq = 350 + 40 * s + 3 * u
            _write_sine_wav(
                split_dir / spk_id / f"interview-{u + 1:02d}-{u + 1:03d}.wav",
                seconds, freq,
            )
    return root


def _build_fake_magicdata(
    root: Path,
    num_speakers: int = 3,
    utts_per_speaker: int = 4,
    seconds: float = 2.0,
    nested_wav_dir: bool = False,
) -> Path:
    base = root / "wav" if nested_wav_dir else root
    split_dir = base / "train"
    for s in range(num_speakers):
        spk_id = f"{s:02d}_{1000 + s}"
        for u in range(utts_per_speaker):
            freq = 260 + 35 * s + 4 * u
            _write_sine_wav(
                split_dir / spk_id / f"{spk_id}_{u:04d}.wav",
                seconds,
                freq,
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


def test_scan_magicdata_prefixes_speakers(tmp_path: Path) -> None:
    root = _build_fake_magicdata(tmp_path / "magicdata", nested_wav_dir=True)
    speakers = scan_magicdata(root)
    assert len(speakers) == 3
    assert all(k.startswith("MD_") for k in speakers.keys())
    assert all(u.dataset == "magicdata" for utts in speakers.values() for u in utts)


def test_scan_cnceleb(tmp_path: Path) -> None:
    root = _build_fake_cnceleb(tmp_path / "cnceleb")
    speakers = scan_cnceleb(root)
    assert len(speakers) == 3
    assert all(k.startswith("id") for k in speakers.keys())
    assert all(len(utts) == 4 for utts in speakers.values())
    assert all(u.dataset == "cnceleb" for utts in speakers.values() for u in utts)


def test_scan_cnceleb_reports_rejected_sample_rates(tmp_path: Path) -> None:
    root = tmp_path / "cnceleb_bad"
    wav = root / "cn-celeb_v2" / "data" / "id00001" / "bad.wav"
    _write_sine_wav_at_sr(wav, seconds=1.0, freq=220.0, sample_rate=8000)

    with pytest.raises(RuntimeError) as excinfo:
        scan_cnceleb(root)

    msg = str(excinfo.value)
    assert "CN-Celeb" in msg
    assert "8000" in msg
    assert "wrong sample rate" in msg


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


def _build_mixer(
    tmp_path: Path,
    with_noise: bool = True,
    *,
    transition_prob: float = 0.0,
    composition_mode: str = "legacy_branch",
    segment_seconds: float = 1.0,
) -> WulfeniteMixer:
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
        segment_seconds=segment_seconds,
        enrollment_seconds=segment_seconds,
        composition_mode=composition_mode,
        target_present_prob=0.75,
        transition_prob=transition_prob,
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


def test_mixer_composer_sample_shapes(tmp_path: Path) -> None:
    mixer = _build_mixer(
        tmp_path,
        composition_mode="clip_composer",
        segment_seconds=4.0,
    )
    sample = mixer[0]
    expected_len = int(4.0 * SR)
    expected_frames = expected_len // 160
    assert sample["mixture"].shape == (expected_len,)
    assert sample["target"].shape == (expected_len,)
    assert sample["enrollment"].shape == (expected_len,)
    assert sample["target_active_frames"].shape == (expected_frames,)
    assert sample["nontarget_active_frames"].shape == (expected_frames,)
    assert sample["overlap_frames"].shape == (expected_frames,)
    assert sample["target_active_frames"].dtype == torch.bool


def test_mixer_legacy_mode_backward_compat(tmp_path: Path) -> None:
    mixer = _build_mixer(tmp_path, composition_mode="legacy_branch")
    sample = mixer[0]
    assert "target_active_frames" in sample
    assert "nontarget_active_frames" in sample
    assert "overlap_frames" in sample
    assert sample["target_active_frames"].ndim == 1


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
    assert collated["enrollment_fbank"].shape[0] == 4
    assert collated["target_present"].shape == (4,)
    assert collated["target_speaker_idx"].shape == (4,)


def test_mixer_composer_collate_framewise(tmp_path: Path) -> None:
    mixer = _build_mixer(
        tmp_path,
        composition_mode="clip_composer",
        segment_seconds=4.0,
    )
    batch = [mixer[i] for i in range(2)]
    collated = collate_mixer_batch(batch)
    assert collated["target_active_frames"].shape == (2, int(4.0 * SR) // 160)
    assert collated["nontarget_active_frames"].shape == (2, int(4.0 * SR) // 160)
    assert collated["overlap_frames"].shape == (2, int(4.0 * SR) // 160)


def test_mixer_collate_recomputes_fbank(tmp_path: Path) -> None:
    mixer = _build_mixer(tmp_path)
    batch = [mixer[i] for i in range(4)]
    collated = collate_mixer_batch(batch)

    assert collated["enrollment_fbank"].shape[0] == 4
    expected_fbank = compute_fbank_batch(collated["enrollment"])
    assert torch.allclose(collated["enrollment_fbank"], expected_fbank, atol=1e-6)


def test_mixer_config_fixed_range_compatibility_shim() -> None:
    cfg = MixerConfig(enrollment_seconds_range=(1.0, 1.0))
    assert cfg.enrollment_seconds == 1.0
    assert cfg.enrollment_seconds_range == (1.0, 1.0)


def test_mixer_config_rejects_variable_range() -> None:
    with pytest.raises(ValueError, match="fixed endpoints"):
        MixerConfig(enrollment_seconds_range=(1.0, 2.0))


def test_mixer_without_noise_pool(tmp_path: Path) -> None:
    """Mixer should fall back to synthetic Gaussian noise when no DNS is provided."""
    mixer = _build_mixer(tmp_path, with_noise=False)
    s = mixer[0]
    assert torch.isfinite(s["mixture"]).all()


@pytest.mark.parametrize(
    ("roll", "expect_prefix_zero"),
    [
        (0.05, True),   # absent_to_present
        (0.85, False),  # present_to_absent
    ],
)
def test_mixer_transition_sample_gates_target(
    tmp_path: Path,
    roll: float,
    expect_prefix_zero: bool,
) -> None:
    class FixedRandom(random.Random):
        def __init__(self, fixed_roll: float) -> None:
            super().__init__(0)
            self.fixed_roll = fixed_roll

        def random(self) -> float:
            return self.fixed_roll

    a1 = _build_fake_aishell1(tmp_path / "a1t", num_speakers=4, utts_per_speaker=4)
    a3 = _build_fake_aishell3(tmp_path / "a3t", num_speakers=4, utts_per_speaker=4)
    speakers = merge_speaker_dicts(scan_aishell1(a1), scan_aishell3(a3))
    mixer = WulfeniteMixer(
        speakers=speakers,
        noise_pool=None,
        config=MixerConfig(
            segment_seconds=1.0,
            enrollment_seconds=1.0,
            composition_mode="legacy_branch",
            transition_prob=1.0,
            transition_min_fraction=0.25,
            apply_reverb=False,
            apply_noise=False,
        ),
        samples_per_epoch=4,
        seed=7,
    )

    sample = mixer._make_transition_sample(FixedRandom(roll))
    target = sample["target"]
    assert sample["mixture"].shape == (SR,)
    assert target.shape == (SR,)
    assert sample["target_present"].item() == 1.0
    assert torch.any(target != 0.0)

    nonzero = torch.nonzero(target.abs() > 1e-7, as_tuple=False).squeeze(-1)
    assert nonzero.numel() > 0
    first = int(nonzero[0].item())
    last = int(nonzero[-1].item())
    min_len = int(0.25 * SR)

    if expect_prefix_zero:
        assert first >= min_len
        assert torch.allclose(target[:first], torch.zeros_like(target[:first]))
    else:
        assert SR - 1 - last >= min_len
        assert torch.allclose(target[last + 1:], torch.zeros_like(target[last + 1:]))


def test_mixer_transition_falls_back_on_low_energy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FixedRandom(random.Random):
        def random(self) -> float:
            return 0.05  # absent_to_present

    a1 = _build_fake_aishell1(tmp_path / "a1g", num_speakers=4, utts_per_speaker=4)
    a3 = _build_fake_aishell3(tmp_path / "a3g", num_speakers=4, utts_per_speaker=4)
    speakers = merge_speaker_dicts(scan_aishell1(a1), scan_aishell3(a3))
    mixer = WulfeniteMixer(
        speakers=speakers,
        noise_pool=None,
        config=MixerConfig(
            segment_seconds=1.0,
            enrollment_seconds=1.0,
            composition_mode="legacy_branch",
            transition_prob=1.0,
            transition_min_fraction=0.25,
            transition_min_target_rms=0.01,
            apply_reverb=False,
            apply_noise=False,
        ),
        samples_per_epoch=4,
        seed=7,
    )

    cutoff = int(0.25 * SR)
    target = torch.zeros(SR, dtype=torch.float32)
    target[:cutoff] = 0.1
    interferer = torch.full((SR,), 0.1, dtype=torch.float32)
    enrollment = torch.full((SR,), 0.1, dtype=torch.float32)

    def fake_prepare_present_sources(
        rng: random.Random,
    ) -> tuple[str, torch.Tensor, torch.Tensor, torch.Tensor]:
        return mixer.speaker_ids[0], target.clone(), interferer.clone(), enrollment.clone()

    seen_rng: list[random.Random] = []

    def fake_make_present_sample(rng: random.Random) -> dict:
        seen_rng.append(rng)
        return {
            "mixture": torch.ones(SR, dtype=torch.float32),
            "target": torch.ones(SR, dtype=torch.float32),
            "enrollment": torch.ones(SR, dtype=torch.float32),
            "target_present": torch.tensor(1.0, dtype=torch.float32),
            "target_speaker_idx": torch.tensor(0, dtype=torch.long),
            "snr_db": torch.tensor(0.0, dtype=torch.float32),
        }

    monkeypatch.setattr(mixer, "_prepare_present_sources", fake_prepare_present_sources)
    monkeypatch.setattr(mixer, "_make_present_sample", fake_make_present_sample)

    rng = FixedRandom(0)
    sample = mixer._make_transition_sample(rng)

    assert seen_rng == [rng]
    assert torch.allclose(sample["target"], torch.ones(SR))
    assert torch.allclose(sample["mixture"], torch.ones(SR))
    assert sample["target_present"].item() == 1.0
