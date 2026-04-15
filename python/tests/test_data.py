"""Unit tests for retained dataset and augmentation utilities."""

from __future__ import annotations

from pathlib import Path
import random

import numpy as np
import pytest
import soundfile as sf
import torch

from wulfenite.data import (
    AudioEntry,
    NoiseEntry,
    ReverbConfig,
    add_noise_at_snr,
    apply_rir,
    merge_speaker_dicts,
    scan_aishell1,
    scan_aishell3,
    scan_cnceleb,
    scan_magicdata,
    scan_noise_dir,
    scale_noise_to_snr,
    synth_room_rir,
)
from wulfenite.data.augmentation import _fit_noise_length


SR = 16000


def _write_sine_wav(path: Path, seconds: float, freq: float) -> None:
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


def _build_fake_aishell1(
    root: Path,
    num_speakers: int = 3,
    utts_per_speaker: int = 4,
    seconds: float = 2.0,
) -> Path:
    split_dir = root / "data_aishell" / "wav" / "train"
    for s in range(num_speakers):
        spk_id = f"S{s:04d}"
        for u in range(utts_per_speaker):
            freq = 200 + 50 * s + 5 * u
            _write_sine_wav(
                split_dir / spk_id / f"BAC009{spk_id}W{u:04d}.wav",
                seconds,
                freq,
            )
    return root


def _build_fake_aishell3(
    root: Path,
    num_speakers: int = 3,
    utts_per_speaker: int = 4,
    seconds: float = 2.0,
) -> Path:
    split_dir = root / "train" / "wav"
    for s in range(num_speakers):
        spk_id = f"SSB{s:04d}"
        for u in range(utts_per_speaker):
            freq = 300 + 60 * s + 7 * u
            _write_sine_wav(
                split_dir / spk_id / f"{spk_id}{u:04d}.wav",
                seconds,
                freq,
            )
    return root


def _build_fake_cnceleb(
    root: Path,
    num_speakers: int = 3,
    utts_per_speaker: int = 4,
    seconds: float = 2.0,
) -> Path:
    split_dir = root / "cn-celeb_v2" / "data"
    for s in range(num_speakers):
        spk_id = f"id{s + 1:05d}"
        for u in range(utts_per_speaker):
            freq = 350 + 40 * s + 3 * u
            _write_sine_wav(
                split_dir / spk_id / f"interview-{u + 1:02d}-{u + 1:03d}.wav",
                seconds,
                freq,
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


def _build_fake_noise_dir(
    root: Path,
    num_files: int = 4,
    seconds: float = 3.0,
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for i in range(num_files):
        t = np.arange(int(seconds * SR)) / SR
        rng = np.random.default_rng(i + 42)
        audio = (0.05 * rng.standard_normal(t.shape)).astype(np.float32)
        sf.write(str(root / f"noise_{i:02d}.wav"), audio, SR)
    return root


def test_scan_aishell1(tmp_path: Path) -> None:
    root = _build_fake_aishell1(tmp_path / "aishell1")
    speakers = scan_aishell1(root)
    assert len(speakers) == 3
    for spk_id, utts in speakers.items():
        assert spk_id.startswith("S")
        assert len(utts) == 4
        assert all(u.dataset == "aishell1" for u in utts)
        assert all(u.num_frames > 0 for u in utts)


def test_scan_aishell1_drops_sparse_speakers(tmp_path: Path) -> None:
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


def test_scan_magicdata_accepts_root_without_nested_wav(tmp_path: Path) -> None:
    root = _build_fake_magicdata(tmp_path / "magicdata_flat", nested_wav_dir=False)
    speakers = scan_magicdata(root)
    assert len(speakers) == 3


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
    assert len(merged) == 4
    assert any(k.startswith("S") for k in merged)
    assert any(k.startswith("SSB") for k in merged)


def test_merge_speaker_dicts_concatenates_on_collision() -> None:
    entry_a = AudioEntry(
        speaker_id="S0001",
        path=Path("/tmp/a.wav"),
        num_frames=16000,
        dataset="aishell1",
    )
    entry_b = AudioEntry(
        speaker_id="S0001",
        path=Path("/tmp/b.wav"),
        num_frames=16000,
        dataset="magicdata",
    )
    merged = merge_speaker_dicts({"S0001": [entry_a]}, {"S0001": [entry_b]})
    assert [entry.dataset for entry in merged["S0001"]] == ["aishell1", "magicdata"]


def test_scan_noise_dir(tmp_path: Path) -> None:
    root = _build_fake_noise_dir(tmp_path / "noise", num_files=5, seconds=3.0)
    noises = scan_noise_dir(root)
    assert len(noises) == 5
    assert all(n.num_frames > 0 for n in noises)


def test_scan_noise_dir_drops_short_files(tmp_path: Path) -> None:
    root = tmp_path / "noise_short"
    root.mkdir()
    _write_sine_wav(root / "short.wav", 0.5, 100)
    _write_sine_wav(root / "long.wav", 2.0, 200)
    noises = scan_noise_dir(root, min_duration_seconds=1.0)
    assert len(noises) == 1
    assert noises[0].path.name == "long.wav"


def test_synth_rir_shape_and_peak() -> None:
    rng = random.Random(0)
    cfg = ReverbConfig()
    rir = synth_room_rir(cfg, rng)
    assert rir.ndim == 1
    assert rir.numel() > 0
    assert float(rir.abs().max()) == pytest.approx(1.0, abs=1e-5)


def test_apply_rir_preserves_length() -> None:
    dry = torch.randn(16000)
    rir = torch.tensor([1.0, 0.5, 0.25])
    wet = apply_rir(dry, rir)
    assert wet.shape == dry.shape


def test_apply_rir_identity_when_delta() -> None:
    dry = torch.randn(8000)
    rir = torch.tensor([1.0])
    wet = apply_rir(dry, rir)
    assert torch.allclose(wet, dry, atol=1e-6)


def test_add_noise_at_snr_roughly_hits_target() -> None:
    clean = torch.ones(16000)
    noise = torch.randn(16000)
    mixed = add_noise_at_snr(clean, noise, snr_db=10.0)

    clean_p = clean.pow(2).mean()
    noise_p = (mixed - clean).pow(2).mean()
    snr = 10.0 * torch.log10(clean_p / noise_p)

    assert float(snr) == pytest.approx(10.0, abs=0.8)


def test_scale_noise_to_snr_hits_requested_ratio() -> None:
    clean = torch.ones(16000)
    noise = torch.randn(8000)
    scaled = scale_noise_to_snr(clean, noise, snr_db=6.0, rng=random.Random(0))
    snr = 10.0 * torch.log10(clean.pow(2).mean() / scaled.pow(2).mean())
    assert float(snr) == pytest.approx(6.0, abs=0.8)


def test_scale_noise_to_snr_handles_silent_reference() -> None:
    clean = torch.zeros(16000)
    noise = torch.randn(16000)
    scaled = scale_noise_to_snr(clean, noise, snr_db=5.0, rng=random.Random(0))
    assert torch.allclose(scaled, torch.zeros_like(clean))


def test_add_noise_at_snr_handles_zero_noise() -> None:
    clean = torch.ones(16000)
    noise = torch.zeros(16000)
    mixed = add_noise_at_snr(clean, noise, snr_db=5.0)
    assert torch.allclose(mixed, clean)


def test_reverb_config_from_preset_produces_valid_ranges() -> None:
    small = ReverbConfig.from_preset("small")
    medium = ReverbConfig.from_preset("medium")
    large = ReverbConfig.from_preset("large")
    mixed = ReverbConfig.from_preset("mixed")
    assert small.rt60_range == (0.08, 0.25)
    assert medium.rt60_range == (0.20, 0.50)
    assert large.rt60_range == (0.45, 0.80)
    assert mixed.rt60_range == (0.08, 0.80)


def test_fit_noise_length_loops_short_noise() -> None:
    noise = torch.tensor([1.0, 2.0])
    fitted = _fit_noise_length(noise, 5, random.Random(0))
    assert torch.equal(fitted, torch.tensor([1.0, 2.0, 1.0, 2.0, 1.0]))


def test_audio_entries_keep_absolute_paths(tmp_path: Path) -> None:
    root = _build_fake_aishell1(tmp_path / "aishell1", num_speakers=1,
                                utts_per_speaker=2)
    entry = next(iter(scan_aishell1(root).values()))[0]
    assert entry.path.is_absolute()


def test_noise_entries_keep_absolute_paths(tmp_path: Path) -> None:
    root = _build_fake_noise_dir(tmp_path / "noise", num_files=1)
    entry = scan_noise_dir(root)[0]
    assert entry.path.is_absolute()


def test_audio_entry_repr_is_stable(tmp_path: Path) -> None:
    root = _build_fake_aishell1(tmp_path / "aishell1", num_speakers=1,
                                utts_per_speaker=2)
    entry = next(iter(scan_aishell1(root).values()))[0]
    text = repr(entry)
    assert "AudioEntry" in text
    assert entry.speaker_id in text


def test_noise_entry_repr_is_stable(tmp_path: Path) -> None:
    root = _build_fake_noise_dir(tmp_path / "noise", num_files=1)
    entry = scan_noise_dir(root)[0]
    text = repr(entry)
    assert "NoiseEntry" in text
    assert entry.path.name in text
