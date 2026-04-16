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
    scan_noise_dirs,
    scale_noise_to_snr,
    synth_room_rir,
)
from wulfenite.data.augmentation import (
    _estimate_signal_rms,
    _fit_noise_length,
    apply_bandwidth_limit,
    apply_random_gain,
)
from wulfenite.scripts.train_pdfnet2 import (
    _build_train_config,
    _build_mixer_kwargs,
    _parse_bucket_table,
    _scan_dataset_splits,
    _scan_noise_inputs,
)


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


def test_scan_noise_dirs_returns_per_category_entries(tmp_path: Path) -> None:
    room_root = _build_fake_noise_dir(tmp_path / "room", num_files=2)
    keyboard_root = _build_fake_noise_dir(tmp_path / "keyboard", num_files=3)
    categorized = scan_noise_dirs({
        "room": room_root,
        "keyboard": keyboard_root,
    })
    assert sorted(categorized) == ["keyboard", "room"]
    assert len(categorized["room"]) == 2
    assert len(categorized["keyboard"]) == 3


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


def test_estimate_signal_rms_active_ignores_silent_tails() -> None:
    signal = torch.cat([torch.ones(8000), torch.zeros(8000)])
    full_rms = _estimate_signal_rms(signal, mode="full")
    active_rms = _estimate_signal_rms(
        signal,
        mode="active",
        frame_samples=512,
        threshold_db=-20.0,
    )
    assert active_rms > full_rms
    assert active_rms > 0.95


def test_estimate_signal_rms_active_falls_back_to_full_when_all_frames_inactive() -> None:
    signal = torch.zeros(16000)
    full_rms = _estimate_signal_rms(signal, mode="full")
    active_rms = _estimate_signal_rms(
        signal,
        mode="active",
        frame_samples=512,
        threshold_db=-20.0,
    )
    assert active_rms == pytest.approx(full_rms, abs=1e-6)


def test_scale_noise_to_snr_active_mode_hits_requested_ratio_on_active_frames() -> None:
    clean = torch.cat([torch.ones(8000), torch.zeros(8000)])
    noise = torch.randn(16000)
    scaled = scale_noise_to_snr(
        clean,
        noise,
        snr_db=6.0,
        rng=random.Random(0),
        rms_mode="active",
        activity_frame_samples=512,
        activity_threshold_db=-20.0,
    )
    clean_rms = _estimate_signal_rms(
        clean,
        mode="active",
        frame_samples=512,
        threshold_db=-20.0,
    )
    noise_rms = _estimate_signal_rms(
        scaled,
        mode="active",
        frame_samples=512,
        threshold_db=-20.0,
    )
    snr = 20.0 * np.log10(clean_rms / noise_rms)
    assert snr == pytest.approx(6.0, abs=0.8)


def test_add_noise_at_snr_handles_zero_noise() -> None:
    clean = torch.ones(16000)
    noise = torch.zeros(16000)
    mixed = add_noise_at_snr(clean, noise, snr_db=5.0)
    assert torch.allclose(mixed, clean)


def test_apply_random_gain_uses_requested_db_range() -> None:
    signal = torch.ones(8)
    gained = apply_random_gain(signal, gain_range_db=(6.0, 6.0), rng=random.Random(0))
    assert torch.allclose(gained, torch.full_like(signal, 10.0 ** (6.0 / 20.0)))


def test_apply_bandwidth_limit_preserves_low_freq_and_attenuates_high_freq() -> None:
    duration_seconds = 1.0
    t = torch.arange(int(SR * duration_seconds), dtype=torch.float32) / SR
    low = torch.sin(2.0 * torch.pi * 1000.0 * t)
    high = torch.sin(2.0 * torch.pi * 6000.0 * t)

    filtered_low = apply_bandwidth_limit(
        low,
        sample_rate=SR,
        cutoff_range_hz=(4000.0, 4000.0),
        rng=random.Random(0),
    )
    filtered_high = apply_bandwidth_limit(
        high,
        sample_rate=SR,
        cutoff_range_hz=(4000.0, 4000.0),
        rng=random.Random(0),
    )

    region = slice(200, -200)
    low_ratio = filtered_low[region].pow(2).mean() / low[region].pow(2).mean()
    high_ratio = filtered_high[region].pow(2).mean() / high[region].pow(2).mean()

    assert filtered_low.shape == low.shape
    assert filtered_high.shape == high.shape
    assert float(low_ratio) > 0.8
    assert float(high_ratio) < 0.2


def test_reverb_config_from_preset_produces_valid_ranges() -> None:
    small = ReverbConfig.from_preset("small")
    medium = ReverbConfig.from_preset("medium")
    large = ReverbConfig.from_preset("large")
    mixed = ReverbConfig.from_preset("mixed")
    assert small.rt60_range == (0.08, 0.20)
    assert medium.rt60_range == (0.20, 0.40)
    assert large.rt60_range == (0.40, 0.65)
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


def test_parse_bucket_table_accepts_toml_like_entries() -> None:
    buckets = _parse_bucket_table([
        {"weight": 0.2, "min_db": -5.0, "max_db": 5.0},
        {"weight": 0.8, "min_db": 5.0, "max_db": 15.0},
    ])
    assert buckets == ((0.2, -5.0, 5.0), (0.8, 5.0, 15.0))


def test_parse_bucket_table_rejects_invalid_entries() -> None:
    with pytest.raises(ValueError):
        _parse_bucket_table([{"weight": 1.0, "min_db": 0.0}])


def test_scan_dataset_splits_returns_per_dataset_maps(tmp_path: Path) -> None:
    aishell1_root = _build_fake_aishell1(tmp_path / "aishell1", num_speakers=2)
    aishell3_root = _build_fake_aishell3(tmp_path / "aishell3", num_speakers=2)
    magicdata_root = _build_fake_magicdata(tmp_path / "magicdata", num_speakers=2)
    train_datasets, val_datasets = _scan_dataset_splits({
        "aishell1_root": str(aishell1_root),
        "aishell3_root": str(aishell3_root),
        "magicdata_root": str(magicdata_root),
    })
    assert sorted(train_datasets) == ["aishell1", "aishell3", "magicdata"]
    assert val_datasets == {}


def test_scan_noise_inputs_prefers_category_roots_when_present(tmp_path: Path) -> None:
    room_root = _build_fake_noise_dir(tmp_path / "room", num_files=1)
    keyboard_root = _build_fake_noise_dir(tmp_path / "keyboard", num_files=1)
    noises = _scan_noise_inputs({
        "noise_root": str(tmp_path / "unused"),
        "noise": {
            "category_roots": {
                "room": str(room_root),
                "keyboard": str(keyboard_root),
            },
            "min_duration_seconds": 1.0,
        },
    })
    assert isinstance(noises, dict)
    assert sorted(noises) == ["keyboard", "room"]


def test_build_mixer_kwargs_reads_new_sections() -> None:
    kwargs = _build_mixer_kwargs(
        {
            "sampling": {
                "dataset_weights": {"magicdata": 0.5, "aishell1": 0.5},
                "interferer_same_dataset_probability": 0.5,
            },
            "scene": {
                "weights": {
                    "target_only_degraded": 0.1,
                    "noise": 0.2,
                    "interference": 0.25,
                    "both": 0.45,
                },
                "snr_buckets": [{"weight": 1.0, "min_db": -5.0, "max_db": 5.0}],
                "sir_buckets": [{"weight": 1.0, "min_db": 0.0, "max_db": 10.0}],
            },
            "reverb": {"room_family_weights": {"small": 1.0, "medium": 0.0, "large": 0.0}},
            "augmentation": {
                "gain_probability": 0.4,
                "gain_range_db": [-3.0, 3.0],
                "bandwidth_limit_probability": 0.1,
                "bandwidth_cutoff_range_hz": [4500.0, 6500.0],
                "mixing_rms_mode": "active",
                "activity_frame_ms": 24.0,
                "activity_threshold_db": -35.0,
            },
            "noise": {"category_weights": {"room": 0.7, "keyboard": 0.3}},
        },
        epoch_size=10,
        segment_length=8000,
        enrollment_length=12000,
        sample_rate=16000,
        reverb_config=ReverbConfig(),
        reverb_probability=0.3,
        seed=42,
    )
    assert kwargs["dataset_weights"] == {"magicdata": 0.5, "aishell1": 0.5}
    assert kwargs["interferer_same_dataset_probability"] == pytest.approx(0.5)
    assert kwargs["scene_weights"] == {
        "target_only_degraded": 0.1,
        "noise": 0.2,
        "interference": 0.25,
        "both": 0.45,
    }
    assert kwargs["snr_buckets"] == ((1.0, -5.0, 5.0),)
    assert kwargs["sir_buckets"] == ((1.0, 0.0, 10.0),)
    assert kwargs["gain_range_db"] == (-3.0, 3.0)
    assert kwargs["bandwidth_cutoff_range_hz"] == (4500.0, 6500.0)
    assert kwargs["mixing_rms_mode"] == "active"
    assert kwargs["noise_category_weights"] == {"room": 0.7, "keyboard": 0.3}


def test_build_train_config_supports_batch_ramp_fields() -> None:
    config = _build_train_config({
        "batch_size_start": 8,
        "batch_size_end": 128,
        "batch_size_ramp_epochs": 20,
        "learning_rate": 5e-4,
        "weight_decay": 0.05,
        "grad_clip_norm": 1.0,
        "max_epochs": 100,
        "patience": 15,
        "lr_warmup_epochs": 3,
        "lr_warmup_start": 1e-4,
    })
    assert config.batch_size_start == 8
    assert config.batch_size_end == 128
    assert config.batch_size_ramp_epochs == 20
    assert config.learning_rate == pytest.approx(5e-4)
    assert config.weight_decay == pytest.approx(0.05)
    assert config.grad_clip_norm == pytest.approx(1.0)
    assert config.max_epochs == 100
    assert config.patience == 15
    assert config.lr_warmup_epochs == 3
    assert config.lr_warmup_start == pytest.approx(1e-4)


def test_build_train_config_keeps_legacy_single_batch_size_behavior() -> None:
    config = _build_train_config({"batch_size": 8})
    assert config.batch_size_start == 8
    assert config.batch_size_end == 8
    assert config.batch_size_ramp_epochs == 1
