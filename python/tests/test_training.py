"""Unit tests for training utilities and KD dataset helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

from wulfenite.data import AudioEntry
from wulfenite.models import TinyECAPA
from wulfenite.training import (
    TinyECAPAKDDataset,
    TrainConfig,
    load_tiny_ecapa_checkpoint,
    split_speakers_for_kd,
)
from wulfenite.training.train_pdfnet2 import _build_lr_scheduler


SR = 16000


def _write_ramp_wav(path: Path, start: float, seconds: float) -> AudioEntry:
    path.parent.mkdir(parents=True, exist_ok=True)
    audio = np.linspace(start, start + 1.0, int(seconds * SR), dtype=np.float32)
    sf.write(str(path), audio, SR)
    return AudioEntry(
        speaker_id=path.parent.name,
        path=path,
        num_frames=audio.shape[0],
        dataset="test",
    )


def _build_speakers(tmp_path: Path) -> dict[str, list[AudioEntry]]:
    return {
        "S0001": [
            _write_ramp_wav(tmp_path / "S0001" / "utt1.wav", 0.0, 1.2),
            _write_ramp_wav(tmp_path / "S0001" / "utt2.wav", 1.0, 1.2),
        ],
        "S0002": [
            _write_ramp_wav(tmp_path / "S0002" / "utt1.wav", 2.0, 1.2),
            _write_ramp_wav(tmp_path / "S0002" / "utt2.wav", 3.0, 1.2),
        ],
    }


def test_build_lr_scheduler_cosine_warmup() -> None:
    module = torch.nn.Linear(2, 2)
    optimizer = torch.optim.Adam(module.parameters(), lr=1e-3)
    config = TrainConfig(
        max_epochs=6,
        lr_scheduler="cosine",
        lr_warmup_epochs=2,
        lr_min_ratio=0.1,
    )
    scheduler = _build_lr_scheduler(optimizer, config)
    assert scheduler is not None
    learning_rates = [optimizer.param_groups[0]["lr"]]
    for _ in range(4):
        optimizer.step()
        scheduler.step()
        learning_rates.append(optimizer.param_groups[0]["lr"])
    assert learning_rates[0] < config.learning_rate
    assert learning_rates[1] == config.learning_rate
    assert learning_rates[-1] <= learning_rates[1]
    assert learning_rates[-1] >= config.learning_rate * config.lr_min_ratio


def test_build_lr_scheduler_none_returns_none() -> None:
    module = torch.nn.Linear(2, 2)
    optimizer = torch.optim.Adam(module.parameters(), lr=1e-3)
    config = TrainConfig(lr_scheduler="none")
    assert _build_lr_scheduler(optimizer, config) is None


def test_build_lr_scheduler_clamps_warmup_to_horizon() -> None:
    module = torch.nn.Linear(2, 2)
    optimizer = torch.optim.Adam(module.parameters(), lr=1e-3)
    config = TrainConfig(
        max_epochs=3,
        lr_scheduler="cosine",
        lr_warmup_epochs=5,
        lr_min_ratio=0.1,
    )
    scheduler = _build_lr_scheduler(optimizer, config)
    assert scheduler is not None
    learning_rates = [optimizer.param_groups[0]["lr"]]
    for _ in range(2):
        optimizer.step()
        scheduler.step()
        learning_rates.append(optimizer.param_groups[0]["lr"])
    # Warmup clamped to max_epochs-1=2, so epoch 1 should reach base LR
    assert learning_rates[1] >= config.learning_rate * 0.99


def test_kd_dataset_returns_same_speaker_pairs(tmp_path: Path) -> None:
    speakers = _build_speakers(tmp_path)
    dataset = TinyECAPAKDDataset(speakers, excerpt_length=8000, epoch_size=2, seed=0)
    sample = dataset[0]
    assert sample["speaker_id"] in speakers
    assert sample["student_waveform"].shape == (8000,)
    assert sample["teacher_waveform"].shape == (8000,)


def test_kd_dataset_changes_with_epoch(tmp_path: Path) -> None:
    speakers = {"S0001": _build_speakers(tmp_path)["S0001"]}
    dataset = TinyECAPAKDDataset(speakers, excerpt_length=8000, epoch_size=1, seed=0)
    dataset.set_epoch(0)
    sample_epoch0 = dataset[0]
    dataset.set_epoch(1)
    sample_epoch1 = dataset[0]
    assert not torch.allclose(sample_epoch0["student_waveform"], sample_epoch1["student_waveform"])


def test_split_speakers_for_kd_disjoint(tmp_path: Path) -> None:
    speakers = _build_speakers(tmp_path)
    train_speakers, val_speakers = split_speakers_for_kd(speakers, val_fraction=0.5, seed=0)
    assert set(train_speakers).isdisjoint(val_speakers)
    assert set(train_speakers) | set(val_speakers) == set(speakers)


def test_load_tiny_ecapa_checkpoint_round_trip(tmp_path: Path) -> None:
    model = TinyECAPA()
    checkpoint_path = tmp_path / "tiny_ecapa_best.pt"
    torch.save(
        {
            "student_state_dict": model.state_dict(),
            "model_kwargs": {"sample_rate": model.sample_rate},
        },
        checkpoint_path,
    )
    loaded = load_tiny_ecapa_checkpoint(checkpoint_path)
    assert not loaded.training
    assert loaded.sample_rate == model.sample_rate
    assert all(not param.requires_grad for param in loaded.parameters())
    for name, value in model.state_dict().items():
        assert torch.equal(loaded.state_dict()[name], value)


def test_load_tiny_ecapa_checkpoint_accepts_raw_state_dict(tmp_path: Path) -> None:
    model = TinyECAPA()
    checkpoint_path = tmp_path / "tiny_ecapa_raw.pt"
    torch.save(model.state_dict(), checkpoint_path)
    loaded = load_tiny_ecapa_checkpoint(checkpoint_path)
    assert set(loaded.state_dict()) == set(model.state_dict())


def test_load_tiny_ecapa_checkpoint_rejects_malformed(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "tiny_ecapa_broken.pt"
    torch.save({"wrong_key": {}}, checkpoint_path)
    with pytest.raises(ValueError, match="student_state_dict|raw state dict"):
        load_tiny_ecapa_checkpoint(checkpoint_path)
