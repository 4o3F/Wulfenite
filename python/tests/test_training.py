"""Tests for the training pipeline.

All tests use synthetic wav fixtures written to pytest's ``tmp_path``
so they do not depend on the real AISHELL / MUSAN / CAM++ weights
being present. Each test either builds a tiny model with a
randomly-initialized CAM++ instance (no checkpoint) or exercises the
checkpoint save/load utilities in isolation.

The ``test_run_training_one_epoch`` test is the end-to-end smoke
test: it builds a mini mixer, a mini separator, runs a single
epoch through ``run_training``, and verifies the loss is finite and
a checkpoint was written.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

from wulfenite.data import MixerConfig, WulfeniteMixer, merge_speaker_dicts, scan_aishell1
from wulfenite.losses import LossWeights, MultiResolutionSTFTLoss, WulfeniteLoss
from wulfenite.models import CAMPPlus, SpeakerBeamSS, SpeakerBeamSSConfig, WulfeniteTSE
from wulfenite.training.checkpoint import load_checkpoint, save_checkpoint
from wulfenite.training.config import TrainingConfig
from wulfenite.training.train import (
    build_dataset,
    build_loss,
    build_optimizer,
    compute_enrollment_shuffle_sdr_drop,
    run_training,
    train_one_epoch,
    validate,
)


SR = 16000


# ---------------------------------------------------------------------------
# Synthetic data fixtures (avoid requiring real AISHELL)
# ---------------------------------------------------------------------------


def _write_sine(path: Path, seconds: float, freq: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.arange(int(seconds * SR)) / SR
    audio = (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    sf.write(str(path), audio, SR)


def _build_aishell1_tree(root: Path, num_speakers: int = 4,
                        utts_per_speaker: int = 3,
                        seconds: float = 2.0) -> Path:
    split_dir = root / "data_aishell" / "wav" / "train"
    for s in range(num_speakers):
        spk_id = f"S{s:04d}"
        for u in range(utts_per_speaker):
            freq = 200 + 50 * s + 5 * u
            _write_sine(
                split_dir / spk_id / f"BAC009{spk_id}W{u:04d}.wav",
                seconds, freq,
            )
    return root


def _small_separator_config() -> SpeakerBeamSSConfig:
    """Tiny config so the smoke tests finish in seconds, not minutes."""
    return SpeakerBeamSSConfig(
        enc_channels=16,
        bottleneck_channels=192,  # keep matching CAM++ output dim
        num_repeats=1,
        num_blocks_per_repeat=2,
        hidden_channels=16,
        s4d_state_dim=8,
    )


def _small_tse(device: torch.device | str = "cpu") -> WulfeniteTSE:
    """Build a TSE model with a random-init CAM++ (no checkpoint) and a
    tiny separator.

    Tests never load CAM++ weights — they use random-init CAMPPlus
    because the goal is to exercise the training/inference plumbing,
    not to measure separation quality.
    """
    encoder = CAMPPlus(feat_dim=80, embedding_size=192)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    separator = SpeakerBeamSS(_small_separator_config())
    return WulfeniteTSE(speaker_encoder=encoder, separator=separator).to(device)


def _small_learnable_tse(
    num_speakers: int,
    device: torch.device | str = "cpu",
) -> WulfeniteTSE:
    separator_cfg = _small_separator_config()
    return WulfeniteTSE.from_learnable_dvector(
        num_speakers=num_speakers,
        separator_config=separator_cfg,
        dvector_kwargs={
            "tdnn_channels": 64,
            "stats_dim": 128,
            "spec_augment": False,
        },
    ).to(device)


def _small_loss() -> WulfeniteLoss:
    """Small MR-STFT so the test finishes quickly."""
    return WulfeniteLoss(
        weights=LossWeights(sdr=1.0, mr_stft=0.5, absent=1.0, presence=0.1),
        mr_stft_loss=MultiResolutionSTFTLoss(
            fft_sizes=(256,), hop_sizes=(64,), win_lengths=(256,),
        ),
    )


def _small_mixer(
    tmp_path: Path,
    samples: int = 8,
    target_present_prob: float = 0.75,
) -> WulfeniteMixer:
    root = _build_aishell1_tree(tmp_path / "aishell1")
    speakers = scan_aishell1(root)
    cfg = MixerConfig(
        segment_seconds=1.0,
        enrollment_seconds=1.0,
        target_present_prob=target_present_prob,
        # Disable reverb/noise to keep tests fast and deterministic enough
        apply_reverb=False,
        apply_noise=False,
    )
    return WulfeniteMixer(
        speakers=speakers, noise_pool=None, config=cfg,
        samples_per_epoch=samples, seed=7,
    )


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def test_training_config_defaults() -> None:
    cfg = TrainingConfig()
    assert cfg.batch_size > 0
    assert cfg.segment_seconds == 4.0
    assert cfg.enrollment_seconds == 4.0
    assert cfg.loss_sdr == 1.0
    assert cfg.use_learnable_encoder is False
    assert cfg.loss_speaker_cls == 0.3


# ---------------------------------------------------------------------------
# Checkpoint save / load
# ---------------------------------------------------------------------------


def test_checkpoint_roundtrip(tmp_path: Path) -> None:
    torch.manual_seed(0)
    model = _small_tse()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-3,
    )
    cfg = TrainingConfig(
        aishell1_root=Path("/fake/path"),
        campplus_checkpoint=Path("/fake/campplus.bin"),
        out_dir=tmp_path / "ckpts",
    )

    ckpt_path = tmp_path / "roundtrip.pt"
    save_checkpoint(
        ckpt_path,
        model=model, optimizer=optimizer, scheduler=None,
        epoch=3, step=42, config=cfg,
        metrics={"train_loss": 0.123, "val_loss": 0.456},
    )
    assert ckpt_path.exists()

    # Build a fresh model, load into it
    model2 = _small_tse()
    info = load_checkpoint(ckpt_path, model=model2)
    assert info["epoch"] == 3
    assert info["step"] == 42
    assert info["metrics"]["train_loss"] == 0.123
    # Paths should have been stringified
    assert isinstance(info["config"]["aishell1_root"], str)
    assert info["config"]["aishell1_root"] == "/fake/path"
    # Model weights should match
    for p1, p2 in zip(model.parameters(), model2.parameters()):
        assert torch.allclose(p1, p2)


def test_checkpoint_optimizer_state_preserved(tmp_path: Path) -> None:
    torch.manual_seed(1)
    model = _small_tse()
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-3,
    )

    # Do one step so the optimizer has running state.
    for p in model.parameters():
        if p.requires_grad:
            if p.grad is None:
                p.grad = torch.zeros_like(p)
            p.grad.normal_()
    opt.step()

    ckpt_path = tmp_path / "opt.pt"
    save_checkpoint(ckpt_path, model=model, optimizer=opt, config=TrainingConfig())

    model2 = _small_tse()
    opt2 = torch.optim.AdamW(
        [p for p in model2.parameters() if p.requires_grad], lr=1e-3,
    )
    load_checkpoint(ckpt_path, model=model2, optimizer=opt2)

    # Optimizer internal state should be non-empty after load.
    assert len(opt2.state) > 0


# ---------------------------------------------------------------------------
# Training loop — single step
# ---------------------------------------------------------------------------


def test_training_single_step_backward(tmp_path: Path) -> None:
    """Forward + loss + backward should yield finite gradients."""
    torch.manual_seed(2)
    mixer = _small_mixer(tmp_path, samples=4)
    model = _small_tse()
    criterion = _small_loss()
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-3,
    )

    from wulfenite.data import collate_mixer_batch
    batch = collate_mixer_batch([mixer[i] for i in range(2)])

    outputs = model(batch["mixture"], batch["enrollment"])
    loss, parts = criterion(
        clean=outputs["clean"],
        target=batch["target"],
        mixture=batch["mixture"],
        target_present=batch["target_present"],
        presence_logit=outputs.get("presence_logit"),
    )
    assert torch.isfinite(loss)
    loss.backward()
    # At least one trainable parameter should have a gradient.
    any_grad = any(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in model.parameters() if p.requires_grad
    )
    assert any_grad
    opt.step()


# ---------------------------------------------------------------------------
# Training loop — full epoch via run_training
# ---------------------------------------------------------------------------


def test_run_training_one_epoch_writes_checkpoint(tmp_path: Path) -> None:
    """run_training should complete a tiny epoch and leave checkpoints on disk.

    This is the end-to-end smoke test: data pipeline → model forward →
    loss → backward → optimizer step → validation → checkpoint save.
    """
    torch.manual_seed(3)
    aishell_root = _build_aishell1_tree(
        tmp_path / "aishell1", num_speakers=4, utts_per_speaker=3, seconds=2.0,
    )

    cfg = TrainingConfig(
        aishell1_root=aishell_root,
        aishell3_root=None,
        noise_root=None,
        campplus_checkpoint=Path("/unused/because/model/is/provided"),
        segment_seconds=1.0,
        enrollment_seconds=1.0,
        batch_size=2,
        epochs=1,
        samples_per_epoch=6,
        val_samples=4,
        lr=1e-3,
        warmup_ratio=0.1,
        num_workers=0,  # tests should not spawn workers
        out_dir=tmp_path / "ckpts",
        log_interval=1,
        device="cpu",
        seed=0,
        # Disable augmentation for determinism
        noise_prob=0.0,
        reverb_prob=0.0,
    )

    # Provide a pre-built model so run_training does not try to load
    # CAM++ from disk.
    model = _small_tse()

    # Monkeypatch the mixer config inside build_dataset by giving the
    # data loader a non-augmented mixer. The simplest way is to set
    # the MixerConfig defaults via the TrainingConfig fields the
    # data pipeline reads, which we already did above.
    run_training(cfg, model=model, show_progress=False)

    # Should have written epoch001.pt and best.pt
    assert (cfg.out_dir / "epoch001.pt").exists()
    assert (cfg.out_dir / "best.pt").exists()
    assert (cfg.out_dir / "train.log").exists()

    # Sanity-check the log has the expected epoch summary line.
    log_text = (cfg.out_dir / "train.log").read_text()
    assert "epoch 1" in log_text


def test_run_training_one_epoch_learnable_encoder_writes_checkpoint(
    tmp_path: Path,
) -> None:
    torch.manual_seed(30)
    aishell_root = _build_aishell1_tree(
        tmp_path / "aishell1", num_speakers=4, utts_per_speaker=3, seconds=2.0,
    )

    cfg = TrainingConfig(
        aishell1_root=aishell_root,
        aishell3_root=None,
        noise_root=None,
        campplus_checkpoint=None,
        use_learnable_encoder=True,
        segment_seconds=1.0,
        enrollment_seconds=1.0,
        batch_size=2,
        epochs=1,
        samples_per_epoch=4,
        val_samples=4,
        lr=1e-3,
        warmup_ratio=0.1,
        encoder_pretrain_epochs=1,
        encoder_pretrain_lr=1e-3,
        num_workers=0,
        out_dir=tmp_path / "ckpts_learnable",
        log_interval=1,
        device="cpu",
        seed=0,
        noise_prob=0.0,
        reverb_prob=0.0,
    )

    model = _small_learnable_tse(num_speakers=2)
    run_training(cfg, model=model, show_progress=False)

    assert (cfg.out_dir / "epoch001.pt").exists()
    assert (cfg.out_dir / "best.pt").exists()
    log_text = (cfg.out_dir / "train.log").read_text()
    assert "[pretrain] epoch 1/1" in log_text
    assert "top1=" in log_text
    assert "shuffle_drop=" in log_text


# ---------------------------------------------------------------------------
# Validation helper alone
# ---------------------------------------------------------------------------


def test_validate_runs(tmp_path: Path) -> None:
    """validate() should run a val pass and return a finite number."""
    torch.manual_seed(4)
    mixer = _small_mixer(tmp_path, samples=4)
    from torch.utils.data import DataLoader
    from wulfenite.data import collate_mixer_batch

    loader = DataLoader(
        mixer, batch_size=2, collate_fn=collate_mixer_batch, num_workers=0,
    )
    model = _small_tse()
    criterion = _small_loss()
    device = torch.device("cpu")

    val_loss, parts = validate(model, loader, criterion, device, show_progress=False)
    assert isinstance(val_loss, float)
    assert val_loss == val_loss  # not NaN


def test_enrollment_shuffle_sdr_drop_present_mode(tmp_path: Path) -> None:
    torch.manual_seed(31)
    mixer = _small_mixer(tmp_path, samples=4, target_present_prob=1.0)
    from wulfenite.data import collate_mixer_batch

    batch = collate_mixer_batch([mixer[i] for i in range(2)])
    model = _small_learnable_tse(num_speakers=len(mixer.speaker_ids))
    drop = compute_enrollment_shuffle_sdr_drop(model, batch, torch.device("cpu"))
    assert isinstance(drop, float)
