"""Tests for the training pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

from wulfenite.data import MixerConfig, WulfeniteMixer, scan_aishell1
from wulfenite.losses import LossWeights, MultiResolutionSTFTLoss, WulfeniteLoss
from wulfenite.models import SpeakerBeamSSConfig, WulfeniteTSE
from wulfenite.training.checkpoint import load_checkpoint, save_checkpoint
from wulfenite.training.config import TrainingConfig
from wulfenite.training.train import (
    _should_update_best_checkpoint,
    build_dataset,
    build_loss,
    build_optimizer,
    compute_checkpoint_score,
    compute_enrollment_shuffle_sdr_drop,
    effective_transition_prob,
    run_training,
    train_one_epoch,
    validate,
)


SR = 16000


def _write_sine(path: Path, seconds: float, freq: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.arange(int(seconds * SR)) / SR
    audio = (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    sf.write(str(path), audio, SR)


def _build_aishell1_tree(
    root: Path,
    num_speakers: int = 4,
    utts_per_speaker: int = 3,
    seconds: float = 2.0,
) -> Path:
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
    return SpeakerBeamSSConfig(
        enc_channels=16,
        bottleneck_channels=16,
        speaker_embed_dim=192,
        r1_repeats=1,
        r2_repeats=1,
        conv_blocks_per_repeat=1,
        hidden_channels=16,
        s4d_state_dim=8,
    )


def _small_tse(device: torch.device | str = "cpu") -> WulfeniteTSE:
    return WulfeniteTSE.from_campplus(
        None,
        separator_config=_small_separator_config(),
    ).to(device)


def _small_loss() -> WulfeniteLoss:
    return WulfeniteLoss(
        weights=LossWeights(
            sdr=1.0,
            mr_stft=0.5,
            absent=1.0,
            presence=0.1,
            inactive=0.25,
            route=0.5,
            overlap_route=0.25,
        ),
        mr_stft_loss=MultiResolutionSTFTLoss(
            fft_sizes=(256,), hop_sizes=(64,), win_lengths=(256,),
        ),
    )


def _small_mixer(
    tmp_path: Path,
    samples: int = 8,
    target_present_prob: float = 0.75,
    *,
    composition_mode: str = "legacy_branch",
    segment_seconds: float = 1.0,
) -> WulfeniteMixer:
    root = _build_aishell1_tree(tmp_path / "aishell1")
    speakers = scan_aishell1(root)
    cfg = MixerConfig(
        segment_seconds=segment_seconds,
        enrollment_seconds=segment_seconds,
        composition_mode=composition_mode,
        target_present_prob=target_present_prob,
        transition_prob=0.0,
        apply_reverb=False,
        apply_noise=False,
    )
    return WulfeniteMixer(
        speakers=speakers,
        noise_pool=None,
        config=cfg,
        samples_per_epoch=samples,
        seed=7,
    )


def test_training_config_defaults() -> None:
    cfg = TrainingConfig()
    assert cfg.batch_size > 0
    assert cfg.segment_seconds == 8.0
    assert cfg.enrollment_seconds == 4.0
    assert cfg.composition_mode == "clip_composer"
    assert cfg.crossfade_ms == pytest.approx(5.0)
    assert cfg.optional_third_speaker_prob == pytest.approx(0.35)
    assert cfg.gain_drift_db_range == pytest.approx((-1.5, 1.5))
    assert cfg.scene_target_only_min_seconds == pytest.approx(0.8)
    assert cfg.scene_nontarget_only_min_seconds == pytest.approx(0.8)
    assert cfg.scene_overlap_min_seconds == pytest.approx(0.4)
    assert cfg.scene_background_min_seconds == pytest.approx(0.3)
    assert cfg.scene_absence_before_return_min_seconds == pytest.approx(1.0)
    assert cfg.loss_sdr == 1.0
    assert cfg.loss_inactive == pytest.approx(0.05)
    assert cfg.loss_route == pytest.approx(0.15)
    assert cfg.loss_overlap_route == pytest.approx(0.05)
    assert cfg.loss_ae == pytest.approx(0.10)
    assert cfg.inactive_threshold == pytest.approx(0.05)
    assert cfg.inactive_topk_fraction == pytest.approx(0.25)
    assert cfg.learning_rate == pytest.approx(5e-4)
    assert cfg.encoder_lr == pytest.approx(3e-5)
    assert cfg.speaker_modulation_lr_scale == pytest.approx(2.0)
    assert cfg.separator_frontend_lr_scale == pytest.approx(0.5)
    assert cfg.absent_warmup_epochs == 15
    assert cfg.inactive_warmup_epochs == 15
    assert cfg.route_warmup_epochs == 20
    assert cfg.overlap_route_warmup_epochs == 20
    assert cfg.ae_warmup_epochs == 2
    assert cfg.speaker_embed_dim == 192
    assert cfg.mask_activation == "scaled_sigmoid"
    assert cfg.transition_prob == pytest.approx(0.0)
    assert cfg.transition_warmup_ratio == pytest.approx(0.0)
    assert cfg.transition_ramp_ratio == pytest.approx(0.0)
    assert cfg.transition_min_fraction == pytest.approx(0.25)
    assert cfg.transition_min_target_rms == pytest.approx(0.01)
    assert cfg.loss_recall == pytest.approx(0.20)
    assert cfg.recall_floor == pytest.approx(0.3)
    assert cfg.recall_frame_size == 320
    assert cfg.route_frame_size == 160
    assert cfg.route_margin == pytest.approx(0.05)
    assert cfg.overlap_margin == pytest.approx(0.02)
    assert cfg.overlap_dominance_margin == pytest.approx(0.02)
    assert cfg.use_plateau_scheduler is True
    assert cfg.plateau_patience == 5
    assert cfg.early_stopping_patience == 20
    assert not hasattr(cfg, "loss_speaker_cls")


def test_effective_transition_prob_warmup_and_ramp() -> None:
    cfg = TrainingConfig(
        epochs=100,
        transition_prob=0.10,
        transition_warmup_ratio=0.5,
        transition_ramp_ratio=0.3,
    )
    assert effective_transition_prob(0, cfg) == 0.0
    assert effective_transition_prob(49, cfg) == 0.0
    assert effective_transition_prob(50, cfg) == pytest.approx(0.0)
    assert effective_transition_prob(65, cfg) == pytest.approx(0.05)
    assert effective_transition_prob(80, cfg) == pytest.approx(0.10)
    assert effective_transition_prob(99, cfg) == pytest.approx(0.10)


def test_effective_transition_prob_no_curriculum() -> None:
    cfg = TrainingConfig(
        epochs=100,
        transition_prob=0.10,
        transition_warmup_ratio=0.0,
        transition_ramp_ratio=0.0,
    )
    assert effective_transition_prob(0, cfg) == pytest.approx(0.10)
    assert effective_transition_prob(50, cfg) == pytest.approx(0.10)


def test_training_config_rejects_invalid_curriculum_ratios() -> None:
    with pytest.raises(ValueError):
        TrainingConfig(transition_warmup_ratio=0.7, transition_ramp_ratio=0.5)
    with pytest.raises(ValueError):
        TrainingConfig(transition_warmup_ratio=-0.1)


def test_build_dataset_disables_transitions_for_validation(tmp_path: Path) -> None:
    aishell_root = _build_aishell1_tree(
        tmp_path / "aishell1_cfg", num_speakers=4, utts_per_speaker=3, seconds=2.0,
    )
    cfg = TrainingConfig(
        aishell1_root=aishell_root,
        segment_seconds=1.0,
        enrollment_seconds=1.0,
        samples_per_epoch=4,
        val_samples=2,
        num_workers=0,
        device="cpu",
        composition_mode="legacy_branch",
        transition_prob=0.35,
        transition_min_target_rms=0.02,
        noise_prob=0.0,
        reverb_prob=0.0,
    )

    train_ds, val_ds = build_dataset(cfg)
    assert train_ds.cfg.transition_prob == pytest.approx(0.35)
    assert train_ds.cfg.transition_min_target_rms == pytest.approx(0.02)
    assert val_ds.cfg.transition_prob == pytest.approx(0.0)
    assert val_ds.cfg.transition_min_target_rms == pytest.approx(0.02)


def test_checkpoint_roundtrip(tmp_path: Path) -> None:
    torch.manual_seed(0)
    model = _small_tse()
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=5e-4,
    )
    cfg = TrainingConfig(
        aishell1_root=Path("/fake/path"),
        out_dir=tmp_path / "ckpts",
    )

    ckpt_path = tmp_path / "roundtrip.pt"
    save_checkpoint(
        ckpt_path,
        model=model,
        optimizer=optimizer,
        scheduler=None,
        epoch=3,
        step=42,
        config=cfg,
        metrics={"train_loss": 0.123, "val_loss": 0.456},
    )
    assert ckpt_path.exists()

    model2 = _small_tse()
    info = load_checkpoint(ckpt_path, model=model2)
    assert info["epoch"] == 3
    assert info["step"] == 42
    assert info["metrics"]["train_loss"] == 0.123
    assert isinstance(info["config"]["aishell1_root"], str)
    assert info["config"]["aishell1_root"] == "/fake/path"
    for p1, p2 in zip(model.parameters(), model2.parameters()):
        assert torch.allclose(p1, p2)


def test_checkpoint_optimizer_state_preserved(tmp_path: Path) -> None:
    torch.manual_seed(1)
    model = _small_tse()
    opt = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=5e-4,
    )

    for p in model.parameters():
        if p.requires_grad:
            if p.grad is None:
                p.grad = torch.zeros_like(p)
            p.grad.normal_()
    opt.step()

    ckpt_path = tmp_path / "opt.pt"
    save_checkpoint(ckpt_path, model=model, optimizer=opt, config=TrainingConfig())

    model2 = _small_tse()
    opt2 = torch.optim.Adam(
        [p for p in model2.parameters() if p.requires_grad], lr=5e-4,
    )
    load_checkpoint(ckpt_path, model=model2, optimizer=opt2)
    assert len(opt2.state) > 0


def test_training_single_step_backward(tmp_path: Path) -> None:
    torch.manual_seed(2)
    mixer = _small_mixer(tmp_path, samples=4)
    model = _small_tse()
    criterion = _small_loss()
    opt = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=5e-4,
    )

    from wulfenite.data import collate_mixer_batch
    batch = collate_mixer_batch([mixer[i] for i in range(2)])

    outputs = model(batch["mixture"], batch["enrollment"])
    loss, _ = criterion(
        clean=outputs["clean"],
        target=batch["target"],
        mixture=batch["mixture"],
        target_present=batch["target_present"],
        presence_logit=outputs.get("presence_logit"),
    )
    assert torch.isfinite(loss)
    loss.backward()
    any_grad = any(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in model.parameters() if p.requires_grad
    )
    assert any_grad
    opt.step()


def test_run_training_one_epoch_writes_checkpoint(tmp_path: Path) -> None:
    torch.manual_seed(3)
    aishell_root = _build_aishell1_tree(
        tmp_path / "aishell1", num_speakers=4, utts_per_speaker=3, seconds=2.0,
    )

    cfg = TrainingConfig(
        aishell1_root=aishell_root,
        aishell3_root=None,
        noise_root=None,
        segment_seconds=1.0,
        enrollment_seconds=1.0,
        composition_mode="legacy_branch",
        batch_size=2,
        epochs=1,
        samples_per_epoch=6,
        val_samples=4,
        learning_rate=5e-4,
        encoder_lr=1e-5,
        num_workers=0,
        out_dir=tmp_path / "ckpts",
        log_interval=1,
        device="cpu",
        seed=0,
        noise_prob=0.0,
        reverb_prob=0.0,
    )

    model = _small_tse()
    run_training(cfg, model=model, show_progress=False)

    assert (cfg.out_dir / "epoch001.pt").exists()
    assert (cfg.out_dir / "best.pt").exists()
    assert (cfg.out_dir / "train.log").exists()

    log_text = (cfg.out_dir / "train.log").read_text()
    assert "epoch 1" in log_text
    assert "val_sdr_db=" in log_text
    assert "val_sdri_db=" in log_text
    assert "speaker_cls" not in log_text
    assert "[pretrain]" not in log_text


def test_validate_runs(tmp_path: Path) -> None:
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
    assert val_loss == val_loss
    assert "sdr_db" in parts
    assert "sdri_db" in parts


def test_enrollment_shuffle_sdr_drop_present_mode(tmp_path: Path) -> None:
    torch.manual_seed(31)
    mixer = _small_mixer(tmp_path, samples=4, target_present_prob=1.0)
    from wulfenite.data import collate_mixer_batch

    batch = collate_mixer_batch([mixer[i] for i in range(2)])
    model = _small_tse()
    drop = compute_enrollment_shuffle_sdr_drop(model, batch, torch.device("cpu"))
    assert isinstance(drop, float)


def test_build_optimizer_uses_four_param_groups() -> None:
    model = _small_tse()
    cfg = TrainingConfig()

    optimizer, scheduler = build_optimizer(model, cfg)

    assert isinstance(optimizer, torch.optim.Adam)
    assert isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)
    assert [group["name"] for group in optimizer.param_groups] == [
        "encoder_backbone",
        "separator_frontend",
        "separator_speaker_modulation",
        "separator_rest",
    ]
    assert optimizer.param_groups[0]["lr"] == pytest.approx(cfg.encoder_lr)
    assert optimizer.param_groups[1]["lr"] == pytest.approx(
        cfg.learning_rate * cfg.separator_frontend_lr_scale
    )
    assert optimizer.param_groups[2]["lr"] == pytest.approx(
        cfg.learning_rate * cfg.speaker_modulation_lr_scale
    )
    assert optimizer.param_groups[3]["lr"] == pytest.approx(cfg.learning_rate)


def test_build_dataset_clip_composer_requires_three_speakers_per_split(
    tmp_path: Path,
) -> None:
    aishell_root = _build_aishell1_tree(
        tmp_path / "aishell1_clip_small",
        num_speakers=5,
        utts_per_speaker=3,
        seconds=2.0,
    )
    cfg = TrainingConfig(
        aishell1_root=aishell_root,
        composition_mode="clip_composer",
        samples_per_epoch=4,
        val_samples=2,
        num_workers=0,
        device="cpu",
    )

    with pytest.raises(RuntimeError, match="at least 6 speakers"):
        build_dataset(cfg)


def test_compute_checkpoint_score_ignores_wrong_enrollment_metric() -> None:
    cfg = TrainingConfig()
    metrics = {
        "sdri_db": 5.0,
        "other_only_energy_true": 0.2,
        "wrong_enrollment_leakage": 0.1,
        "overlap_energy_wrong": 0.3,
    }
    shifted = {
        **metrics,
        "wrong_enrollment_leakage": 0.9,
    }

    assert compute_checkpoint_score(metrics, cfg) == pytest.approx(
        compute_checkpoint_score(shifted, cfg)
    )


def test_build_loss_matches_config_weights() -> None:
    cfg = TrainingConfig(
        loss_sdr=0.7,
        loss_mr_stft=0.8,
        loss_absent=0.9,
        loss_presence=0.05,
        loss_recall=0.6,
        loss_inactive=0.4,
        loss_route=0.3,
        loss_overlap_route=0.2,
        recall_floor=0.4,
        recall_frame_size=160,
        inactive_threshold=0.07,
        inactive_topk_fraction=0.5,
    )

    loss = build_loss(cfg)

    assert loss.weights == LossWeights(
        sdr=0.7,
        mr_stft=0.8,
        absent=0.9,
        presence=0.05,
        recall=0.6,
        inactive=0.4,
        route=0.3,
        overlap_route=0.2,
        ae=cfg.loss_ae,
    )
    assert loss.recall_floor == pytest.approx(0.4)
    assert loss.recall_frame_size == 160
    assert loss.inactive_threshold == pytest.approx(0.07)
    assert loss.inactive_topk_fraction == pytest.approx(0.5)


def test_train_one_epoch_runs(tmp_path: Path) -> None:
    torch.manual_seed(5)
    mixer = _small_mixer(tmp_path, samples=4)
    model = _small_tse()
    criterion = _small_loss()
    optimizer, _ = build_optimizer(model, TrainingConfig())

    from torch.utils.data import DataLoader
    from wulfenite.data import collate_mixer_batch

    loader = DataLoader(
        mixer, batch_size=2, collate_fn=collate_mixer_batch, num_workers=0,
    )
    loss, global_step = train_one_epoch(
        model,
        loader,
        criterion,
        optimizer,
        torch.device("cpu"),
        TrainingConfig(log_interval=1),
        epoch=1,
        global_step=0,
        log_fn=lambda _: None,
        show_progress=False,
    )

    assert isinstance(loss, float)
    assert global_step == len(loader)


def test_training_step_with_composer_labels(tmp_path: Path) -> None:
    torch.manual_seed(6)
    mixer = _small_mixer(
        tmp_path,
        samples=2,
        composition_mode="clip_composer",
        segment_seconds=8.0,
    )
    model = _small_tse()
    criterion = _small_loss()

    from wulfenite.data import collate_mixer_batch

    batch = collate_mixer_batch([mixer[i] for i in range(2)])
    outputs = model(batch["mixture"], batch["enrollment"], batch["enrollment_fbank"])
    loss, _ = criterion(
        clean=outputs["clean"],
        target=batch["target"],
        mixture=batch["mixture"],
        target_present=batch["target_present"],
        presence_logit=outputs.get("presence_logit"),
        target_active_frames=batch["target_active_frames"],
        nontarget_active_frames=batch["nontarget_active_frames"],
        overlap_frames=batch["overlap_frames"],
        background_frames=batch["background_frames"],
        scene_id=batch["scene_id"],
        view_role_id=batch["view_role_id"],
    )
    assert torch.isfinite(loss)


def test_checkpoint_lexicographic_selection() -> None:
    assert _should_update_best_checkpoint(
        1.20,
        0.50,
        best_sdri=1.00,
        best_inactive=0.20,
    )
    assert _should_update_best_checkpoint(
        1.03,
        0.10,
        best_sdri=1.00,
        best_inactive=0.20,
    )
    assert not _should_update_best_checkpoint(
        1.03,
        0.30,
        best_sdri=1.00,
        best_inactive=0.20,
    )
