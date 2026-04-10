"""Tests for the CAM++ speaker-encoder adapter."""

from __future__ import annotations

import torch
from torch import nn

from wulfenite.models.campplus import CAMPPlus
from wulfenite.models.campplus_encoder import CampPlusSpeakerEncoder
from wulfenite.training.config import TrainingConfig


def _small_backbone() -> CAMPPlus:
    return CAMPPlus(
        embedding_size=192,
        growth_rate=8,
        bn_size=2,
        init_channels=32,
        memory_efficient=False,
    )


def test_campplus_frozen_backbone_projection_and_classifier_get_grad() -> None:
    torch.manual_seed(0)
    encoder = CampPlusSpeakerEncoder(
        backbone=_small_backbone(),
        bottleneck_dim=16,
        freeze_backbone=True,
        num_speakers=5,
    )
    encoder.train()
    waveform = torch.randn(2, 8000)

    output = encoder(waveform)
    assert output.speaker_logits is not None
    loss = output.separator_embedding[:, 0].sum() + output.speaker_logits[:, 0].sum()
    loss.backward()

    assert all(p.grad is None for p in encoder.backbone.parameters())
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in encoder.to_separator.parameters()
    )
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in encoder.classifier.parameters()
    )


def test_campplus_finetune_backbone_projection_and_classifier_get_grad() -> None:
    torch.manual_seed(1)
    encoder = CampPlusSpeakerEncoder(
        backbone=_small_backbone(),
        bottleneck_dim=16,
        freeze_backbone=False,
        num_speakers=6,
    )
    encoder.train()
    waveform = torch.randn(2, 8000)

    output = encoder(waveform)
    assert output.speaker_logits is not None
    loss = output.separator_embedding[:, 0].sum() + output.speaker_logits[:, 0].sum()
    loss.backward()

    assert any(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in encoder.backbone.parameters() if p.requires_grad
    )
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in encoder.to_separator.parameters()
    )
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in encoder.classifier.parameters()
    )


def test_campplus_frozen_train_keeps_backbone_in_eval() -> None:
    encoder = CampPlusSpeakerEncoder(
        backbone=_small_backbone(),
        bottleneck_dim=16,
        freeze_backbone=True,
    )
    encoder.train()

    assert encoder.training is True
    assert encoder.backbone.training is False


def test_campplus_forward_output_shape() -> None:
    torch.manual_seed(2)
    encoder = CampPlusSpeakerEncoder(
        backbone=_small_backbone(),
        bottleneck_dim=24,
        freeze_backbone=True,
        num_speakers=7,
    ).eval()
    waveform = torch.randn(2, 8000)

    output = encoder(waveform)

    assert output.separator_embedding.shape == (2, 24)
    assert output.native_embedding.shape == (2, 192)
    assert output.speaker_logits is not None
    assert output.speaker_logits.shape == (2, 7)
    assert isinstance(encoder.to_separator, nn.Sequential)
    assert torch.allclose(
        output.separator_embedding.norm(dim=-1),
        torch.ones(2),
        atol=1e-5,
    )


def test_campplus_forward_without_classifier_returns_none_logits() -> None:
    torch.manual_seed(20)
    encoder = CampPlusSpeakerEncoder(
        backbone=_small_backbone(),
        bottleneck_dim=24,
        freeze_backbone=True,
        num_speakers=None,
    ).eval()
    waveform = torch.randn(2, 8000)

    output = encoder(waveform)

    assert output.speaker_logits is None
    assert encoder.supports_classifier is False
    assert encoder.supports_pretrain is False


def test_campplus_encode_enrollment_matches_forward_shape() -> None:
    torch.manual_seed(3)
    encoder = CampPlusSpeakerEncoder(
        backbone=_small_backbone(),
        bottleneck_dim=20,
        freeze_backbone=True,
    ).eval()
    waveform = torch.randn(2, 8000)

    with torch.no_grad():
        forward_embedding = encoder(waveform).separator_embedding
        enrollment_embedding = encoder.encode_enrollment(waveform)

    assert enrollment_embedding.shape == forward_embedding.shape
    assert torch.allclose(enrollment_embedding, forward_embedding, atol=1e-6)


def test_campplus_forward_preserves_batch_dimension() -> None:
    torch.manual_seed(4)
    encoder = CampPlusSpeakerEncoder(
        backbone=_small_backbone(),
        bottleneck_dim=16,
        freeze_backbone=True,
    ).eval()
    waveform = torch.randn(4, 8000)

    output = encoder(waveform)

    assert output.separator_embedding.shape[0] == 4
    assert output.native_embedding.shape[0] == 4


def test_campplus_linear_projection_type_preserves_legacy_mode() -> None:
    torch.manual_seed(5)
    encoder = CampPlusSpeakerEncoder(
        backbone=_small_backbone(),
        bottleneck_dim=12,
        freeze_backbone=True,
        projection_type="linear",
    ).eval()
    waveform = torch.randn(2, 8000)

    output = encoder(waveform)

    assert isinstance(encoder.to_separator, nn.Linear)
    assert output.separator_embedding.shape == (2, 12)
    assert output.speaker_logits is None


def test_campplus_optimizer_groups_include_classifier_and_scaled_backbone() -> None:
    encoder = CampPlusSpeakerEncoder(
        backbone=_small_backbone(),
        bottleneck_dim=16,
        freeze_backbone=False,
        num_speakers=8,
    )
    cfg = TrainingConfig(learning_rate=1e-3, encoder_lr_scale=0.1)

    groups = encoder.optimizer_groups(cfg, base_lr=2e-4)

    assert [group["name"] for group in groups] == [
        "encoder_backbone",
        "encoder_projection",
        "encoder_classifier",
    ]
    assert groups[0]["lr"] == 2e-5
    assert groups[1]["lr"] == 2e-4
    assert groups[2]["lr"] == 2e-4
    assert all(group["params"] for group in groups)
