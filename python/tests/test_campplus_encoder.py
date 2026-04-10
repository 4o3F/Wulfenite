"""Tests for the CAM++ speaker-encoder adapter."""

from __future__ import annotations

import torch

from wulfenite.models.campplus import CAMPPlus
from wulfenite.models.campplus_encoder import CampPlusSpeakerEncoder


def _small_backbone() -> CAMPPlus:
    return CAMPPlus(
        embedding_size=192,
        growth_rate=8,
        bn_size=2,
        init_channels=32,
        memory_efficient=False,
    )


def test_campplus_frozen_backbone_projection_gets_grad() -> None:
    torch.manual_seed(0)
    encoder = CampPlusSpeakerEncoder(
        backbone=_small_backbone(),
        bottleneck_dim=16,
        freeze_backbone=True,
    )
    encoder.train()
    waveform = torch.randn(2, 8000)

    output = encoder(waveform)
    loss = output.separator_embedding[:, 0].sum()
    loss.backward()

    assert all(p.grad is None for p in encoder.backbone.parameters())
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in encoder.to_separator.parameters()
    )


def test_campplus_finetune_backbone_and_projection_get_grad() -> None:
    torch.manual_seed(1)
    encoder = CampPlusSpeakerEncoder(
        backbone=_small_backbone(),
        bottleneck_dim=16,
        freeze_backbone=False,
    )
    encoder.train()
    waveform = torch.randn(2, 8000)

    output = encoder(waveform)
    loss = output.separator_embedding[:, 0].sum()
    loss.backward()

    assert any(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in encoder.backbone.parameters() if p.requires_grad
    )
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in encoder.to_separator.parameters()
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
    ).eval()
    waveform = torch.randn(2, 8000)

    output = encoder(waveform)

    assert output.separator_embedding.shape == (2, 24)
    assert output.native_embedding.shape == (2, 192)
    assert output.speaker_logits is None


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
