"""Tests for the simplified CAM++ speaker encoder."""

from __future__ import annotations

import pytest
import torch

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


def test_campplus_forward_output_shape() -> None:
    torch.manual_seed(0)
    encoder = CampPlusSpeakerEncoder(_small_backbone()).eval()
    waveform = torch.randn(2, 8000)

    raw, norm = encoder(waveform)

    assert raw.shape == (2, 192)
    assert norm.shape == (2, 192)
    assert torch.allclose(
        norm.norm(dim=-1),
        torch.ones(2),
        atol=1e-5,
    )


def test_campplus_forward_accepts_fbank() -> None:
    torch.manual_seed(1)
    encoder = CampPlusSpeakerEncoder(_small_backbone()).eval()
    fbank = torch.randn(2, 120, 80)

    raw, norm = encoder(fbank=fbank)

    assert raw.shape == (2, 192)
    assert norm.shape == (2, 192)


def test_campplus_encode_enrollment_matches_forward() -> None:
    torch.manual_seed(2)
    encoder = CampPlusSpeakerEncoder(_small_backbone()).eval()
    waveform = torch.randn(2, 8000)

    with torch.no_grad():
        _, forward_norm = encoder(waveform)
        enrollment_norm = encoder.encode_enrollment(waveform)

    assert enrollment_norm.shape == forward_norm.shape
    assert torch.allclose(enrollment_norm, forward_norm, atol=1e-6)


def test_campplus_backbone_gets_gradients() -> None:
    torch.manual_seed(3)
    encoder = CampPlusSpeakerEncoder(_small_backbone())
    waveform = torch.randn(2, 8000)

    raw, norm = encoder(waveform)
    loss = raw[:, 0].sum() + norm[:, 0].sum()
    loss.backward()

    assert any(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in encoder.backbone.parameters() if p.requires_grad
    )


def test_campplus_optimizer_groups_use_encoder_lr() -> None:
    encoder = CampPlusSpeakerEncoder(_small_backbone())
    cfg = TrainingConfig(encoder_lr=2e-5)

    groups = encoder.optimizer_groups(cfg)

    assert len(groups) == 1
    assert groups[0]["name"] == "encoder_backbone"
    assert groups[0]["lr"] == pytest.approx(2e-5)
    assert groups[0]["params"]


def test_campplus_optimizer_groups_accept_base_lr_override() -> None:
    encoder = CampPlusSpeakerEncoder(_small_backbone())
    cfg = TrainingConfig(encoder_lr=2e-5)

    groups = encoder.optimizer_groups(cfg, base_lr=7e-6)

    assert groups[0]["lr"] == pytest.approx(7e-6)


def test_campplus_requires_waveform_or_fbank() -> None:
    encoder = CampPlusSpeakerEncoder(_small_backbone())

    with pytest.raises(ValueError, match="Either waveform or fbank"):
        encoder()
