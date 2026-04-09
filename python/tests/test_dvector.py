"""Tests for the learnable d-vector encoder."""

from __future__ import annotations

import torch

from wulfenite.models.dvector import LearnableDVector, SpecAugment, compute_fbank_batch


def test_dvector_forward_shape() -> None:
    enc = LearnableDVector(num_speakers=100)
    fbank = torch.randn(4, 300, 80)
    raw, norm, logits = enc(fbank)
    assert raw.shape == (4, 192)
    assert norm.shape == (4, 192)
    assert logits is not None
    assert logits.shape == (4, 100)
    assert torch.allclose(norm.norm(dim=-1), torch.ones(4), atol=1e-5)


def test_dvector_no_classifier_in_eval() -> None:
    enc = LearnableDVector(num_speakers=None)
    fbank = torch.randn(2, 200, 80)
    raw, norm, logits = enc(fbank)
    assert raw.shape == (2, 192)
    assert norm.shape == (2, 192)
    assert logits is None


def test_compute_fbank_batch_shape() -> None:
    wav = torch.randn(3, 48000)
    fbank = compute_fbank_batch(wav)
    assert fbank.dim() == 3
    assert fbank.size(0) == 3
    assert fbank.size(2) == 80


def test_dvector_param_count_under_1m() -> None:
    enc = LearnableDVector(num_speakers=618)
    total = sum(p.numel() for p in enc.parameters())
    assert total < 1_000_000, f"Expected <1M params, got {total}"


def test_spec_augment_disabled_in_eval() -> None:
    aug = SpecAugment()
    aug.eval()
    x = torch.randn(2, 100, 80)
    assert torch.equal(aug(x), x)
