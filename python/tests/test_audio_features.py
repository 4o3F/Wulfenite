"""Tests for shared audio feature extraction helpers."""

from __future__ import annotations

import torch

from wulfenite.audio_features import compute_fbank_batch


def test_compute_fbank_batch_shape() -> None:
    wav = torch.randn(3, 48000)
    fbank = compute_fbank_batch(wav)
    assert fbank.dim() == 3
    assert fbank.size(0) == 3
    assert fbank.size(2) == 80


def test_compute_fbank_batch_pads_to_longest_item() -> None:
    wav = torch.randn(2, 48000)
    wav[1, 20000:] = 0.0

    fbank = compute_fbank_batch(wav)

    assert fbank.size(0) == 2
    assert fbank.size(1) >= 1
    assert fbank[1].shape == fbank[0].shape
