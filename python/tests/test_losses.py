"""Unit tests for retained generic enhancement losses."""

from __future__ import annotations

import pytest
import torch

from wulfenite.losses import (
    MultiResolutionLoss,
    MultiResolutionSTFTLoss,
    OverSuppressionLoss,
    PDfNet2Loss,
    SpectralLoss,
    STFTLoss,
    compute_sdr_db,
    compute_sdri_db,
    sdr_loss,
)


def test_sdr_perfect_estimate_is_very_negative() -> None:
    target = torch.randn(2, 16000)
    loss = sdr_loss(target.clone(), target)
    assert loss.item() < -40.0


def test_sdr_passthrough_on_mixture_gives_zero_db() -> None:
    torch.manual_seed(0)
    target = torch.randn(4, 16000)
    interferer = torch.randn(4, 16000)
    target = target / target.std(dim=-1, keepdim=True)
    interferer = interferer / interferer.std(dim=-1, keepdim=True)
    mixture = target + interferer
    loss = sdr_loss(mixture, target)
    assert -1.0 < loss.item() < 1.0


def test_sdr_tiny_output_is_not_rewarded() -> None:
    torch.manual_seed(1)
    target = torch.randn(4, 16000)
    tiny = target * 1e-6
    loss = sdr_loss(tiny, target)
    assert -1.0 < loss.item() < 1.0


def test_sdr_per_sample_reduction() -> None:
    target = torch.randn(3, 8000)
    estimate = torch.randn(3, 8000)
    per_sample = sdr_loss(estimate, target, reduction="none")
    assert per_sample.shape == (3,)
    mean = sdr_loss(estimate, target, reduction="mean")
    assert torch.allclose(per_sample.mean(), mean)


def test_compute_sdr_db_matches_negative_loss() -> None:
    target = torch.randn(2, 8000)
    estimate = torch.randn(2, 8000)
    sdr_db = compute_sdr_db(estimate, target, reduction="mean")
    loss = sdr_loss(estimate, target, reduction="mean")
    assert torch.allclose(sdr_db, -loss)


def test_compute_sdri_db_zero_for_passthrough() -> None:
    torch.manual_seed(2)
    target = torch.randn(2, 8000)
    interferer = torch.randn(2, 8000)
    mixture = target + interferer
    sdri = compute_sdri_db(mixture, target, mixture, reduction="mean")
    assert abs(float(sdri.item())) < 1e-5


def test_stft_loss_perfect_is_near_zero() -> None:
    torch.manual_seed(0)
    x = torch.randn(2, 16000)
    loss_fn = STFTLoss(n_fft=512, hop_length=128, win_length=512)
    sc, lm = loss_fn(x, x)
    assert sc.item() < 1e-5
    assert lm.item() < 1e-5


def test_stft_sc_clamp() -> None:
    torch.manual_seed(1)
    good = torch.randn(16000)
    loud = torch.randn(16000) * 10.0
    zero = torch.zeros(16000)

    estimate = torch.stack([good, loud], dim=0)
    target = torch.stack([good, zero], dim=0)

    loss_fn = STFTLoss(n_fft=512, hop_length=128, win_length=512)
    sc, _ = loss_fn(estimate, target)

    assert torch.isfinite(sc)
    assert 2.4 <= sc.item() <= 2.6


def test_mr_stft_loss_runs_and_is_positive() -> None:
    torch.manual_seed(2)
    est = torch.randn(2, 16000)
    tgt = torch.randn(2, 16000)
    loss_fn = MultiResolutionSTFTLoss()
    loss = loss_fn(est, tgt)
    assert loss.item() > 0.0
    assert torch.isfinite(loss)


def test_mr_stft_buffers_move_with_device() -> None:
    loss_fn = MultiResolutionSTFTLoss()
    for module in loss_fn.stft_losses:
        assert module.window.dtype == torch.float32


def test_spectral_loss_perfect_is_near_zero() -> None:
    spec = torch.view_as_real(torch.randn(2, 8, 161, dtype=torch.complex64))
    loss = SpectralLoss()(spec, spec)
    assert loss.item() < 1e-6


def test_multi_resolution_loss_perfect_is_zero() -> None:
    wav = torch.randn(2, 16000)
    loss = MultiResolutionLoss()(wav, wav)
    assert loss.item() < 1e-6


def test_over_suppression_loss_penalizes_quiet_estimate() -> None:
    target = torch.view_as_real(torch.randn(2, 8, 161, dtype=torch.complex64))
    estimate = torch.zeros_like(target)
    loss = OverSuppressionLoss()(estimate, target)
    assert loss.item() > 0.0


def test_pdfnet2_loss_returns_components() -> None:
    wav = torch.randn(2, 16000)
    loss_fn = PDfNet2Loss()
    total, terms = loss_fn(wav, wav)
    assert total.item() < 1e-4
    assert set(terms) == {"spectral", "multi_res", "over_suppression"}
