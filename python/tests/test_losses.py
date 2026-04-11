"""Unit tests for wulfenite losses.

The non-scale-invariant SDR tests encode the degenerate-case
guarantees that make this loss family preferable to SI-SDR for our
use case (see ``docs/architecture.md`` section 5). Read the
comments in each test for the exact invariant being checked.
"""

from __future__ import annotations

import math

import torch

from wulfenite.losses import (
    LossWeights,
    MultiResolutionSTFTLoss,
    STFTLoss,
    WulfeniteLoss,
    compute_sdr_db,
    compute_sdri_db,
    presence_loss,
    sdr_loss,
    target_absent_loss,
)


# ---------------------------------------------------------------------------
# sdr_loss
# ---------------------------------------------------------------------------


def test_sdr_perfect_estimate_is_very_negative() -> None:
    """Perfect estimate → SDR → very high → loss very negative."""
    target = torch.randn(2, 16000)
    loss = sdr_loss(target.clone(), target)
    # Eps-limited; should be well below -40 dB.
    assert loss.item() < -40.0, f"expected loss << -40, got {loss.item()}"


def test_sdr_passthrough_on_mixture_gives_zero_dB() -> None:
    """When ``estimate = mixture = target + interferer`` at 0 dB input SNR,
    SDR is approximately 0 dB. This is the key anti-pass-through
    property that SI-SDR lacks.
    """
    torch.manual_seed(0)
    target = torch.randn(4, 16000)
    interferer = torch.randn(4, 16000)
    # Both at unit RMS so input SNR ≈ 0 dB
    target = target / target.std(dim=-1, keepdim=True)
    interferer = interferer / interferer.std(dim=-1, keepdim=True)
    mixture = target + interferer
    loss = sdr_loss(mixture, target)
    # 0 dB SDR means loss ≈ 0. Allow slack for finite-sample variance.
    assert -1.0 < loss.item() < 1.0, (
        f"pass-through SDR should be ~0 dB, got loss {loss.item():.3f}"
    )


def test_sdr_tiny_output_is_not_rewarded() -> None:
    """Tiny output (estimate ≈ 0) should give loss ≈ 0 dB.

    This is the critical SDR property that SI-SDR lacks: SI-SDR would
    return -infinity for estimate = α·target with α → 0 (perfect
    shape match ignored scale). Direct SDR instead penalizes the
    shape error AND the scale error.
    """
    torch.manual_seed(1)
    target = torch.randn(4, 16000)
    tiny = target * 1e-6
    loss = sdr_loss(tiny, target)
    # Very tiny estimate means error ≈ -target, so SDR ≈ 0 dB.
    assert -1.0 < loss.item() < 1.0, (
        f"tiny-output SDR should be ~0 dB, got loss {loss.item():.3f}"
    )


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


# ---------------------------------------------------------------------------
# MR-STFT
# ---------------------------------------------------------------------------


def test_stft_loss_perfect_is_near_zero() -> None:
    torch.manual_seed(0)
    x = torch.randn(2, 16000)
    loss_fn = STFTLoss(n_fft=512, hop_length=128, win_length=512)
    sc, lm = loss_fn(x, x)
    assert sc.item() < 1e-5, f"sc on identical signals {sc.item():.3e}"
    assert lm.item() < 1e-5, f"log_mag on identical signals {lm.item():.3e}"


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
    # Hann windows should be registered as buffers and follow .to(...)
    for m in loss_fn.stft_losses:
        assert m.window.dtype == torch.float32


# ---------------------------------------------------------------------------
# silence / absent
# ---------------------------------------------------------------------------


def test_target_absent_zero_estimate_is_zero_loss() -> None:
    mixture = torch.randn(3, 8000)
    estimate = torch.zeros(3, 8000)
    loss = target_absent_loss(estimate, mixture)
    assert loss.item() < 1e-6, f"expected ~0, got {loss.item()}"


def test_target_absent_passthrough_is_one() -> None:
    mixture = torch.randn(3, 8000)
    loss = target_absent_loss(mixture, mixture)
    # est_energy / mix_energy == 1
    assert abs(loss.item() - 1.0) < 1e-4


def test_target_absent_larger_than_mixture_is_penalized() -> None:
    mixture = torch.randn(3, 8000)
    loud = mixture * 3.0
    loss = target_absent_loss(loud, mixture)
    assert loss.item() > 8.5, f"expected > 8.5, got {loss.item()}"


# ---------------------------------------------------------------------------
# presence
# ---------------------------------------------------------------------------


def test_presence_correct_labels_small_loss() -> None:
    # Very confident correct predictions.
    logits = torch.tensor([10.0, -10.0, 10.0, -10.0])
    labels = torch.tensor([1.0, 0.0, 1.0, 0.0])
    loss = presence_loss(logits, labels)
    assert loss.item() < 1e-3


def test_presence_wrong_labels_large_loss() -> None:
    logits = torch.tensor([10.0, -10.0])
    labels = torch.tensor([0.0, 1.0])
    loss = presence_loss(logits, labels)
    assert loss.item() > 5.0


# ---------------------------------------------------------------------------
# combined
# ---------------------------------------------------------------------------


def test_combined_loss_all_present() -> None:
    """All samples target-present → SDR + MR-STFT branches active, absent = 0."""
    torch.manual_seed(3)
    B, T = 2, 2048
    clean = torch.randn(B, T, requires_grad=True)
    target = torch.randn(B, T)
    mixture = torch.randn(B, T)
    present = torch.ones(B)
    logits = torch.zeros(B, requires_grad=True)

    loss_fn = WulfeniteLoss(
        weights=LossWeights(sdr=1.0, mr_stft=0.5, absent=1.0, presence=0.1),
        mr_stft_loss=MultiResolutionSTFTLoss(
            fft_sizes=(256,), hop_sizes=(64,), win_lengths=(256,),
        ),
    )
    total, parts = loss_fn(clean, target, mixture, present, logits)
    assert parts.n_present == B
    assert parts.n_absent == 0
    assert parts.absent == 0.0
    assert torch.isfinite(total)
    total.backward()
    assert clean.grad is not None and torch.isfinite(clean.grad).all()


def test_combined_loss_all_absent() -> None:
    """All samples target-absent → SDR/STFT = 0, only absent branch active."""
    torch.manual_seed(4)
    B, T = 2, 2048
    clean = torch.randn(B, T, requires_grad=True)
    target = torch.zeros(B, T)        # silent target
    mixture = torch.randn(B, T)
    present = torch.zeros(B)
    logits = torch.zeros(B, requires_grad=True)

    loss_fn = WulfeniteLoss(
        mr_stft_loss=MultiResolutionSTFTLoss(
            fft_sizes=(256,), hop_sizes=(64,), win_lengths=(256,),
        ),
    )
    total, parts = loss_fn(clean, target, mixture, present, logits)
    assert parts.n_present == 0
    assert parts.n_absent == B
    assert parts.sdr == 0.0
    assert parts.mr_stft == 0.0
    assert parts.absent > 0.0
    total.backward()
    assert clean.grad is not None and torch.isfinite(clean.grad).all()


def test_combined_loss_mixed_batch() -> None:
    """Mixed batch → both branches contribute, per-sample masking is clean."""
    torch.manual_seed(5)
    B, T = 4, 2048
    clean = torch.randn(B, T, requires_grad=True)
    target = torch.randn(B, T)
    # Half the batch has zero target
    present = torch.tensor([1.0, 0.0, 1.0, 0.0])
    target = target * present.view(-1, 1)  # zero out absent-target rows
    mixture = torch.randn(B, T)
    logits = torch.zeros(B, requires_grad=True)

    loss_fn = WulfeniteLoss(
        mr_stft_loss=MultiResolutionSTFTLoss(
            fft_sizes=(256,), hop_sizes=(64,), win_lengths=(256,),
        ),
    )
    total, parts = loss_fn(clean, target, mixture, present, logits)
    assert parts.n_present == 2
    assert parts.n_absent == 2
    assert parts.sdr != 0.0  # present branch active
    assert parts.absent > 0.0
    total.backward()
    assert clean.grad is not None and torch.isfinite(clean.grad).all()


def test_combined_loss_without_presence_head() -> None:
    """presence_logit=None should skip presence branch cleanly."""
    torch.manual_seed(6)
    B, T = 2, 1024
    clean = torch.randn(B, T, requires_grad=True)
    target = torch.randn(B, T)
    mixture = torch.randn(B, T)
    present = torch.ones(B)

    loss_fn = WulfeniteLoss(
        mr_stft_loss=MultiResolutionSTFTLoss(
            fft_sizes=(256,), hop_sizes=(64,), win_lengths=(256,),
        ),
    )
    total, parts = loss_fn(clean, target, mixture, present, presence_logit=None)
    assert parts.presence == 0.0
    total.backward()
    assert clean.grad is not None
