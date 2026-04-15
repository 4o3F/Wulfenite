"""Unit tests for optional evaluation metrics."""

from __future__ import annotations

import math

import torch

import wulfenite.evaluation.metrics as metrics_module
from wulfenite.evaluation import evaluate_pair, si_sdr


def test_si_sdr_identity_is_high() -> None:
    target = torch.randn(2, 16000)
    score = si_sdr(target, target)
    assert torch.all(score > 60.0)


def test_si_sdr_noise_is_low() -> None:
    torch.manual_seed(0)
    target = torch.randn(2, 16000)
    estimate = torch.randn(2, 16000)
    score = si_sdr(estimate, target)
    assert torch.all(score < 5.0)


def test_si_sdr_ignores_dc_offset() -> None:
    target = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    estimate = target + 100.0
    score = si_sdr(estimate, target)
    assert torch.all(score > 60.0)


def test_evaluate_pair_returns_expected_keys() -> None:
    target = torch.randn(16000)
    scores = evaluate_pair(target, target, on_missing="nan")
    assert set(scores) == {"si_sdr", "pesq", "stoi"}


def test_optional_metrics_nan_when_missing(monkeypatch) -> None:
    def _raise_import_error(module_name: str) -> object:
        raise ImportError(module_name)

    monkeypatch.setattr(metrics_module, "_import_optional", _raise_import_error)
    target = torch.randn(16000)
    scores = evaluate_pair(target, target, on_missing="nan")
    assert math.isnan(scores["pesq"])
    assert math.isnan(scores["stoi"])
