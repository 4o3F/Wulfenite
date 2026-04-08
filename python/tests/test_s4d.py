"""Unit tests for the S4D layer.

The most important test is the agreement between the parallel and
step forward modes — they must produce the same output (up to
floating-point precision) because the parallel mode IS the impulse
response of the recurrent form.
"""

from __future__ import annotations

import torch

from wulfenite.models.s4d import S4D


def test_s4d_parallel_shape() -> None:
    torch.manual_seed(0)
    layer = S4D(d_model=8, d_state=16).eval()
    x = torch.randn(2, 50, 8)
    with torch.no_grad():
        y = layer(x)
    assert y.shape == x.shape, f"expected {x.shape}, got {y.shape}"
    assert torch.isfinite(y).all(), "parallel output contains non-finite values"


def test_s4d_step_shape() -> None:
    torch.manual_seed(0)
    layer = S4D(d_model=8, d_state=16).eval()
    state = layer.initial_state(batch_size=2)
    assert state.shape == (2, 8, 8, 2), f"bad state shape: {state.shape}"

    x_t = torch.randn(2, 8)
    with torch.no_grad():
        y_t, new_state = layer.forward_step(x_t, state)
    assert y_t.shape == (2, 8)
    assert new_state.shape == state.shape
    assert torch.isfinite(y_t).all()


def test_s4d_parallel_equals_step() -> None:
    """Parallel conv mode must agree with the sequential recurrent mode."""
    torch.manual_seed(42)
    d_model, d_state, L, B = 4, 16, 32, 2
    layer = S4D(d_model=d_model, d_state=d_state).eval()
    x = torch.randn(B, L, d_model)

    with torch.no_grad():
        y_par = layer(x)  # [B, L, H]

        state = layer.initial_state(B)
        y_seq_list = []
        for t in range(L):
            y_t, state = layer.forward_step(x[:, t], state)
            y_seq_list.append(y_t)
        y_seq = torch.stack(y_seq_list, dim=1)  # [B, L, H]

    diff = (y_par - y_seq).abs().max().item()
    # Allow some slack because the parallel form uses FFT, which has
    # its own rounding behavior distinct from the recurrent evaluation.
    assert diff < 1e-3, f"parallel/step disagreement: {diff:.2e}"
