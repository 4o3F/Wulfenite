"""Unit tests for causal layers.

Key invariants:
- ``CausalConv1d.forward_step`` is equivalent to splitting the input
  and feeding chunks with state passed through — this is the
  correctness guarantee for streaming inference.
- ``ChannelwiseLayerNorm`` is scale/shift invariant per timestep.
"""

from __future__ import annotations

import torch

from wulfenite.models.layers import CausalConv1d, ChannelwiseLayerNorm


def test_causal_conv_whole_vs_chunked() -> None:
    torch.manual_seed(0)
    conv = CausalConv1d(
        in_channels=4, out_channels=6, kernel_size=5, dilation=2
    ).eval()

    x = torch.randn(1, 4, 40)

    with torch.no_grad():
        y_whole = conv(x)  # [1, 6, 40]

        # Now feed the same input in 4 chunks of 10 samples each and
        # pass the state through.
        chunks = list(x.split(10, dim=-1))
        state = conv.zero_state(batch_size=1, device=x.device)
        y_chunked_list = []
        for chunk in chunks:
            y_chunk, state = conv.forward_step(chunk, state)
            y_chunked_list.append(y_chunk)
        y_chunked = torch.cat(y_chunked_list, dim=-1)

    diff = (y_whole - y_chunked).abs().max().item()
    assert diff < 1e-5, f"whole vs chunked causal conv disagreement: {diff:.2e}"


def test_causal_conv_is_causal() -> None:
    """Modifying input at time t must not change output at times < t."""
    torch.manual_seed(1)
    conv = CausalConv1d(in_channels=2, out_channels=2, kernel_size=3).eval()
    with torch.no_grad():
        x = torch.randn(1, 2, 10)
        y1 = conv(x)

        x2 = x.clone()
        x2[..., 7:] = 999.0  # change everything from t=7 onwards
        y2 = conv(x2)

        # Outputs before t=7 must match exactly.
        assert torch.allclose(y1[..., :7], y2[..., :7]), \
            "CausalConv1d leaked future information into past outputs"


def test_cln_shape() -> None:
    norm = ChannelwiseLayerNorm(num_channels=8)
    x = torch.randn(2, 8, 25)
    y = norm(x)
    assert y.shape == x.shape
    # Per-timestep mean/var over channels should be ~0 and ~1 respectively.
    mean = y.mean(dim=1)
    var = y.var(dim=1, unbiased=False)
    assert mean.abs().max().item() < 1e-5
    assert (var - 1.0).abs().max().item() < 1e-3
