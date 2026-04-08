"""Minimal S4D (Diagonal State Space) layer for causal streaming.

Based on Gu et al. 2022, "On the Parameterization and Initialization
of Diagonal State Space Models" (`arXiv:2206.11893`). Initialization
follows the S4D-Lin variant, which matches HiPPO quality with the
simplest reparameterization.

Key design decisions driven by Wulfenite's ONNX export requirement:

1. **Training (conv mode)** uses complex arithmetic + FFT convolution
   over the full sequence — fast and numerically clean but **not**
   ONNX-friendly because of complex dtypes / complex FFT.
2. **Inference (step mode)** uses only real arithmetic: state is
   stored as a real ``[B, H, N/2, 2]`` tensor where the last axis
   holds ``(re, im)`` pairs, and complex multiplication is expanded
   by hand (4 real multiplies). This form exports cleanly to ONNX
   opset ≥ 13 using only Exp / Sin / Cos / Mul / Add / Div / Sum.

Both modes produce numerically equivalent outputs on the same input
(the conv kernel is the exact impulse response of the recurrent form,
so they are mathematically identical up to floating-point rounding).
The agreement is verified by the unit test in ``tests/test_s4d.py``.

The implementation is deliberately narrow: no variants, no
dense-vs-diagonal switching, no dropout-inside-kernel tricks. One
class, two forward methods, and an ``initial_state`` allocator.
"""

from __future__ import annotations

import math

import torch
from torch import nn


class S4D(nn.Module):
    """S4D-Lin layer with parallel (training) and step (inference) modes.

    Args:
        d_model: number of independent SSM channels ``H``. This is the
            bottleneck feature dimension of the enclosing block.
        d_state: full state size ``N``. Internally we keep ``N // 2``
            complex modes — the conjugate pair is implicit and is
            accounted for by a factor of 2 on the output.
        dt_min / dt_max: range for log-uniform initialization of the
            per-channel timestep. Defaults follow the paper
            (``1e-3`` to ``1e-1``).
        dropout: optional output dropout rate. 0 disables.
        transposed: if ``True``, the parallel forward expects inputs
            shaped ``[B, H, L]`` instead of ``[B, L, H]``.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        dt_min: float = 1e-3,
        dt_max: float = 1e-1,
        dropout: float = 0.0,
        transposed: bool = False,
    ) -> None:
        super().__init__()
        if d_state % 2 != 0:
            raise ValueError("d_state must be even (conjugate pairs implicit)")
        self.h = d_model
        self.n = d_state // 2  # number of complex modes
        self.transposed = transposed

        # Per-channel timestep, learnable, log-uniform init.
        log_dt = (
            torch.rand(self.h) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        self.log_dt = nn.Parameter(log_dt)  # [H]

        # S4D-Lin A initialization:
        #   Re(A) = -0.5   (stored as log(0.5), actual value -exp(log_A_real))
        #   Im(A) = pi * n
        # The -exp(...) parameterization guarantees Re(A) < 0 without
        # explicit clamping, keeping the recurrence stable.
        log_A_real = torch.log(0.5 * torch.ones(self.h, self.n))
        A_imag = math.pi * torch.arange(self.n).repeat(self.h, 1).float()
        self.log_A_real = nn.Parameter(log_A_real)  # [H, N/2]
        self.A_imag = nn.Parameter(A_imag)          # [H, N/2]

        # C: complex, stored as (real, imag) for ONNX friendliness.
        # B is fixed to all-ones (absorbed into C — a standard S4D
        # simplification that the paper justifies).
        C = torch.randn(self.h, self.n, 2) / math.sqrt(self.n)
        self.C = nn.Parameter(C)  # [H, N/2, 2]

        # D: real skip connection, per channel.
        self.D = nn.Parameter(torch.randn(self.h))  # [H]

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _A_complex(self) -> torch.Tensor:
        """Return the diagonal A as a complex tensor of shape [H, N/2]."""
        return -torch.exp(self.log_A_real) + 1j * self.A_imag

    def _C_complex(self) -> torch.Tensor:
        """Return C as a complex tensor of shape [H, N/2]."""
        return torch.view_as_complex(self.C)

    def initial_state(
        self,
        batch_size: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Zero state for streaming.

        The returned tensor has shape ``[B, H, N/2, 2]`` — a real tensor
        where the last dimension stores the ``(re, im)`` pair of each
        complex mode.
        """
        return torch.zeros(
            batch_size, self.h, self.n, 2,
            device=device, dtype=dtype,
        )

    # ------------------------------------------------------------------
    # Parallel (training) forward
    # ------------------------------------------------------------------

    def _kernel(self, L: int) -> torch.Tensor:
        """Compute the SSM convolution kernel of length L. Returns [H, L] real.

        Vandermonde form:
            K[h, l] = 2 · Re( Σ_n  C_n · (exp(dt·A_n) - 1) / A_n · exp(dt·A_n)^l )
        The factor of 2 arises from the implicit conjugate half of the
        spectrum; both the parallel and the step form apply it identically.
        """
        dt = torch.exp(self.log_dt)           # [H]
        A = self._A_complex()                 # [H, N/2]
        C = self._C_complex()                 # [H, N/2]

        dtA = A * dt.unsqueeze(-1)            # [H, N/2]
        Cw = C * (torch.exp(dtA) - 1.0) / A   # [H, N/2]

        l = torch.arange(L, device=A.device)
        # exp(dtA) ** l  =  exp(dtA * l) — stable, no large intermediate.
        vand = torch.exp(dtA.unsqueeze(-1) * l)        # [H, N/2, L]
        K = 2.0 * torch.einsum("hn,hnl->hl", Cw, vand).real
        return K

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Whole-sequence forward via FFT convolution.

        Args:
            x: ``[B, L, H]`` by default, or ``[B, H, L]`` if
               ``transposed=True`` was passed to the constructor.

        Returns:
            Same shape as ``x``. Strictly equivalent to running
            ``forward_step`` over each timestep sequentially with
            ``initial_state`` as the starting state.
        """
        if not self.transposed:
            x = x.transpose(-1, -2)  # [B, H, L]
        _, _, L = x.shape

        K = self._kernel(L)  # [H, L]

        # FFT conv with length 2L to avoid circular wrap.
        n = 2 * L
        Kf = torch.fft.rfft(K.to(torch.float32), n=n)   # [H, n/2+1]
        xf = torch.fft.rfft(x.to(torch.float32), n=n)   # [B, H, n/2+1]
        y = torch.fft.irfft(xf * Kf, n=n)[..., :L]      # [B, H, L]

        # Real skip connection.
        y = y + x * self.D.unsqueeze(-1)
        y = self.dropout(y)
        if not self.transposed:
            y = y.transpose(-1, -2)
        return y.to(x.dtype)

    # ------------------------------------------------------------------
    # Recurrent (inference / ONNX) forward step
    # ------------------------------------------------------------------

    def forward_step(
        self,
        x_t: torch.Tensor,
        state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Single-timestep recurrent update, real arithmetic only.

        Args:
            x_t: ``[B, H]`` new input at the current timestep.
            state: ``[B, H, N/2, 2]`` real state tensor from the
                previous call (or ``initial_state(...)`` on the first
                call of the session).

        Returns:
            Tuple of:
            - ``y_t``: ``[B, H]`` output at the current timestep.
            - ``new_state``: ``[B, H, N/2, 2]`` updated state.

        The whole function uses only ``Exp``, ``Sin``, ``Cos``, ``Mul``,
        ``Add``, ``Div``, and ``Sum`` — every op lands in ONNX opset 13
        or earlier, so the exported streaming graph has no blockers.
        """
        dt = torch.exp(self.log_dt)                 # [H]
        a_re = -torch.exp(self.log_A_real)          # [H, N/2]
        a_im = self.A_imag                          # [H, N/2]

        # dtA = dt * A  (broadcast dt over N/2)
        dtA_re = a_re * dt.unsqueeze(-1)
        dtA_im = a_im * dt.unsqueeze(-1)

        # dA = exp(dtA)   (complex exp via Euler)
        exp_re = torch.exp(dtA_re)
        dA_re = exp_re * torch.cos(dtA_im)          # [H, N/2]
        dA_im = exp_re * torch.sin(dtA_im)

        # dB = (dA - 1) / A   (complex division expanded)
        num_re = dA_re - 1.0
        num_im = dA_im
        denom = a_re * a_re + a_im * a_im           # [H, N/2]
        dB_re = (num_re * a_re + num_im * a_im) / denom
        dB_im = (num_im * a_re - num_re * a_im) / denom

        # state ← dA · state + dB · x_t   (complex mul, real form)
        s_re = state[..., 0]
        s_im = state[..., 1]
        ns_re = dA_re * s_re - dA_im * s_im + dB_re * x_t.unsqueeze(-1)
        ns_im = dA_re * s_im + dA_im * s_re + dB_im * x_t.unsqueeze(-1)
        new_state = torch.stack((ns_re, ns_im), dim=-1)

        # y_t = 2 · Re( Σ_n C_n · state_n ) + D · x_t
        c_re = self.C[..., 0]                       # [H, N/2]
        c_im = self.C[..., 1]
        y = 2.0 * (c_re * ns_re - c_im * ns_im).sum(dim=-1)  # [B, H]
        y = y + self.D * x_t
        return y, new_state
