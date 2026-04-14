"""Streaming-friendly layers shared by the SpeakerBeam-SS separator.

Every layer in this module supports two forward modes:

- ``forward(x)`` — whole-sequence mode for training, takes the whole
  input sequence and returns the whole output sequence.
- ``forward_step(x, state)`` — streaming mode for inference, takes one
  or more new timesteps plus the carry-state from the previous call,
  returns the new output slice and the updated state.

Both modes must be numerically equivalent on the same input; the
streaming unit tests assert this invariant.

The modules are kept deliberately small. ONNX export paths use the
streaming form so the exported graph is stateful and frame-oriented
(see ``docs/onnx_contract.md``).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


# ---------------------------------------------------------------------------
# Causal 1D convolution
# ---------------------------------------------------------------------------


class CausalConv1d(nn.Module):
    """1D convolution with configurable right context.

    Given kernel size ``k`` and dilation ``d``, the layer uses a total
    padding budget of ``(k - 1) * d`` samples. By default that entire
    budget is assigned to the left side, making the layer strictly
    causal. When ``right_context > 0``, some of the padding budget moves
    to the right side so the whole-sequence path can consult bounded
    future context.

    Streaming state always stores the last ``pad_total = (k - 1) * d``
    input samples of the previous call as a ``[B, C, pad_total]``
    tensor. The streaming caller is responsible for model-level
    alignment when ``right_context > 0`` because the first
    ``right_context`` outputs are startup transients.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        right_context: int = 0,
    ) -> None:
        super().__init__()
        if stride != 1:
            # A strided causal conv is well-defined but complicates the
            # streaming state shape. We do not need stride > 1 anywhere
            # in SpeakerBeam-SS's post-encoder path, so we forbid it
            # here to avoid shipping subtly wrong behavior.
            raise ValueError("CausalConv1d does not support stride > 1")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.pad_total = (kernel_size - 1) * dilation
        if not 0 <= right_context <= self.pad_total:
            raise ValueError(
                "right_context must be between 0 and "
                f"{self.pad_total}; got {right_context}"
            )
        self.right_context = right_context
        self.pad_left = self.pad_total - right_context
        self.pad_right = right_context
        # Keep the legacy attribute name for backward compatibility with
        # tests and any external callers.
        self.pad_len = self.pad_total
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Whole-sequence mode.

        Args:
            x: ``[B, in_channels, T]``.

        Returns:
            ``[B, out_channels, T]`` (same time length — the total
            padding budget keeps the output aligned with the input).
        """
        if self.pad_total > 0:
            x = F.pad(x, (self.pad_left, self.pad_right))
        return self.conv(x)

    def zero_state(self, batch_size: int, device: torch.device,
                   dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Allocate a fresh zero streaming state.

        The state shape is ``[B, in_channels, pad_len]``. If ``pad_len``
        is 0 (kernel_size == 1) the returned tensor has ``T = 0``.
        """
        return torch.zeros(
            batch_size, self.in_channels, self.pad_len,
            device=device, dtype=dtype,
        )

    def forward_step(
        self,
        x_new: torch.Tensor,
        state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Streaming mode.

        Args:
            x_new: ``[B, in_channels, T_new]`` — the new samples arriving
                since the last call. ``T_new`` can be any positive value.
            state: ``[B, in_channels, pad_len]`` from the previous call
                (or ``zero_state(...)`` on the first call).

        Returns:
            Tuple of:
            - ``[B, out_channels, T_new]`` output covering exactly the
              new samples. When ``right_context > 0`` the leading outputs
              are startup transients that the caller must align away.
            - updated state ``[B, in_channels, pad_len]`` to feed into
              the next call.
        """
        if self.pad_len == 0:
            # Kernel size 1, no state needed.
            return self.conv(x_new), state
        # Prepend the cached left-context, convolve, emit only the new
        # portion. The conv consumes pad_len + T_new samples and
        # produces T_new outputs.
        x_full = torch.cat([state, x_new], dim=-1)
        y = self.conv(x_full)
        new_state = x_full[..., -self.pad_len:]
        return y, new_state


# ---------------------------------------------------------------------------
# Channel-wise LayerNorm (cLN from Conv-TasNet)
# ---------------------------------------------------------------------------


class ChannelwiseLayerNorm(nn.Module):
    """Channel-wise Layer Normalization from Luo & Mesgarani (Conv-TasNet).

    Normalizes across the channel dimension at each time step, so every
    timestep has zero mean and unit variance across channels. Strictly
    causal (depends only on the current timestep) and therefore safe
    for streaming.

    This is NOT the same as Conv-TasNet's global LayerNorm (gLN), which
    normalizes over time+channel and is not causal. SpeakerBeam-SS uses
    cLN throughout because it targets real-time streaming.
    """

    def __init__(self, num_channels: int, eps: float = 1e-8) -> None:
        super().__init__()
        self.num_channels = num_channels
        # Delegate to torch's native LayerNorm over the channel dim.
        self.norm = nn.LayerNorm(num_channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply cLN.

        Args:
            x: ``[B, C, T]``.

        Returns:
            ``[B, C, T]``, same shape, normalized per-timestep across
            channels.
        """
        # LayerNorm expects the normalized dim to be last.
        return self.norm(x.transpose(1, 2)).transpose(1, 2)

    # cLN has no temporal dependency, so its streaming state is empty.
    # We still provide the method for API uniformity with causal convs.
    def zero_state(self, *args, **kwargs) -> None:
        return None

    def forward_step(
        self,
        x_new: torch.Tensor,
        state: None,
    ) -> tuple[torch.Tensor, None]:
        return self.forward(x_new), None
