"""Shared neural network modules for the pDFNet2 family."""

from __future__ import annotations

import math
from typing import Callable, cast

import torch
from torch import nn


class GroupedLinear(nn.Module):
    """Linear projection with independent weight blocks per group."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        groups: int = 1,
        bias: bool = True,
        shuffle: bool = False,
    ) -> None:
        super().__init__()
        if input_size % groups != 0:
            raise ValueError(f"input_size={input_size} must be divisible by groups={groups}")
        if output_size % groups != 0:
            raise ValueError(
                f"output_size={output_size} must be divisible by groups={groups}"
            )
        self.input_size = input_size
        self.output_size = output_size
        self.groups = groups
        self.in_per_group = input_size // groups
        self.out_per_group = output_size // groups
        self.shuffle = shuffle and groups > 1
        self.weight = nn.Parameter(
            torch.empty(groups, self.in_per_group, self.out_per_group)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(groups, self.out_per_group))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            bound = 1.0 / math.sqrt(self.in_per_group)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(-1) != self.input_size:
            raise ValueError(
                f"expected last dim {self.input_size}, got {x.size(-1)}"
            )
        shape = x.shape[:-1]
        x = x.reshape(*shape, self.groups, self.in_per_group)
        y = torch.einsum("...gi,gio->...go", x, self.weight)
        if self.bias is not None:
            y = y + self.bias
        y = y.reshape(*shape, self.output_size)
        if self.shuffle:
            y = (
                y.reshape(*shape, self.groups, self.out_per_group)
                .transpose(-1, -2)
                .reshape(*shape, self.output_size)
            )
        return y


class _GroupedGRULayer(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        groups: int,
        *,
        batch_first: bool = True,
        bias: bool = True,
        bidirectional: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if input_size % groups != 0:
            raise ValueError(f"input_size={input_size} must be divisible by groups={groups}")
        if hidden_size % groups != 0:
            raise ValueError(
                f"hidden_size={hidden_size} must be divisible by groups={groups}"
            )
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.groups = groups
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.in_per_group = input_size // groups
        self.hidden_per_group = hidden_size // groups
        self.layers = nn.ModuleList(
            [
                nn.GRU(
                    self.in_per_group,
                    self.hidden_per_group,
                    bias=bias,
                    batch_first=batch_first,
                    dropout=dropout,
                    bidirectional=bidirectional,
                )
                for _ in range(groups)
            ]
        )

    def flatten_parameters(self) -> None:
        for layer in self.layers:
            cast(nn.GRU, layer).flatten_parameters()

    def get_h0(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        return torch.zeros(
            self.groups * self.num_directions,
            batch_size,
            self.hidden_per_group,
            device=device,
            dtype=dtype,
        )

    def forward(
        self,
        x: torch.Tensor,
        state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if x.dim() != 3:
            raise ValueError(f"x must be [B, T, C] or [T, B, C], got {tuple(x.shape)}")
        batch_size = x.size(0) if self.batch_first else x.size(1)
        if state is None:
            state = self.get_h0(batch_size, x.device, x.dtype)
        outputs: list[torch.Tensor] = []
        states: list[torch.Tensor] = []
        h_per_group = self.num_directions
        for idx, layer in enumerate(self.layers):
            x_group = x[..., idx * self.in_per_group : (idx + 1) * self.in_per_group]
            y_group, s_group = layer(
                x_group,
                state[idx * h_per_group : (idx + 1) * h_per_group].contiguous(),
            )
            outputs.append(y_group)
            states.append(s_group)
        return torch.cat(outputs, dim=-1), torch.cat(states, dim=0)


class GroupedGRU(nn.Module):
    """Stacked grouped GRUs with optional group shuffling."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 256,
        num_layers: int = 1,
        groups: int = 1,
        *,
        bias: bool = True,
        batch_first: bool = True,
        dropout: float = 0.0,
        bidirectional: bool = False,
        shuffle: bool = True,
        add_outputs: bool = False,
    ) -> None:
        super().__init__()
        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}")
        if input_size % groups != 0:
            raise ValueError(f"input_size={input_size} must be divisible by groups={groups}")
        if hidden_size % groups != 0:
            raise ValueError(
                f"hidden_size={hidden_size} must be divisible by groups={groups}"
            )
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.groups = groups
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.shuffle = shuffle and groups > 1
        self.add_outputs = add_outputs
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.hidden_per_group = hidden_size // groups
        self.layers = nn.ModuleList()
        self.layers.append(
            _GroupedGRULayer(
                input_size,
                hidden_size,
                groups,
                batch_first=batch_first,
                bias=bias,
                bidirectional=bidirectional,
                dropout=dropout,
            )
        )
        for _ in range(1, num_layers):
            self.layers.append(
                _GroupedGRULayer(
                    hidden_size,
                    hidden_size,
                    groups,
                    batch_first=batch_first,
                    bias=bias,
                    bidirectional=bidirectional,
                    dropout=dropout,
                )
            )
        self.flatten_parameters()

    def flatten_parameters(self) -> None:
        for layer in self.layers:
            cast(_GroupedGRULayer, layer).flatten_parameters()

    def get_h0(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        return torch.zeros(
            self.num_layers * self.groups * self.num_directions,
            batch_size,
            self.hidden_per_group,
            device=device,
            dtype=dtype,
        )

    def forward(
        self,
        x: torch.Tensor,
        state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if x.dim() != 3:
            raise ValueError(f"x must be [B, T, C] or [T, B, C], got {tuple(x.shape)}")
        dim0, dim1 = x.shape[:2]
        batch_size = dim0 if self.batch_first else dim1
        if state is None:
            state = self.get_h0(batch_size, x.device, x.dtype)
        outputs = torch.zeros(
            dim0,
            dim1,
            self.hidden_size * self.num_directions,
            device=x.device,
            dtype=x.dtype,
        )
        states: list[torch.Tensor] = []
        state_block = self.groups * self.num_directions
        for idx, layer in enumerate(self.layers):
            x, layer_state = layer(
                x,
                state[idx * state_block : (idx + 1) * state_block],
            )
            states.append(layer_state)
            if self.shuffle and idx < self.num_layers - 1:
                x = (
                    x.reshape(dim0, dim1, self.groups, -1)
                    .transpose(2, 3)
                    .reshape(dim0, dim1, -1)
                )
            if self.add_outputs:
                outputs = outputs + x
            else:
                outputs = x
        return outputs, torch.cat(states, dim=0)


class Conv2dNormAct(nn.Sequential):
    """Causal 2-D convolution over ``[B, C, T, F]`` inputs."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int | tuple[int, int],
        *,
        fstride: int = 1,
        dilation: int = 1,
        fpad: bool = True,
        bias: bool = True,
        separable: bool = False,
        norm_layer: Callable[..., nn.Module] | None = nn.BatchNorm2d,
        activation_layer: Callable[..., nn.Module] | None = nn.ReLU,
    ) -> None:
        kernel = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        t_kernel, f_kernel = kernel
        f_padding = f_kernel // 2 + dilation - 1 if fpad else 0
        layers: list[nn.Module] = []
        if t_kernel > 1:
            layers.append(nn.ConstantPad2d((0, 0, t_kernel - 1, 0), 0.0))
        groups = math.gcd(in_ch, out_ch) if separable else 1
        if groups == 1 or max(kernel) == 1:
            groups = 1
        layers.append(
            nn.Conv2d(
                in_ch,
                out_ch,
                kernel_size=kernel,
                padding=(0, f_padding),
                stride=(1, fstride),
                dilation=(1, dilation),
                groups=groups,
                bias=bias,
            )
        )
        if groups > 1:
            layers.append(nn.Conv2d(out_ch, out_ch, kernel_size=1, bias=False))
        if norm_layer is not None:
            layers.append(norm_layer(out_ch))
        if activation_layer is not None:
            layers.append(activation_layer())
        super().__init__(*layers)


class ConvTranspose2dNormAct(nn.Sequential):
    """Causal transposed convolution over ``[B, C, T, F]`` inputs."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int | tuple[int, int],
        *,
        fstride: int = 1,
        dilation: int = 1,
        fpad: bool = True,
        bias: bool = True,
        separable: bool = False,
        norm_layer: Callable[..., nn.Module] | None = nn.BatchNorm2d,
        activation_layer: Callable[..., nn.Module] | None = nn.ReLU,
    ) -> None:
        kernel = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        t_kernel, f_kernel = kernel
        f_padding = f_kernel // 2 if fpad else 0
        layers: list[nn.Module] = []
        if t_kernel > 1:
            layers.append(nn.ConstantPad2d((0, 0, t_kernel - 1, 0), 0.0))
        groups = math.gcd(in_ch, out_ch) if separable else 1
        if groups == 1:
            groups = 1
        layers.append(
            nn.ConvTranspose2d(
                in_ch,
                out_ch,
                kernel_size=kernel,
                padding=(t_kernel - 1, f_padding + dilation - 1),
                output_padding=(0, f_padding),
                stride=(1, fstride),
                dilation=(1, dilation),
                groups=groups,
                bias=bias,
            )
        )
        if groups > 1:
            layers.append(nn.Conv2d(out_ch, out_ch, kernel_size=1, bias=False))
        if norm_layer is not None:
            layers.append(norm_layer(out_ch))
        if activation_layer is not None:
            layers.append(activation_layer())
        super().__init__(*layers)


class SqueezedGRU(nn.Module):
    """Grouped-linear bottleneck followed by a GRU."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        *,
        output_size: int | None = None,
        num_layers: int = 1,
        linear_groups: int = 1,
        batch_first: bool = True,
        activation_layer: Callable[..., nn.Module] = nn.ReLU,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size if output_size is not None else hidden_size
        self.linear_in = nn.Sequential(
            GroupedLinear(input_size, hidden_size, groups=linear_groups),
            activation_layer(),
        )
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers=num_layers, batch_first=batch_first)
        if output_size is None:
            self.linear_out = nn.Identity()
        else:
            self.linear_out = nn.Sequential(
                GroupedLinear(hidden_size, output_size, groups=linear_groups),
                activation_layer(),
            )

    def forward(
        self,
        x: torch.Tensor,
        state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.linear_in(x)
        y, state = self.gru(x, state)
        return self.linear_out(y), state


SepConv2d = Conv2dNormAct
ConvTranspose = ConvTranspose2dNormAct


__all__ = [
    "GroupedLinear",
    "GroupedGRU",
    "Conv2dNormAct",
    "ConvTranspose2dNormAct",
    "ConvTranspose",
    "SepConv2d",
    "SqueezedGRU",
]
