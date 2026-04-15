"""Native WeSpeaker-compatible ECAPA-TDNN speaker encoder."""

from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F


class Res2Conv1dReluBn(nn.Module):
    """Res2Net-style 1D convolution block used by ECAPA-TDNN."""

    def __init__(
        self,
        channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        bias: bool = True,
        scale: int = 4,
    ) -> None:
        super().__init__()
        if channels % scale != 0:
            raise ValueError(f"{channels} % {scale} != 0")
        self.scale = scale
        self.width = channels // scale
        self.nums = scale if scale == 1 else scale - 1

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _ in range(self.nums):
            self.convs.append(
                nn.Conv1d(
                    self.width,
                    self.width,
                    kernel_size,
                    stride,
                    padding,
                    dilation,
                    bias=bias,
                )
            )
            self.bns.append(nn.BatchNorm1d(self.width))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: list[torch.Tensor] = []
        splits = torch.split(x, self.width, dim=1)
        span = splits[0]
        for index, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            if index >= 1:
                span = span + splits[index]
            span = conv(span)
            span = bn(F.relu(span))
            out.append(span)
        if self.scale != 1:
            out.append(splits[self.nums])
        return torch.cat(out, dim=1)


class Conv1dReluBn(nn.Module):
    """Conv1d -> ReLU -> BatchNorm1d block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            bias=bias,
        )
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(F.relu(self.conv(x)))


class SE_Connect(nn.Module):
    """Squeeze-excitation over temporal channel means."""

    def __init__(self, channels: int, se_bottleneck_dim: int | None = None) -> None:
        super().__init__()
        if se_bottleneck_dim is None:
            se_bottleneck_dim = channels // 4
        self.linear1 = nn.Linear(channels, se_bottleneck_dim)
        self.linear2 = nn.Linear(se_bottleneck_dim, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x.mean(dim=2)
        out = F.relu(self.linear1(out))
        out = torch.sigmoid(self.linear2(out))
        return x * out.unsqueeze(2)


class SE_Res2Block(nn.Module):
    """SE-Res2Net residual block."""

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        dilation: int,
        scale: int,
    ) -> None:
        super().__init__()
        self.se_res2block = nn.Sequential(
            Conv1dReluBn(
                channels,
                channels,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            Res2Conv1dReluBn(
                channels,
                kernel_size,
                stride,
                padding,
                dilation,
                scale=scale,
            ),
            Conv1dReluBn(
                channels,
                channels,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            SE_Connect(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.se_res2block(x)


class ASTP(nn.Module):
    """Attentive statistics pooling used by WeSpeaker ECAPA-TDNN."""

    def __init__(
        self,
        in_dim: int,
        bottleneck_dim: int = 128,
        global_context_att: bool = False,
        **_: object,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.global_context_att = global_context_att

        linear1_in_dim = in_dim * 3 if global_context_att else in_dim
        self.linear1 = nn.Conv1d(linear1_in_dim, bottleneck_dim, kernel_size=1)
        self.linear2 = nn.Conv1d(bottleneck_dim, in_dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            x = x.reshape(x.shape[0], x.shape[1] * x.shape[2], x.shape[3])
        if x.dim() != 3:
            raise ValueError(f"Expected [B, F, T] input, got {tuple(x.shape)}")

        if self.global_context_att:
            context_mean = torch.mean(x, dim=-1, keepdim=True).expand_as(x)
            context_std = torch.sqrt(
                torch.var(x, dim=-1, keepdim=True) + 1e-7
            ).expand_as(x)
            x_in = torch.cat((x, context_mean, context_std), dim=1)
        else:
            x_in = x

        alpha = torch.tanh(self.linear1(x_in))
        alpha = torch.softmax(self.linear2(alpha), dim=2)
        mean = torch.sum(alpha * x, dim=2)
        var = torch.sum(alpha * (x**2), dim=2) - mean**2
        std = torch.sqrt(var.clamp(min=1e-7))
        return torch.cat([mean, std], dim=1)

    def get_out_dim(self) -> int:
        return self.in_dim * 2


class ECAPA_TDNN(nn.Module):
    """Native ECAPA-TDNN with upstream-compatible parameter names."""

    def __init__(
        self,
        channels: int = 512,
        feat_dim: int = 80,
        embed_dim: int = 192,
        pooling_func: str = "ASTP",
        global_context_att: bool = False,
        emb_bn: bool = False,
    ) -> None:
        super().__init__()
        if pooling_func != "ASTP":
            raise ValueError(
                f"Unsupported pooling_func={pooling_func!r}; only 'ASTP' is supported."
            )

        self.layer1 = Conv1dReluBn(feat_dim, channels, kernel_size=5, padding=2)
        self.layer2 = SE_Res2Block(
            channels,
            kernel_size=3,
            stride=1,
            padding=2,
            dilation=2,
            scale=8,
        )
        self.layer3 = SE_Res2Block(
            channels,
            kernel_size=3,
            stride=1,
            padding=3,
            dilation=3,
            scale=8,
        )
        self.layer4 = SE_Res2Block(
            channels,
            kernel_size=3,
            stride=1,
            padding=4,
            dilation=4,
            scale=8,
        )

        cat_channels = channels * 3
        out_channels = 512 * 3
        self.conv = nn.Conv1d(cat_channels, out_channels, kernel_size=1)
        self.pool = ASTP(
            in_dim=out_channels,
            global_context_att=global_context_att,
        )
        self.pool_out_dim = self.pool.get_out_dim()
        self.bn = nn.BatchNorm1d(self.pool_out_dim)
        self.linear = nn.Linear(self.pool_out_dim, embed_dim)
        self.emb_bn = emb_bn
        self.bn2 = nn.BatchNorm1d(embed_dim) if emb_bn else nn.Identity()

    def _get_frame_level_feat(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = x.permute(0, 2, 1)

        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)

        out = torch.cat([out2, out3, out4], dim=1)
        out = self.conv(out)
        return out, out4

    def get_frame_level_feat(self, x: torch.Tensor) -> torch.Tensor:
        out = self._get_frame_level_feat(x)[0].permute(0, 2, 1)
        return out

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out, out4 = self._get_frame_level_feat(x)
        out = F.relu(out)
        out = self.bn(self.pool(out))
        out = self.linear(out)
        out = self.bn2(out)
        return out4, out


def ECAPA_TDNN_c1024(
    feat_dim: int,
    embed_dim: int,
    pooling_func: str = "ASTP",
    emb_bn: bool = False,
) -> ECAPA_TDNN:
    return ECAPA_TDNN(
        channels=1024,
        feat_dim=feat_dim,
        embed_dim=embed_dim,
        pooling_func=pooling_func,
        emb_bn=emb_bn,
    )


def ECAPA_TDNN_GLOB_c1024(
    feat_dim: int,
    embed_dim: int,
    pooling_func: str = "ASTP",
    emb_bn: bool = False,
) -> ECAPA_TDNN:
    return ECAPA_TDNN(
        channels=1024,
        feat_dim=feat_dim,
        embed_dim=embed_dim,
        pooling_func=pooling_func,
        global_context_att=True,
        emb_bn=emb_bn,
    )


def ECAPA_TDNN_c512(
    feat_dim: int,
    embed_dim: int,
    pooling_func: str = "ASTP",
    emb_bn: bool = False,
) -> ECAPA_TDNN:
    return ECAPA_TDNN(
        channels=512,
        feat_dim=feat_dim,
        embed_dim=embed_dim,
        pooling_func=pooling_func,
        emb_bn=emb_bn,
    )


def ECAPA_TDNN_GLOB_c512(
    feat_dim: int,
    embed_dim: int,
    pooling_func: str = "ASTP",
    emb_bn: bool = False,
) -> ECAPA_TDNN:
    return ECAPA_TDNN(
        channels=512,
        feat_dim=feat_dim,
        embed_dim=embed_dim,
        pooling_func=pooling_func,
        global_context_att=True,
        emb_bn=emb_bn,
    )


def detect_ecapa_variant(
    state_dict: Mapping[str, torch.Tensor],
) -> dict[str, int | bool]:
    """Infer ECAPA checkpoint configuration from tensor shapes."""

    required_keys = (
        "layer1.conv.weight",
        "conv.weight",
        "pool.linear1.weight",
        "pool.linear2.weight",
        "linear.weight",
    )
    missing = [key for key in required_keys if key not in state_dict]
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"Checkpoint does not look like ECAPA-TDNN; missing: {joined}")

    layer1_weight = state_dict["layer1.conv.weight"]
    conv_weight = state_dict["conv.weight"]
    pool_linear1_weight = state_dict["pool.linear1.weight"]
    pool_linear2_weight = state_dict["pool.linear2.weight"]
    linear_weight = state_dict["linear.weight"]

    if layer1_weight.ndim != 3:
        raise ValueError("layer1.conv.weight must be a 3D Conv1d tensor")
    if conv_weight.ndim != 3:
        raise ValueError("conv.weight must be a 3D Conv1d tensor")
    if pool_linear1_weight.ndim != 3:
        raise ValueError("pool.linear1.weight must be a 3D Conv1d tensor")
    if pool_linear2_weight.ndim != 3:
        raise ValueError("pool.linear2.weight must be a 3D Conv1d tensor")
    if linear_weight.ndim != 2:
        raise ValueError("linear.weight must be a 2D Linear tensor")

    channels = int(layer1_weight.shape[0])
    feat_dim = int(layer1_weight.shape[1])
    embed_dim = int(linear_weight.shape[0])
    fusion_in_channels = int(conv_weight.shape[1])
    fusion_out_channels = int(conv_weight.shape[0])
    pool_in_channels = int(pool_linear2_weight.shape[0])
    pool_linear1_in_channels = int(pool_linear1_weight.shape[1])
    pool_out_dim = int(linear_weight.shape[1])

    if channels not in (512, 1024):
        raise ValueError(f"Unsupported ECAPA channel size: {channels}")
    if feat_dim <= 0:
        raise ValueError(f"Invalid feature dimension: {feat_dim}")
    if fusion_in_channels != channels * 3:
        raise ValueError(
            f"Unexpected fusion input channels: expected {channels * 3}, got {fusion_in_channels}"
        )
    if fusion_out_channels != 1536:
        raise ValueError(
            f"Unexpected fusion output channels: expected 1536, got {fusion_out_channels}"
        )
    if pool_in_channels != 1536:
        raise ValueError(
            f"Unexpected pooling input channels: expected 1536, got {pool_in_channels}"
        )
    if pool_out_dim != 3072:
        raise ValueError(
            f"Unexpected embedding head input dim: expected 3072, got {pool_out_dim}"
        )

    if pool_linear1_in_channels == pool_in_channels:
        global_context_att = False
    elif pool_linear1_in_channels == pool_in_channels * 3:
        global_context_att = True
    else:
        raise ValueError(
            "Unable to infer global_context_att from pool.linear1.weight shape "
            f"{tuple(pool_linear1_weight.shape)}"
        )

    emb_bn = any(key.startswith("bn2.") for key in state_dict)
    return {
        "channels": channels,
        "feat_dim": feat_dim,
        "embed_dim": embed_dim,
        "global_context_att": global_context_att,
        "emb_bn": emb_bn,
    }


__all__ = [
    "ASTP",
    "Conv1dReluBn",
    "ECAPA_TDNN",
    "ECAPA_TDNN_GLOB_c1024",
    "ECAPA_TDNN_GLOB_c512",
    "ECAPA_TDNN_c1024",
    "ECAPA_TDNN_c512",
    "Res2Conv1dReluBn",
    "SE_Connect",
    "SE_Res2Block",
    "detect_ecapa_variant",
]
