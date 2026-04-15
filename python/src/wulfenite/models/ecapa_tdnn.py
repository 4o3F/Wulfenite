"""Native SpeechBrain-compatible ECAPA-TDNN speaker encoder."""

from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F


def _length_to_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    """Build a time mask from relative or absolute lengths."""
    if lengths.dim() != 1:
        raise ValueError(f"lengths must be [B], got {tuple(lengths.shape)}")

    if torch.is_floating_point(lengths) and float(lengths.max()) <= 1.0 + 1e-6:
        valid = torch.round(lengths * max_len).to(dtype=torch.long)
    else:
        valid = lengths.to(dtype=torch.long)
    valid = valid.clamp(min=0, max=max_len)
    steps = torch.arange(max_len, device=lengths.device)
    return steps.unsqueeze(0) < valid.unsqueeze(1)


class _ConvWrap(nn.Module):
    """Wrapper producing ``conv.conv.*`` keys."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__()
        self.conv = nn.Conv1d(*args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class _NormWrap(nn.Module):
    """Wrapper producing ``norm.norm.*`` keys."""

    def __init__(self, num_features: int) -> None:
        super().__init__()
        self.norm = nn.BatchNorm1d(num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x)


class TDNNBlock(nn.Module):
    """SpeechBrain-style TDNN block."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int,
        dilation: int = 1,
        activation: type[nn.Module] = nn.ReLU,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        padding = ((kernel_size - 1) * dilation) // 2
        self.conv = _ConvWrap(
            in_ch,
            out_ch,
            kernel_size,
            dilation=dilation,
            padding=padding,
        )
        self.activation = activation()
        self.norm = _NormWrap(out_ch)
        self.dropout = nn.Dropout1d(p=dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.norm(self.activation(self.conv(x))))


class Res2NetBlock(nn.Module):
    """SpeechBrain Res2Net block with ``blocks`` key layout."""

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dilation: int,
        scale: int = 8,
        activation: type[nn.Module] = nn.ReLU,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if channels % scale != 0:
            raise ValueError(f"{channels} % {scale} != 0")
        width = channels // scale
        self.blocks = nn.ModuleList(
            [
                TDNNBlock(
                    width,
                    width,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    activation=activation,
                    dropout=dropout,
                )
                for _ in range(scale - 1)
            ]
        )
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs: list[torch.Tensor] = []
        residual_branch: torch.Tensor | None = None
        for index, split in enumerate(torch.chunk(x, self.scale, dim=1)):
            if index == 0:
                branch = split
            elif index == 1:
                branch = self.blocks[index - 1](split)
            else:
                assert residual_branch is not None
                branch = self.blocks[index - 1](split + residual_branch)
            outputs.append(branch)
            residual_branch = branch
        return torch.cat(outputs, dim=1)


class SEBlock(nn.Module):
    """SpeechBrain squeeze-excitation block using Conv1d wrappers."""

    def __init__(self, channels: int, se_channels: int) -> None:
        super().__init__()
        self.conv1 = _ConvWrap(channels, se_channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = _ConvWrap(se_channels, channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if lengths is None:
            pooled = x.mean(dim=2, keepdim=True)
        else:
            mask = _length_to_mask(lengths.to(x.device), x.size(-1)).unsqueeze(1)
            weights = mask.to(dtype=x.dtype)
            denom = weights.sum(dim=2, keepdim=True).clamp_min(1.0)
            pooled = (x * weights).sum(dim=2, keepdim=True) / denom

        pooled = self.relu(self.conv1(pooled))
        pooled = self.sigmoid(self.conv2(pooled))
        return pooled * x


class SERes2NetBlock(nn.Module):
    """SpeechBrain SE-Res2Net residual block."""

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dilation: int,
        scale: int = 8,
        se_channels: int = 128,
        activation: type[nn.Module] = nn.ReLU,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.tdnn1 = TDNNBlock(
            channels,
            channels,
            kernel_size=1,
            dilation=1,
            activation=activation,
            dropout=dropout,
        )
        self.res2net_block = Res2NetBlock(
            channels,
            kernel_size=kernel_size,
            dilation=dilation,
            scale=scale,
            activation=activation,
            dropout=dropout,
        )
        self.tdnn2 = TDNNBlock(
            channels,
            channels,
            kernel_size=1,
            dilation=1,
            activation=activation,
            dropout=dropout,
        )
        self.se_block = SEBlock(channels, se_channels)

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        residual = x
        x = self.tdnn1(x)
        x = self.res2net_block(x)
        x = self.tdnn2(x)
        x = self.se_block(x, lengths=lengths)
        return x + residual


class AttentiveStatisticsPooling(nn.Module):
    """SpeechBrain attentive statistics pooling with optional global context."""

    def __init__(
        self,
        in_dim: int,
        attention_channels: int = 128,
        global_context: bool = True,
        activation: type[nn.Module] = nn.ReLU,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.eps = 1e-12
        self.in_dim = in_dim
        self.global_context = global_context
        tdnn_in_dim = in_dim * 3 if global_context else in_dim
        self.tdnn = TDNNBlock(
            tdnn_in_dim,
            attention_channels,
            kernel_size=1,
            dilation=1,
            activation=activation,
            dropout=dropout,
        )
        self.tanh = nn.Tanh()
        self.conv = _ConvWrap(attention_channels, in_dim, kernel_size=1)

    def _compute_statistics(
        self,
        x: torch.Tensor,
        weights: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mean = (weights * x).sum(dim=2)
        var = (weights * (x - mean.unsqueeze(2)).pow(2)).sum(dim=2)
        std = torch.sqrt(var.clamp_min(self.eps))
        return mean, std

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        frames = x.size(-1)
        if lengths is None:
            mask = x.new_ones((x.size(0), 1, frames), dtype=torch.bool)
        else:
            mask = _length_to_mask(lengths.to(x.device), frames).unsqueeze(1)

        if self.global_context:
            mask_float = mask.to(dtype=x.dtype)
            total = mask_float.sum(dim=2, keepdim=True).clamp_min(1.0)
            mean, std = self._compute_statistics(x, mask_float / total)
            mean = mean.unsqueeze(2).expand(-1, -1, frames)
            std = std.unsqueeze(2).expand(-1, -1, frames)
            attn_input = torch.cat([x, mean, std], dim=1)
        else:
            attn_input = x

        attn = self.conv(self.tanh(self.tdnn(attn_input)))
        attn = attn.masked_fill(~mask, float("-inf"))
        attn = torch.softmax(attn, dim=2)
        mean, std = self._compute_statistics(x, attn)
        return torch.cat([mean, std], dim=1).unsqueeze(2)


class ECAPA_TDNN(nn.Module):
    """SpeechBrain-compatible ECAPA-TDNN."""

    def __init__(
        self,
        channels: int = 512,
        feat_dim: int = 80,
        embed_dim: int = 192,
        pooling_func: str = "ASTP",
        global_context_att: bool = True,
        emb_bn: bool = False,
        attention_channels: int = 128,
        scale: int = 8,
        se_channels: int = 128,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if pooling_func != "ASTP":
            raise ValueError(
                f"Unsupported pooling_func={pooling_func!r}; only 'ASTP' is supported."
            )

        self.channels = channels
        self.blocks = nn.ModuleList(
            [
                TDNNBlock(feat_dim, channels, kernel_size=5, dilation=1, dropout=dropout),
                SERes2NetBlock(
                    channels,
                    kernel_size=3,
                    dilation=2,
                    scale=scale,
                    se_channels=se_channels,
                    dropout=dropout,
                ),
                SERes2NetBlock(
                    channels,
                    kernel_size=3,
                    dilation=3,
                    scale=scale,
                    se_channels=se_channels,
                    dropout=dropout,
                ),
                SERes2NetBlock(
                    channels,
                    kernel_size=3,
                    dilation=4,
                    scale=scale,
                    se_channels=se_channels,
                    dropout=dropout,
                ),
            ]
        )
        self.mfa = TDNNBlock(
            channels * 3,
            channels * 3,
            kernel_size=1,
            dilation=1,
            dropout=dropout,
        )
        self.asp = AttentiveStatisticsPooling(
            channels * 3,
            attention_channels=attention_channels,
            global_context=global_context_att,
            dropout=dropout,
        )
        self.asp_bn = _NormWrap(channels * 6)
        self.fc = _ConvWrap(channels * 6, embed_dim, kernel_size=1)
        self.emb_bn = emb_bn
        self.bn2 = _NormWrap(embed_dim) if emb_bn else nn.Identity()

    def _compute_frame_features(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x.transpose(1, 2)
        outputs: list[torch.Tensor] = []
        for index, block in enumerate(self.blocks):
            if index == 0:
                x = block(x)
            else:
                x = block(x, lengths=lengths)
            outputs.append(x)
        x = torch.cat(outputs[1:], dim=1)
        return self.mfa(x)

    def get_frame_level_feat(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self._compute_frame_features(x, lengths=lengths).transpose(1, 2)

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if x.dim() != 3:
            raise ValueError(f"Expected [B, T, F] features, got {tuple(x.shape)}")
        frame_features = self._compute_frame_features(x, lengths=lengths)
        pooled = self.asp(frame_features, lengths=lengths)
        pooled = self.asp_bn(pooled)
        embedding = self.fc(pooled)
        embedding = self.bn2(embedding)
        return frame_features, embedding.squeeze(-1)


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
        global_context_att=False,
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
        global_context_att=False,
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
    """Infer SpeechBrain ECAPA checkpoint configuration from tensor shapes."""

    required_keys = (
        "blocks.0.conv.conv.weight",
        "mfa.conv.conv.weight",
        "asp.tdnn.conv.conv.weight",
        "asp_bn.norm.weight",
        "fc.conv.weight",
    )
    missing = [key for key in required_keys if key not in state_dict]
    if missing:
        joined = ", ".join(missing)
        raise ValueError(
            "Checkpoint does not look like a SpeechBrain ECAPA-TDNN checkpoint; "
            f"missing: {joined}"
        )

    stem_weight = state_dict["blocks.0.conv.conv.weight"]
    mfa_weight = state_dict["mfa.conv.conv.weight"]
    asp_tdnn_weight = state_dict["asp.tdnn.conv.conv.weight"]
    asp_bn_weight = state_dict["asp_bn.norm.weight"]
    fc_weight = state_dict["fc.conv.weight"]

    if stem_weight.ndim != 3:
        raise ValueError("blocks.0.conv.conv.weight must be a 3D Conv1d tensor")
    if mfa_weight.ndim != 3:
        raise ValueError("mfa.conv.conv.weight must be a 3D Conv1d tensor")
    if asp_tdnn_weight.ndim != 3:
        raise ValueError("asp.tdnn.conv.conv.weight must be a 3D Conv1d tensor")
    if asp_bn_weight.ndim != 1:
        raise ValueError("asp_bn.norm.weight must be a 1D BatchNorm tensor")
    if fc_weight.ndim != 3:
        raise ValueError("fc.conv.weight must be a 3D Conv1d tensor")

    channels = int(stem_weight.shape[0])
    feat_dim = int(stem_weight.shape[1])
    embed_dim = int(fc_weight.shape[0])
    mfa_out_channels = int(mfa_weight.shape[0])
    mfa_in_channels = int(mfa_weight.shape[1])
    pooled_channels = int(asp_bn_weight.shape[0])
    asp_tdnn_in_channels = int(asp_tdnn_weight.shape[1])

    if channels not in (512, 1024):
        raise ValueError(f"Unsupported ECAPA channel size: {channels}")
    if feat_dim <= 0:
        raise ValueError(f"Invalid feature dimension: {feat_dim}")
    if mfa_in_channels != channels * 3 or mfa_out_channels != channels * 3:
        raise ValueError(
            "Unexpected MFA dimensions: expected "
            f"{channels * 3}->{channels * 3}, got {mfa_in_channels}->{mfa_out_channels}"
        )
    if pooled_channels != channels * 6:
        raise ValueError(
            f"Unexpected pooled feature size: expected {channels * 6}, got {pooled_channels}"
        )
    if int(fc_weight.shape[1]) != channels * 6:
        raise ValueError(
            "Unexpected FC input channels: expected "
            f"{channels * 6}, got {int(fc_weight.shape[1])}"
        )

    if asp_tdnn_in_channels == mfa_out_channels * 3:
        global_context_att = True
    elif asp_tdnn_in_channels == mfa_out_channels:
        global_context_att = False
    else:
        raise ValueError(
            "Unable to infer global_context_att from asp.tdnn.conv.conv.weight "
            f"shape {tuple(asp_tdnn_weight.shape)}"
        )

    emb_bn = "bn2.norm.weight" in state_dict
    return {
        "channels": channels,
        "feat_dim": feat_dim,
        "embed_dim": embed_dim,
        "global_context_att": global_context_att,
        "emb_bn": emb_bn,
    }


__all__ = [
    "AttentiveStatisticsPooling",
    "ECAPA_TDNN",
    "ECAPA_TDNN_GLOB_c1024",
    "ECAPA_TDNN_GLOB_c512",
    "ECAPA_TDNN_c1024",
    "ECAPA_TDNN_c512",
    "Res2NetBlock",
    "SEBlock",
    "SERes2NetBlock",
    "TDNNBlock",
    "detect_ecapa_variant",
]
