"""CAM++ zh-cn speaker encoder (frozen).

Adapted from 3D-Speaker. The core model is a faithful port of
``speakerlab/models/campplus/DTDNN.py`` + ``layers.py`` merged into a
single torch-only module. The default configuration matches the
``iic/speech_campplus_sv_zh-cn_16k-common`` ModelScope checkpoint
(embedding_size=192, feat_dim=80, ~200k Chinese speakers).

Sources:
- https://github.com/modelscope/3D-Speaker/blob/main/speakerlab/models/campplus/DTDNN.py
- https://github.com/modelscope/3D-Speaker/blob/main/speakerlab/models/campplus/layers.py
- License: Apache-2.0 (3D-Speaker, alibaba-damo-academy)

Usage in Wulfenite:
This module is imported by ``wulfenite.models.tse`` as the frozen
speaker encoder. Only ``encode_enrollment`` is part of the public
interface — it takes a raw 16 kHz waveform, computes 80-dim Kaldi
FBank internally, runs the CAM++ forward pass, and L2-normalizes the
192-d embedding. Callers should invoke it ONCE per session and cache
the result for every subsequent separator call. See
``docs/onnx_contract.md`` for the full pipeline.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.utils.checkpoint as cp
import torchaudio.compliance.kaldi as kaldi
from torch import nn


SAMPLE_RATE = 16000
FEAT_DIM = 80
EMBEDDING_SIZE = 192


# ---------------------------------------------------------------------------
# Primitive layers (from layers.py)
# ---------------------------------------------------------------------------


def get_nonlinear(config_str: str, channels: int) -> nn.Sequential:
    nonlinear = nn.Sequential()
    for name in config_str.split("-"):
        if name == "relu":
            nonlinear.add_module("relu", nn.ReLU(inplace=True))
        elif name == "prelu":
            nonlinear.add_module("prelu", nn.PReLU(channels))
        elif name == "batchnorm":
            nonlinear.add_module("batchnorm", nn.BatchNorm1d(channels))
        elif name == "batchnorm_":
            nonlinear.add_module("batchnorm", nn.BatchNorm1d(channels, affine=False))
        else:
            raise ValueError(f"Unexpected module ({name}).")
    return nonlinear


def statistics_pooling(x: torch.Tensor, dim: int = -1, keepdim: bool = False,
                       unbiased: bool = True) -> torch.Tensor:
    mean = x.mean(dim=dim)
    std = x.std(dim=dim, unbiased=unbiased)
    stats = torch.cat([mean, std], dim=-1)
    if keepdim:
        stats = stats.unsqueeze(dim=dim)
    return stats


class StatsPool(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return statistics_pooling(x)


class TDNNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, bias=False, config_str="batchnorm-relu"):
        super().__init__()
        if padding < 0:
            assert kernel_size % 2 == 1, (
                f"Expect equal paddings, but got even kernel size ({kernel_size})"
            )
            padding = (kernel_size - 1) // 2 * dilation
        self.linear = nn.Conv1d(in_channels, out_channels, kernel_size,
                                stride=stride, padding=padding, dilation=dilation, bias=bias)
        self.nonlinear = get_nonlinear(config_str, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        x = self.nonlinear(x)
        return x


class CAMLayer(nn.Module):
    def __init__(self, bn_channels, out_channels, kernel_size, stride, padding,
                 dilation, bias, reduction: int = 2):
        super().__init__()
        self.linear_local = nn.Conv1d(bn_channels, out_channels, kernel_size,
                                      stride=stride, padding=padding, dilation=dilation, bias=bias)
        self.linear1 = nn.Conv1d(bn_channels, bn_channels // reduction, 1)
        self.relu = nn.ReLU(inplace=True)
        self.linear2 = nn.Conv1d(bn_channels // reduction, out_channels, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.linear_local(x)
        context = x.mean(-1, keepdim=True) + self.seg_pooling(x)
        context = self.relu(self.linear1(context))
        m = self.sigmoid(self.linear2(context))
        return y * m

    @staticmethod
    def seg_pooling(x: torch.Tensor, seg_len: int = 100, stype: str = "avg") -> torch.Tensor:
        if stype == "avg":
            seg = F.avg_pool1d(x, kernel_size=seg_len, stride=seg_len, ceil_mode=True)
        elif stype == "max":
            seg = F.max_pool1d(x, kernel_size=seg_len, stride=seg_len, ceil_mode=True)
        else:
            raise ValueError("Wrong segment pooling type.")
        shape = seg.shape
        seg = seg.unsqueeze(-1).expand(*shape, seg_len).reshape(*shape[:-1], -1)
        seg = seg[..., : x.shape[-1]]
        return seg


class CAMDenseTDNNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, bn_channels, kernel_size,
                 stride=1, dilation=1, bias=False, config_str="batchnorm-relu",
                 memory_efficient: bool = False):
        super().__init__()
        assert kernel_size % 2 == 1, (
            f"Expect equal paddings, but got even kernel size ({kernel_size})"
        )
        padding = (kernel_size - 1) // 2 * dilation
        self.memory_efficient = memory_efficient
        self.nonlinear1 = get_nonlinear(config_str, in_channels)
        self.linear1 = nn.Conv1d(in_channels, bn_channels, 1, bias=False)
        self.nonlinear2 = get_nonlinear(config_str, bn_channels)
        self.cam_layer = CAMLayer(bn_channels, out_channels, kernel_size,
                                  stride=stride, padding=padding, dilation=dilation, bias=bias)

    def bn_function(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear1(self.nonlinear1(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and self.memory_efficient:
            x = cp.checkpoint(self.bn_function, x)
        else:
            x = self.bn_function(x)
        x = self.cam_layer(self.nonlinear2(x))
        return x


class CAMDenseTDNNBlock(nn.ModuleList):
    def __init__(self, num_layers, in_channels, out_channels, bn_channels,
                 kernel_size, stride=1, dilation=1, bias=False,
                 config_str="batchnorm-relu", memory_efficient: bool = False):
        super().__init__()
        for i in range(num_layers):
            layer = CAMDenseTDNNLayer(
                in_channels=in_channels + i * out_channels,
                out_channels=out_channels,
                bn_channels=bn_channels,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                bias=bias,
                config_str=config_str,
                memory_efficient=memory_efficient,
            )
            self.add_module(f"tdnnd{i + 1}", layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self:
            x = torch.cat([x, layer(x)], dim=1)
        return x


class TransitLayer(nn.Module):
    def __init__(self, in_channels, out_channels, bias: bool = True,
                 config_str: str = "batchnorm-relu"):
        super().__init__()
        self.nonlinear = get_nonlinear(config_str, in_channels)
        self.linear = nn.Conv1d(in_channels, out_channels, 1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.nonlinear(x)
        x = self.linear(x)
        return x


class DenseLayer(nn.Module):
    def __init__(self, in_channels, out_channels, bias: bool = False,
                 config_str: str = "batchnorm-relu"):
        super().__init__()
        self.linear = nn.Conv1d(in_channels, out_channels, 1, bias=bias)
        self.nonlinear = get_nonlinear(config_str, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = self.linear(x.unsqueeze(dim=-1)).squeeze(dim=-1)
        else:
            x = self.linear(x)
        x = self.nonlinear(x)
        return x


class BasicResBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,
                               stride=(stride, 1), padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1,
                          stride=(stride, 1), bias=False),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        out = F.relu(out)
        return out


# ---------------------------------------------------------------------------
# CAM++ model (from DTDNN.py)
# ---------------------------------------------------------------------------


class FCM(nn.Module):
    """Front Context Module — a shallow 2D ResNet over the FBank features."""

    def __init__(self, block=BasicResBlock, num_blocks=(2, 2),
                 m_channels: int = 32, feat_dim: int = 80):
        super().__init__()
        self.in_planes = m_channels
        self.conv1 = nn.Conv2d(1, m_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(m_channels)

        self.layer1 = self._make_layer(block, m_channels, num_blocks[0], stride=2)
        self.layer2 = self._make_layer(block, m_channels, num_blocks[1], stride=2)

        self.conv2 = nn.Conv2d(m_channels, m_channels, kernel_size=3,
                               stride=(2, 1), padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(m_channels)
        self.out_channels = m_channels * (feat_dim // 8)

    def _make_layer(self, block, planes: int, num_blocks: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = F.relu(self.bn2(self.conv2(out)))
        shape = out.shape
        out = out.reshape(shape[0], shape[1] * shape[2], shape[3])
        return out


class CAMPPlus(nn.Module):
    """CAM++ speaker encoder.

    Defaults match the iic/speech_campplus_sv_zh-cn_16k-common checkpoint
    (embedding_size=192, feat_dim=80), trained on ~200k Chinese speakers.
    """

    def __init__(
        self,
        feat_dim: int = 80,
        embedding_size: int = 192,
        growth_rate: int = 32,
        bn_size: int = 4,
        init_channels: int = 128,
        config_str: str = "batchnorm-relu",
        memory_efficient: bool = True,
    ):
        super().__init__()

        self.head = FCM(feat_dim=feat_dim)
        channels = self.head.out_channels

        self.xvector = nn.Sequential(
            OrderedDict(
                [
                    (
                        "tdnn",
                        TDNNLayer(
                            channels,
                            init_channels,
                            5,
                            stride=2,
                            dilation=1,
                            padding=-1,
                            config_str=config_str,
                        ),
                    ),
                ]
            )
        )
        channels = init_channels
        for i, (num_layers, kernel_size, dilation) in enumerate(
            zip((12, 24, 16), (3, 3, 3), (1, 2, 2))
        ):
            block = CAMDenseTDNNBlock(
                num_layers=num_layers,
                in_channels=channels,
                out_channels=growth_rate,
                bn_channels=bn_size * growth_rate,
                kernel_size=kernel_size,
                dilation=dilation,
                config_str=config_str,
                memory_efficient=memory_efficient,
            )
            self.xvector.add_module(f"block{i + 1}", block)
            channels = channels + num_layers * growth_rate
            self.xvector.add_module(
                f"transit{i + 1}",
                TransitLayer(channels, channels // 2, bias=False, config_str=config_str),
            )
            channels //= 2

        self.xvector.add_module("out_nonlinear", get_nonlinear(config_str, channels))
        self.xvector.add_module("stats", StatsPool())
        self.xvector.add_module(
            "dense",
            DenseLayer(channels * 2, embedding_size, config_str="batchnorm_"),
        )

        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass over FBank features.

        Args:
            x: ``[B, T, feat_dim]`` log-mel FBank, e.g. the output of
               ``compute_fbank``.

        Returns:
            ``[B, embedding_size]`` raw (unnormalized) speaker embedding.
        """
        x = x.permute(0, 2, 1)  # (B,T,F) -> (B,F,T)
        x = self.head(x)
        x = self.xvector(x)
        return x


# ---------------------------------------------------------------------------
# Feature extraction + enrollment pipeline
# ---------------------------------------------------------------------------


def compute_fbank(
    waveform: torch.Tensor,
    sample_rate: int = SAMPLE_RATE,
    num_mel_bins: int = FEAT_DIM,
    dither: float = 0.0,
    mean_norm: bool = True,
) -> torch.Tensor:
    """Kaldi-style log-mel FBank used as CAM++ input.

    Matches 3D-Speaker's ``FBank(mean_nor=True)`` in
    ``speakerlab/process/processor.py``.

    Args:
        waveform: ``[T]`` or ``[1, T]`` mono 16 kHz audio in ``[-1, 1]``.
        sample_rate: 16000.
        num_mel_bins: 80.
        dither: 0.0 at inference; nonzero only for training.
        mean_norm: utterance-level mean subtraction (default True).

    Returns:
        ``[1, T_frames, num_mel_bins]`` log-mel features. The leading
        batch dim is added so it plugs straight into ``CAMPPlus.forward``.
    """
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    elif waveform.dim() == 2 and waveform.size(0) > 1:
        waveform = waveform[:1]
    feats = kaldi.fbank(
        waveform,
        num_mel_bins=num_mel_bins,
        sample_frequency=sample_rate,
        dither=dither,
    )
    if mean_norm:
        feats = feats - feats.mean(dim=0, keepdim=True)
    return feats.unsqueeze(0)  # [1, T, num_mel_bins]


def encode_enrollment(
    model: CAMPPlus,
    waveform: torch.Tensor,
    sample_rate: int = SAMPLE_RATE,
) -> torch.Tensor:
    """Full enrollment pipeline: raw waveform -> L2-normalized embedding.

    This is the one function called at Wulfenite session startup. It
    runs Kaldi FBank on the enrollment waveform, feeds it to CAM++,
    and L2-normalizes the 192-d output. The frozen-path
    :class:`wulfenite.models.tse.WulfeniteTSE` wrapper then projects
    that raw CAM++ embedding into the separator bottleneck space.

    Args:
        model: a loaded ``CAMPPlus`` instance (typically from
            ``load_campplus_cn_common``). Should be in eval mode.
        waveform: ``[T]`` or ``[1, T]`` 16 kHz mono enrollment, 3-10 s
            recommended. No pre-normalization required.
        sample_rate: 16000.

    Returns:
        ``[1, 192]`` unit-L2 CAM++ speaker embedding.
    """
    fbank = compute_fbank(waveform, sample_rate=sample_rate)
    embedding = model(fbank)  # [1, 192]
    embedding = F.normalize(embedding, p=2, dim=-1)
    return embedding


def load_campplus_cn_common(
    checkpoint_path: str | Path,
    device: str | torch.device = "cpu",
) -> CAMPPlus:
    """Load the CAM++ zh-cn 192-dim common checkpoint into a CAMPPlus instance.

    The ``iic/speech_campplus_sv_zh-cn_16k-common`` checkpoint from
    ModelScope is distributed as a plain ``state_dict`` .bin file.
    Keeps the returned model in ``eval()`` mode so frozen inference is
    the default; the caller can explicitly ``.train()`` if they want.
    """
    model = CAMPPlus(feat_dim=FEAT_DIM, embedding_size=EMBEDDING_SIZE)
    state = torch.load(str(checkpoint_path), map_location=device, weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model
