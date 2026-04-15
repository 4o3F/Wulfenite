"""TinyECAPA speaker encoder used for pDFNet2+ refinement."""

from __future__ import annotations

import math

import torch
from torch import nn
from torchaudio.functional import compute_deltas

from wulfenite.audio_features import compute_fbank_batch


class SEBlock(nn.Module):
    """Squeeze-excitation over the temporal axis."""

    def __init__(self, channels: int, se_channels: int) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Conv1d(channels, se_channels, kernel_size=1)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv1d(se_channels, channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.pool(x)
        scale = self.fc2(self.relu(self.fc1(scale)))
        return x * self.sigmoid(scale)


class ConvBlock(nn.Module):
    """Depthwise-separable temporal block with squeeze-excitation."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, se_ch: int) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.depthwise = nn.Conv1d(
            in_ch,
            in_ch,
            kernel_size=kernel_size,
            padding=padding,
            groups=in_ch,
            bias=False,
        )
        self.pointwise = nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm1d(out_ch)
        self.prelu = nn.PReLU(out_ch)
        self.se = SEBlock(out_ch, se_ch)
        if in_ch == out_ch:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=False)
        self.shortcut_bn = nn.BatchNorm1d(out_ch) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut_bn(self.shortcut(x))
        y = self.depthwise(x)
        y = self.prelu(self.bn(self.pointwise(y)))
        y = self.se(y)
        return self.prelu(y + residual)


class TinyECAPA(nn.Module):
    """Lightweight speaker encoder with chunk-wise inference support."""

    def __init__(self, sample_rate: int = 16000) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.conv_block1 = ConvBlock(80, 80, kernel_size=3, se_ch=20)
        self.conv_block2 = ConvBlock(80, 128, kernel_size=5, se_ch=32)
        self.conv_block3 = ConvBlock(128, 192, kernel_size=7, se_ch=48)
        self.bn_stats = nn.BatchNorm1d(384)
        self.fc = nn.Linear(384, 192)
        self.bn_out = nn.BatchNorm1d(192)

    def extract_features(self, waveform: torch.Tensor) -> torch.Tensor:
        """Compute 80-D streaming-friendly speaker features."""
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        if waveform.dim() != 2:
            raise ValueError(
                f"waveform must be [T] or [B, T], got {tuple(waveform.shape)}"
            )
        fbank = compute_fbank_batch(waveform, sample_rate=self.sample_rate, num_mel_bins=80)
        # The paper stack is MFCC + delta + delta^2; in this reset tree we reuse the
        # existing 80-D FBank helper so the speaker path stays dependency-light.
        delta = compute_deltas(fbank.transpose(1, 2))
        delta2 = compute_deltas(delta)
        features = (fbank.transpose(1, 2) + 0.5 * delta + 0.25 * delta2) / 1.75
        return features.transpose(1, 2)

    def forward_features(self, features: torch.Tensor) -> torch.Tensor:
        if features.dim() != 3 or features.size(-1) != 80:
            raise ValueError(
                f"features must have shape [B, T, 80], got {tuple(features.shape)}"
            )
        x = features.transpose(1, 2)
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        mean = x.mean(dim=-1)
        std = x.var(dim=-1, unbiased=False).add(1e-5).sqrt()
        pooled = torch.cat((mean, std), dim=1)
        pooled = self.bn_stats(pooled)
        emb = self.fc(pooled)
        emb = self.bn_out(emb)
        return torch.nn.functional.normalize(emb, dim=-1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.dim() == 2:
            features = self.extract_features(inputs)
        elif inputs.dim() == 3:
            features = inputs
        else:
            raise ValueError(
                f"inputs must be waveform [B, T] or features [B, T, 80], got {tuple(inputs.shape)}"
            )
        return self.forward_features(features)

    def forward_chunks(
        self,
        waveform: torch.Tensor,
        *,
        chunk_seconds: float = 1.0,
        overlap: float = 0.5,
    ) -> torch.Tensor:
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        if waveform.dim() != 2:
            raise ValueError(
                f"waveform must be [T] or [B, T], got {tuple(waveform.shape)}"
            )
        if not 0.0 <= overlap < 1.0:
            raise ValueError(f"overlap must be in [0, 1), got {overlap}")
        chunk_len = max(1, int(round(chunk_seconds * self.sample_rate)))
        step = max(1, int(round(chunk_len * (1.0 - overlap))))
        total = waveform.size(-1)
        starts = list(range(0, max(total - chunk_len, 0) + 1, step))
        if not starts:
            starts = [0]
        last_start = max(0, total - chunk_len)
        if starts[-1] != last_start:
            starts.append(last_start)

        chunks: list[torch.Tensor] = []
        for start in starts:
            chunk = waveform[:, start : start + chunk_len]
            if chunk.size(-1) < chunk_len:
                chunk = torch.nn.functional.pad(chunk, (0, chunk_len - chunk.size(-1)))
            chunks.append(chunk)
        batch = torch.cat(chunks, dim=0)
        emb = self.forward(batch)
        emb = emb.view(len(chunks), waveform.size(0), -1).permute(1, 2, 0)
        return emb

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


__all__ = ["SEBlock", "ConvBlock", "TinyECAPA"]
