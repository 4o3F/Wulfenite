"""Learnable d-vector speaker encoder for Plan C5.

This module keeps the CAM++ path intact while providing a small
trainable x-vector-style encoder that can be optimized jointly with
the separator. The classifier head regularizes the raw embedding;
the separator always consumes the L2-normalized embedding.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import torchaudio.compliance.kaldi as kaldi
from torch import nn

from .campplus import FEAT_DIM, SAMPLE_RATE, StatsPool, TDNNLayer


def compute_fbank_batch(
    waveform: torch.Tensor,
    sample_rate: int = SAMPLE_RATE,
    num_mel_bins: int = FEAT_DIM,
    dither: float = 0.0,
    mean_norm: bool = True,
) -> torch.Tensor:
    """Compute Kaldi FBank features for ``[B, T]`` or ``[T]`` waveforms.

    The existing CAM++ helper is single-utterance. This batch-aware
    variant preserves the same per-item feature extraction settings and
    pads shorter sequences to the batch maximum frame count.
    """
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    if waveform.dim() != 2:
        raise ValueError(
            f"waveform must be [T] or [B, T], got {tuple(waveform.shape)}"
        )

    out_device = waveform.device
    features: list[torch.Tensor] = []
    max_frames = 0
    for i in range(waveform.size(0)):
        feat = kaldi.fbank(
            waveform[i : i + 1].cpu(),
            num_mel_bins=num_mel_bins,
            sample_frequency=sample_rate,
            dither=dither,
        )
        if mean_norm:
            feat = feat - feat.mean(dim=0, keepdim=True)
        features.append(feat)
        max_frames = max(max_frames, feat.size(0))

    padded = torch.stack(
        [F.pad(feat, (0, 0, 0, max_frames - feat.size(0))) for feat in features],
        dim=0,
    )
    return padded.to(out_device)


class SpecAugment(nn.Module):
    """Light time/frequency masking for enrollment FBank features."""

    def __init__(
        self,
        time_mask: int = 20,
        freq_mask: int = 10,
        num_time: int = 2,
        num_freq: int = 2,
    ) -> None:
        super().__init__()
        self.time_mask = time_mask
        self.freq_mask = freq_mask
        self.num_time = num_time
        self.num_freq = num_freq

    def forward(self, fbank: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return fbank
        if fbank.dim() != 3:
            raise ValueError(f"fbank must be [B, T, F], got {tuple(fbank.shape)}")

        out = fbank.clone()
        batch, frames, bins = out.shape
        for b in range(batch):
            for _ in range(self.num_time):
                width = int(
                    torch.randint(
                        0, self.time_mask + 1, (1,), device=out.device
                    ).item()
                )
                if width == 0:
                    continue
                start = int(
                    torch.randint(
                        0,
                        max(1, frames - width + 1),
                        (1,),
                        device=out.device,
                    ).item()
                )
                out[b, start : start + width, :] = 0
            for _ in range(self.num_freq):
                width = int(
                    torch.randint(
                        0, self.freq_mask + 1, (1,), device=out.device
                    ).item()
                )
                if width == 0:
                    continue
                start = int(
                    torch.randint(
                        0,
                        max(1, bins - width + 1),
                        (1,),
                        device=out.device,
                    ).item()
                )
                out[b, :, start : start + width] = 0
        return out


class LearnableDVector(nn.Module):
    """Small x-vector-style speaker encoder used in Plan C5."""

    def __init__(
        self,
        num_speakers: int | None = None,
        emb_dim: int = 192,
        tdnn_channels: int = 192,
        stats_dim: int = 576,
        feat_dim: int = FEAT_DIM,
        spec_augment: bool = True,
    ) -> None:
        super().__init__()
        channels = tdnn_channels

        self.frame = nn.Sequential(
            TDNNLayer(feat_dim, channels, kernel_size=5, padding=-1),
            TDNNLayer(channels, channels, kernel_size=3, dilation=2, padding=-1),
            TDNNLayer(channels, channels, kernel_size=3, dilation=3, padding=-1),
            TDNNLayer(channels, channels, kernel_size=1),
            TDNNLayer(channels, stats_dim, kernel_size=1),
        )
        self.pool = StatsPool()
        self.embed = nn.Sequential(
            nn.Linear(2 * stats_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, emb_dim),
        )
        self.classifier = (
            nn.Linear(emb_dim, num_speakers) if num_speakers is not None else None
        )
        self.spec_augment = SpecAugment() if spec_augment else None

    def forward(
        self,
        fbank: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Encode ``[B, T, F]`` log-mel features into d-vectors."""
        if fbank.dim() != 3:
            raise ValueError(f"fbank must be [B, T, F], got {tuple(fbank.shape)}")
        if self.spec_augment is not None:
            fbank = self.spec_augment(fbank)
        x = fbank.permute(0, 2, 1)  # [B, F, T]
        x = self.frame(x)
        x = self.pool(x)
        raw_emb = self.embed(x)
        norm_emb = F.normalize(raw_emb, p=2, dim=-1)
        logits = self.classifier(raw_emb) if self.classifier is not None else None
        return raw_emb, norm_emb, logits

    @torch.no_grad()
    def encode_enrollment(self, waveform: torch.Tensor) -> torch.Tensor:
        """Drop-in raw-waveform helper matching the CAM++ API."""
        was_training = self.training
        self.eval()
        fbank = compute_fbank_batch(waveform)
        _, norm_emb, _ = self.forward(fbank)
        if was_training:
            self.train()
        return norm_emb
