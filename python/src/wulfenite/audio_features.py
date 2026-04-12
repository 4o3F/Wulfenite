"""Shared audio feature extraction helpers."""

from __future__ import annotations

import torch
import torch.nn.functional as F
import torchaudio.compliance.kaldi as kaldi


SAMPLE_RATE = 16000
FEAT_DIM = 80


def compute_fbank_batch(
    waveform: torch.Tensor,
    sample_rate: int = SAMPLE_RATE,
    num_mel_bins: int = FEAT_DIM,
    dither: float = 0.0,
    mean_norm: bool = True,
) -> torch.Tensor:
    """Compute Kaldi FBank features for ``[B, T]`` or ``[T]`` waveforms."""
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
