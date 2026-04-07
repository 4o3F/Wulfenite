"""BSRNN separator + CAM++ (zh-cn) speaker encoder.

Phase 0a of the Chinese fine-tune plan:
- Load the released WeSep BSRNN English checkpoint (``assets/avg_model.pt``).
- Delete the English ECAPA-TDNN speaker encoder.
- Plug in the Chinese CAM++ 192-dim encoder.
- Re-initialize the single ``SpeakerFuseLayer`` linear projection
  (Linear(192 -> 128), ~24k parameters). Everything else is frozen.
- Train only that one linear layer on Chinese 2-speaker mixtures so the
  frozen BSRNN separator learns to consume CAM++ embeddings.

The wrapper exposes ``forward(mixture, enrollment_wav)`` taking raw waveforms
in both arguments — the enrollment side runs Kaldi-style FBank internally to
feed CAM++, which is a different feature pipeline from WeSep's original
custom FBank used by ECAPA.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch
import torchaudio.compliance.kaldi as kaldi
from torch import nn

from campplus import CAMPPlus
from wesep_bsrnn import WeSepBSRNN, load_released_checkpoint


# ---------------------------------------------------------------------------
# Feature extraction (Kaldi FBank, matches 3D-Speaker's CAM++ pipeline)
# ---------------------------------------------------------------------------


def compute_kaldi_fbank(waveform: torch.Tensor, sample_rate: int = 16000,
                        num_mel_bins: int = 80, dither: float = 0.0,
                        mean_norm: bool = True) -> torch.Tensor:
    """Compute 80-dim Kaldi FBank for a single-channel waveform.

    Args:
        waveform: [T] or [1, T] float tensor in [-1, 1].
        sample_rate: 16000.
        num_mel_bins: 80 (CAM++ default).
        dither: 0.0 at inference/validation time.
        mean_norm: utterance-level mean subtraction, matches 3D-Speaker's
            ``FBank(mean_nor=True)`` in ``speakerlab/process/processor.py``.

    Returns:
        [T_frames, num_mel_bins] tensor.
    """
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    elif waveform.dim() == 2 and waveform.size(0) > 1:
        waveform = waveform[:1]
    feat = kaldi.fbank(
        waveform,
        num_mel_bins=num_mel_bins,
        sample_frequency=sample_rate,
        dither=dither,
    )
    if mean_norm:
        feat = feat - feat.mean(dim=0, keepdim=True)
    return feat


def batch_kaldi_fbank(waveforms: Iterable[torch.Tensor],
                      sample_rate: int = 16000,
                      num_mel_bins: int = 80,
                      dither: float = 0.0) -> torch.Tensor:
    """FBank a list of variable-length waveforms into a padded batch.

    Returns ``[B, T_max, num_mel_bins]`` with zero-padding on the right.
    """
    feats = [
        compute_kaldi_fbank(w, sample_rate, num_mel_bins, dither=dither)
        for w in waveforms
    ]
    max_len = max(f.size(0) for f in feats)
    n_bins = feats[0].size(1)
    out = feats[0].new_zeros(len(feats), max_len, n_bins)
    for i, f in enumerate(feats):
        out[i, : f.size(0)] = f
    return out


# ---------------------------------------------------------------------------
# The wrapper model
# ---------------------------------------------------------------------------


class BSRNNCampPlus(nn.Module):
    """WeSep BSRNN separator with CAM++ as the speaker encoder.

    Forward: ``(mixture_wav, enrollment_wav) -> estimated_wav``

    - ``mixture_wav``: [B, T_mix], 16 kHz float tensor.
    - ``enrollment_wav``: [B, T_enr], 16 kHz float tensor. Variable T_enr is
      not supported per-batch — use padded batches where all samples share
      the same enrollment length, or B=1. The training pipeline guarantees
      fixed-length enrollment chunks.
    """

    def __init__(self, bsrnn: WeSepBSRNN, campplus: CAMPPlus):
        super().__init__()
        # Delete the English ECAPA; it is never used again.
        if hasattr(bsrnn, "spk_model"):
            del bsrnn.spk_model

        # Move everything else over as submodules.
        self.sr = bsrnn.sr
        self.win = bsrnn.win
        self.stride = bsrnn.stride
        self.enc_dim = bsrnn.enc_dim
        self.feature_dim = bsrnn.feature_dim
        self.spk_emb_dim = bsrnn.spk_emb_dim
        self.band_width = list(bsrnn.band_width)
        self.nband = bsrnn.nband

        self.BN = bsrnn.BN
        self.separator = bsrnn.separator
        self.mask = bsrnn.mask

        # New Chinese encoder.
        self.spk_model = campplus

        # Re-initialize the fuse-layer projection so it does not carry
        # ECAPA-specific weights. This is the only thing Phase 0a trains.
        fuse = self.separator.separation[0].fc.linear  # nn.Linear
        nn.init.kaiming_normal_(fuse.weight)
        if fuse.bias is not None:
            nn.init.zeros_(fuse.bias)

    # ------------------------------------------------------------------
    # Freezing helpers
    # ------------------------------------------------------------------

    def freeze_all_except_fuse_layer(self) -> None:
        """Phase 0a: only the fuse layer is trainable."""
        for p in self.parameters():
            p.requires_grad = False
        for p in self.fuse_layer_parameters():
            p.requires_grad = True

    def freeze_only_campplus(self) -> None:
        """Phase 0b: CAM++ stays frozen (it's already good at Chinese
        speakers and we don't want to destabilize a 7M-param encoder on
        limited data). Everything else — fuse layer, BSRNN separator,
        band-BN layers, mask heads — is trainable."""
        for p in self.parameters():
            p.requires_grad = True
        for p in self.spk_model.parameters():
            p.requires_grad = False

    def fuse_layer_parameters(self):
        """Return the parameters of the single ``Linear(192, feature_dim)``."""
        return self.separator.separation[0].fc.linear.parameters()

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def _compute_spk_embedding(self, enrollment_wav: torch.Tensor) -> torch.Tensor:
        """Run the Kaldi FBank front-end then CAM++ on enrollment waveforms."""
        # enrollment_wav: [B, T]
        feats = []
        for i in range(enrollment_wav.size(0)):
            f = compute_kaldi_fbank(enrollment_wav[i], sample_rate=self.sr,
                                    num_mel_bins=80, dither=0.0)
            feats.append(f)
        max_len = max(f.size(0) for f in feats)
        n_bins = feats[0].size(1)
        batched = feats[0].new_zeros(len(feats), max_len, n_bins)
        for i, f in enumerate(feats):
            batched[i, : f.size(0)] = f
        return self.spk_model(batched)  # [B, 192]

    def forward(self, mixture: torch.Tensor, enrollment_wav: torch.Tensor) -> torch.Tensor:
        if mixture.dim() != 2:
            raise ValueError(f"Expected mixture shape [B, T], got {tuple(mixture.shape)}")
        if enrollment_wav.dim() != 2:
            raise ValueError(
                f"Expected enrollment shape [B, T], got {tuple(enrollment_wav.shape)}"
            )

        batch_size, num_samples = mixture.shape

        # --- CAM++ speaker embedding ---
        spk_embedding = self._compute_spk_embedding(enrollment_wav)  # [B, 192]
        spk_embedding = spk_embedding.unsqueeze(1).unsqueeze(3)  # [B, 1, 192, 1]

        # --- BSRNN spectral front-end (copied from WeSepBSRNN.forward) ---
        window = torch.hann_window(self.win, dtype=mixture.dtype, device=mixture.device)
        spec = torch.stft(
            mixture,
            n_fft=self.win,
            hop_length=self.stride,
            window=window,
            return_complex=True,
        )
        spec_ri = torch.stack([spec.real, spec.imag], dim=1)

        subband_spec = []
        subband_mix_spec = []
        band_idx = 0
        for band_width in self.band_width:
            subband_spec.append(
                spec_ri[:, :, band_idx : band_idx + band_width].contiguous()
            )
            subband_mix_spec.append(spec[:, band_idx : band_idx + band_width])
            band_idx += band_width

        subband_feature = []
        for band, bn in zip(subband_spec, self.BN):
            subband_feature.append(
                bn(band.view(batch_size, band.size(1) * band.size(2), -1))
            )
        subband_feature = torch.stack(subband_feature, dim=1)

        sep_output = self.separator(subband_feature, spk_embedding)

        separated_bands = []
        for idx, mask_head in enumerate(self.mask):
            band_out = mask_head(sep_output[:, idx]).view(
                batch_size, 2, 2, self.band_width[idx], -1
            )
            mask = band_out[:, 0] * torch.sigmoid(band_out[:, 1])
            mask_real = mask[:, 0]
            mask_imag = mask[:, 1]
            est_real = (
                subband_mix_spec[idx].real * mask_real
                - subband_mix_spec[idx].imag * mask_imag
            )
            est_imag = (
                subband_mix_spec[idx].real * mask_imag
                + subband_mix_spec[idx].imag * mask_real
            )
            separated_bands.append(torch.complex(est_real, est_imag))

        est_spec = torch.cat(separated_bands, dim=1)
        output = torch.istft(
            est_spec.view(batch_size, self.enc_dim, -1),
            n_fft=self.win,
            hop_length=self.stride,
            window=window,
            length=num_samples,
        )
        return output


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


def build_bsrnn_campplus(
    bsrnn_ckpt: str | Path,
    campplus_ckpt: str | Path,
    device: str | torch.device = "cpu",
) -> BSRNNCampPlus:
    """Construct a ready-to-train BSRNNCampPlus model.

    - Loads the released English WeSep BSRNN checkpoint into a WeSepBSRNN
      instance (which includes the ECAPA that we will throw away).
    - Loads the Chinese CAM++ checkpoint into a standalone CAMPPlus.
    - Builds the wrapper, which deletes ECAPA and re-inits the fuse layer.
    - Moves to device and returns.
    """
    bsrnn = WeSepBSRNN()
    load_released_checkpoint(bsrnn, bsrnn_ckpt)

    from campplus import load_campplus_cn_common

    cam = load_campplus_cn_common(campplus_ckpt, device="cpu")

    model = BSRNNCampPlus(bsrnn, cam)
    model.to(device)
    return model
