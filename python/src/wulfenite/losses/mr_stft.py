"""Multi-resolution STFT loss.

A frequency-domain supervisory signal that complements the time-
domain :func:`wulfenite.losses.sdr.sdr_loss`. Following Yamamoto
et al., "Parallel WaveGAN: A fast waveform generation model based
on generative adversarial networks with multi-resolution spectrogram"
(ICASSP 2020), we compute STFT at several window sizes and sum two
terms per resolution:

1. **Spectral convergence** — normalized Frobenius-norm error on
   the magnitude spectrogram, good at picking up broadband
   reconstruction errors.
2. **Log-magnitude L1** — L1 on ``log(|STFT|)``, stable near zero
   and more sensitive to low-energy frequency bands.

Using several resolutions at once gives the model supervision at
different time-frequency trade-offs (small windows favor transient
accuracy, large windows favor stationary tones). This is standard in
neural vocoders and source-separation systems to get rid of coloured
residual noise that a pure time-domain SDR loss would ignore.

The loss is **not** scale-invariant, so it combines cleanly with the
direct SDR loss without reintroducing the Phase 0a/0b degenerate
solutions described in ``docs/architecture.md``.
"""

from __future__ import annotations

import torch
from torch import nn


def _stft_magnitude(
    x: torch.Tensor,
    n_fft: int,
    hop_length: int,
    win_length: int,
    window: torch.Tensor,
) -> torch.Tensor:
    """Compute |STFT(x)| with a cached window.

    Args:
        x: ``[B, T]`` waveform.
        n_fft / hop_length / win_length: standard STFT parameters.
        window: ``[win_length]`` window tensor on the same device as ``x``.

    Returns:
        ``[B, F, T_frames]`` real magnitude spectrogram.
    """
    spec = torch.stft(
        x,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        return_complex=True,
        center=True,
    )
    # |a + bi| = sqrt(a^2 + b^2)
    return torch.sqrt(spec.real.pow(2) + spec.imag.pow(2) + 1e-9)


class STFTLoss(nn.Module):
    """Single-resolution STFT loss (spectral convergence + log-mag L1).

    Internally used by :class:`MultiResolutionSTFTLoss`. Exposed for
    users who want to build custom resolution sets.
    """

    def __init__(
        self,
        n_fft: int,
        hop_length: int,
        win_length: int,
        eps: float = 1e-4,
    ) -> None:
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.eps = eps
        self.register_buffer(
            "window",
            torch.hann_window(win_length),
            persistent=False,
        )

    def forward(
        self,
        estimate: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute spectral-convergence and log-magnitude L1 losses.

        Args:
            estimate: ``[B, T]`` estimated waveform.
            target: ``[B, T]`` target waveform.

        Returns:
            ``(sc_loss, log_mag_loss)`` both as scalar tensors.
        """
        est_mag = _stft_magnitude(
            estimate, self.n_fft, self.hop_length, self.win_length, self.window,
        )
        tgt_mag = _stft_magnitude(
            target, self.n_fft, self.hop_length, self.win_length, self.window,
        )

        # Spectral convergence: ||est - tgt||_F / ||tgt||_F, batched.
        diff_norm = torch.norm(est_mag - tgt_mag, p="fro", dim=(-2, -1))
        tgt_norm = torch.norm(tgt_mag, p="fro", dim=(-2, -1)) + self.eps
        sc_loss = (diff_norm / tgt_norm).clamp(max=5.0).mean()

        # Log magnitude L1.
        log_est = torch.log(est_mag + self.eps)
        log_tgt = torch.log(tgt_mag + self.eps)
        log_mag_loss = (log_est - log_tgt).abs().mean()

        return sc_loss, log_mag_loss


class MultiResolutionSTFTLoss(nn.Module):
    """Sum of STFT losses at several resolutions.

    The default configuration uses three resolutions that cover short
    (fine time), medium, and long (fine frequency) trade-offs:
    ``(512, 128, 512)``, ``(1024, 256, 1024)``, ``(2048, 512, 2048)``.

    Args:
        fft_sizes: sequence of ``n_fft`` values.
        hop_sizes: sequence of ``hop_length`` values (same length).
        win_lengths: sequence of ``win_length`` values (same length).
        sc_weight: weight on the spectral-convergence term.
        log_mag_weight: weight on the log-magnitude L1 term.
        eps: log stabilizer.
    """

    def __init__(
        self,
        fft_sizes: tuple[int, ...] = (512, 1024, 2048),
        hop_sizes: tuple[int, ...] = (128, 256, 512),
        win_lengths: tuple[int, ...] = (512, 1024, 2048),
        sc_weight: float = 0.5,
        log_mag_weight: float = 0.5,
        eps: float = 1e-4,
    ) -> None:
        super().__init__()
        if not (len(fft_sizes) == len(hop_sizes) == len(win_lengths)):
            raise ValueError(
                "fft_sizes, hop_sizes, win_lengths must have equal length"
            )
        self.sc_weight = sc_weight
        self.log_mag_weight = log_mag_weight
        self.stft_losses = nn.ModuleList([
            STFTLoss(n_fft=f, hop_length=h, win_length=w, eps=eps)
            for f, h, w in zip(fft_sizes, hop_sizes, win_lengths)
        ])

    def forward(
        self,
        estimate: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Sum the losses across all resolutions.

        Args:
            estimate: ``[B, T]`` estimated waveform.
            target: ``[B, T]`` target waveform.

        Returns:
            Scalar loss averaged over resolutions and batch.
        """
        if estimate.shape != target.shape:
            raise ValueError(
                f"estimate {tuple(estimate.shape)} vs target {tuple(target.shape)}"
            )

        sc_total = torch.zeros((), device=estimate.device, dtype=estimate.dtype)
        log_total = torch.zeros((), device=estimate.device, dtype=estimate.dtype)
        for stft_loss in self.stft_losses:
            sc, log_mag = stft_loss(estimate, target)
            sc_total = sc_total + sc
            log_total = log_total + log_mag

        n = len(self.stft_losses)
        return (self.sc_weight * sc_total + self.log_mag_weight * log_total) / n
