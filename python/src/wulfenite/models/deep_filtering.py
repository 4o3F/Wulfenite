"""Deep filtering operator for complex spectrograms."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class DfOp(nn.Module):
    """Apply causal deep-filter coefficients to a real-valued complex STFT."""

    def __init__(
        self,
        df_bins: int = 96,
        df_order: int = 5,
        df_lookahead: int = 0,
        method: str = "real_unfold",
    ) -> None:
        super().__init__()
        if df_bins <= 0:
            raise ValueError(f"df_bins must be positive, got {df_bins}")
        if df_order <= 0:
            raise ValueError(f"df_order must be positive, got {df_order}")
        if df_lookahead < 0:
            raise ValueError(f"df_lookahead must be non-negative, got {df_lookahead}")
        if method != "real_unfold":
            raise NotImplementedError(
                "Only the 'real_unfold' DfOp path is implemented in Wulfenite."
            )
        self.df_bins = df_bins
        self.df_order = df_order
        self.df_lookahead = df_lookahead
        self.method = method

    def forward(
        self,
        spec: torch.Tensor,
        coefs: torch.Tensor,
        alpha: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if spec.dim() != 5 or spec.size(1) != 1 or spec.size(-1) != 2:
            raise ValueError(
                "spec must have shape [B, 1, T, F, 2], "
                f"got {tuple(spec.shape)}"
            )
        if coefs.dim() != 5 or coefs.size(2) != self.df_order or coefs.size(-1) != 2:
            raise ValueError(
                "coefs must have shape [B, T, O, F_df, 2], "
                f"got {tuple(coefs.shape)}"
            )
        if spec.size(0) != coefs.size(0) or spec.size(2) != coefs.size(1):
            raise ValueError(
                f"spec {tuple(spec.shape)} and coefs {tuple(coefs.shape)} "
                "do not agree on batch/time dimensions"
            )
        if coefs.size(3) > spec.size(3):
            raise ValueError(
                f"coefs df_bins={coefs.size(3)} exceed spec freq bins={spec.size(3)}"
            )
        if alpha is not None and alpha.shape != (spec.size(0), spec.size(2), 1):
            raise ValueError(
                "alpha must have shape [B, T, 1], "
                f"got {tuple(alpha.shape)}"
            )

        df_bins = min(self.df_bins, coefs.size(3), spec.size(3))
        left_pad = self.df_order - self.df_lookahead - 1
        right_pad = self.df_lookahead
        padded = F.pad(
            spec[..., :df_bins, :].squeeze(1),
            (0, 0, 0, 0, left_pad, right_pad),
        )
        unfolded = padded.unfold(1, self.df_order, 1).permute(0, 1, 4, 2, 3)
        unfolded = unfolded[..., :df_bins, :]
        coefs = coefs[..., :df_bins, :]

        real = unfolded[..., 0] * coefs[..., 0] - unfolded[..., 1] * coefs[..., 1]
        imag = unfolded[..., 1] * coefs[..., 0] + unfolded[..., 0] * coefs[..., 1]
        filtered = torch.stack((real, imag), dim=-1).sum(dim=2).unsqueeze(1)

        output = spec.clone()
        if alpha is None:
            output[..., :df_bins, :] = filtered
        else:
            mix = alpha.view(spec.size(0), 1, spec.size(2), 1, 1)
            output[..., :df_bins, :] = filtered * mix + spec[..., :df_bins, :] * (1.0 - mix)
        return output


__all__ = ["DfOp"]
