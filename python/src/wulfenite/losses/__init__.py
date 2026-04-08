"""wulfenite.losses — training loss components.

Public entry points:
- :class:`WulfeniteLoss` / :class:`LossWeights` — the combined
  mask-aware loss used by the training loop.
- :func:`sdr_loss` — direct non-scale-invariant SDR.
- :class:`MultiResolutionSTFTLoss` / :class:`STFTLoss` — frequency
  supervision.
- :func:`target_absent_loss` — energy penalty for silent-target samples.
- :func:`presence_loss` — BCE on the target-presence head.
"""

from .combined import LossParts, LossWeights, WulfeniteLoss
from .mr_stft import MultiResolutionSTFTLoss, STFTLoss
from .presence import presence_loss
from .sdr import sdr_loss
from .silence import target_absent_loss

__all__ = [
    "LossParts",
    "LossWeights",
    "WulfeniteLoss",
    "MultiResolutionSTFTLoss",
    "STFTLoss",
    "presence_loss",
    "sdr_loss",
    "target_absent_loss",
]
