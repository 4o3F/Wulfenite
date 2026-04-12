"""wulfenite.losses — training loss components.

Public entry points:
- :class:`WulfeniteLoss` / :class:`LossWeights` — the combined
  mask-aware loss used by the training loop.
- :func:`sdr_loss` — direct non-scale-invariant SDR.
- :class:`MultiResolutionSTFTLoss` / :class:`STFTLoss` — frequency
  supervision.
- :func:`target_recall_loss` — anti-suppression floor on target-active frames.
- :func:`target_inactive_loss` — energy penalty on target-inactive frames.
- :func:`target_absent_loss` — energy penalty for silent-target samples.
- :func:`presence_loss` — BCE on the target-presence head.
"""

from .combined import LossParts, LossWeights, WulfeniteLoss
from .inactive import target_inactive_loss
from .mr_stft import MultiResolutionSTFTLoss, STFTLoss
from .presence import presence_loss
from .recall import target_recall_loss
from .sdr import compute_sdr_db, compute_sdri_db, sdr_loss
from .silence import target_absent_loss

__all__ = [
    "LossParts",
    "LossWeights",
    "WulfeniteLoss",
    "MultiResolutionSTFTLoss",
    "STFTLoss",
    "presence_loss",
    "target_recall_loss",
    "target_inactive_loss",
    "compute_sdr_db",
    "compute_sdri_db",
    "sdr_loss",
    "target_absent_loss",
]
