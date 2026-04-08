"""wulfenite.models — architecture definitions.

Public entry points:
- :class:`WulfeniteTSE` — end-to-end TSE model (CAM++ + SpeakerBeam-SS).
- :class:`SpeakerBeamSS` / :class:`SpeakerBeamSSConfig` — separator alone.
- :class:`CAMPPlus` / :func:`encode_enrollment` — speaker encoder alone.
- :class:`S4D` — diagonal state-space layer (parallel + step forms).

Internal building blocks (CausalConv1d, ChannelwiseLayerNorm, TCNBlock,
S4DBlock) are available via ``wulfenite.models.layers`` and
``wulfenite.models.speakerbeam_ss`` if needed, but callers should prefer
the higher-level ``WulfeniteTSE`` wrapper.
"""

from .campplus import (
    CAMPPlus,
    compute_fbank,
    encode_enrollment,
    load_campplus_cn_common,
)
from .s4d import S4D
from .speakerbeam_ss import SpeakerBeamSS, SpeakerBeamSSConfig
from .tse import WulfeniteTSE

__all__ = [
    "CAMPPlus",
    "compute_fbank",
    "encode_enrollment",
    "load_campplus_cn_common",
    "S4D",
    "SpeakerBeamSS",
    "SpeakerBeamSSConfig",
    "WulfeniteTSE",
]
