"""wulfenite.models — architecture definitions.

Public entry points:
- :class:`WulfeniteTSE` — end-to-end TSE model (speaker encoder + separator).
- :class:`SpeakerBeamSS` / :class:`SpeakerBeamSSConfig` — separator alone.
- :class:`LearnableDVector` — trainable speaker encoder for Plan C5.
- :class:`S4D` — diagonal state-space layer (parallel + step forms).

Internal building blocks (CausalConv1d, ChannelwiseLayerNorm, TCNBlock,
S4DBlock) are available via ``wulfenite.models.layers`` and
``wulfenite.models.speakerbeam_ss`` if needed, but callers should prefer
the higher-level ``WulfeniteTSE`` wrapper.
"""

from .dvector import LearnableDVector, SpecAugment, compute_fbank_batch
from .campplus_encoder import CampPlusSpeakerEncoder, SpeakerEncoderOutput
from .s4d import S4D
from .speakerbeam_ss import SpeakerBeamSS, SpeakerBeamSSConfig
from .tse import WulfeniteTSE

__all__ = [
    "CampPlusSpeakerEncoder",
    "LearnableDVector",
    "SpeakerEncoderOutput",
    "SpecAugment",
    "compute_fbank_batch",
    "S4D",
    "SpeakerBeamSS",
    "SpeakerBeamSSConfig",
    "WulfeniteTSE",
]
