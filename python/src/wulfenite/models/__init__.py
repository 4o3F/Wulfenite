"""wulfenite.models — pDFNet2+ model stack."""

from .deep_filtering import DfOp
from .dfnet2 import DfDecoder, DfNet, DfNetStreamState, Encoder, ErbDecoder
from .ecapa_tdnn import (
    ECAPA_TDNN,
    ECAPA_TDNN_GLOB_c1024,
    ECAPA_TDNN_GLOB_c512,
    ECAPA_TDNN_c1024,
    ECAPA_TDNN_c512,
    detect_ecapa_variant,
)
from .erb import erb2freq, erb_fb, erb_fb_inverse, freq2erb
from .modules import (
    Conv2dNormAct,
    ConvTranspose,
    ConvTranspose2dNormAct,
    GroupedGRU,
    GroupedLinear,
    SepConv2d,
    SqueezedGRU,
)
from .pdfnet2 import PDfNet2
from .pdfnet2_plus import PDfNet2Plus
from .speaker_encoder import SpeakerEncoder
from .tiny_ecapa import ConvBlock, SEBlock, TinyECAPA

__all__ = [
    "freq2erb",
    "erb2freq",
    "erb_fb",
    "erb_fb_inverse",
    "DfOp",
    "GroupedLinear",
    "GroupedGRU",
    "ECAPA_TDNN",
    "ECAPA_TDNN_GLOB_c1024",
    "ECAPA_TDNN_GLOB_c512",
    "ECAPA_TDNN_c1024",
    "ECAPA_TDNN_c512",
    "Conv2dNormAct",
    "ConvTranspose2dNormAct",
    "ConvTranspose",
    "SepConv2d",
    "SqueezedGRU",
    "Encoder",
    "ErbDecoder",
    "DfDecoder",
    "DfNet",
    "DfNetStreamState",
    "PDfNet2",
    "SEBlock",
    "ConvBlock",
    "TinyECAPA",
    "PDfNet2Plus",
    "SpeakerEncoder",
    "detect_ecapa_variant",
]
