"""wulfenite.data — training data pipeline.

Public entry points:
- :class:`WulfeniteMixer` / :class:`MixerConfig` / :func:`collate_mixer_batch`
- :func:`scan_aishell1` / :func:`scan_aishell3` / :func:`merge_speaker_dicts`
- :func:`scan_dns_noise`
- :func:`synth_room_rir` / :func:`apply_rir` / :func:`add_noise_at_snr` /
  :func:`add_gaussian_noise` / :class:`ReverbConfig`
- :class:`AudioEntry` / :class:`NoiseEntry`
"""

from .aishell import (
    AudioEntry,
    merge_speaker_dicts,
    scan_aishell1,
    scan_aishell3,
)
from .augmentation import (
    ReverbConfig,
    add_gaussian_noise,
    add_noise_at_snr,
    apply_rir,
    synth_room_rir,
)
from .dns_noise import NoiseEntry, scan_dns_noise
from .mixer import MixerConfig, WulfeniteMixer, collate_mixer_batch

__all__ = [
    "AudioEntry",
    "merge_speaker_dicts",
    "scan_aishell1",
    "scan_aishell3",
    "ReverbConfig",
    "add_gaussian_noise",
    "add_noise_at_snr",
    "apply_rir",
    "synth_room_rir",
    "NoiseEntry",
    "scan_dns_noise",
    "MixerConfig",
    "WulfeniteMixer",
    "collate_mixer_batch",
]
