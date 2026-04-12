"""wulfenite.data — training data pipeline.

Public entry points:
- :class:`WulfeniteMixer` / :class:`MixerConfig` / :func:`collate_mixer_batch`
- :func:`scan_aishell1` / :func:`scan_aishell3` / :func:`scan_magicdata` /
  :func:`scan_cnceleb` /
  :func:`merge_speaker_dicts`
- :func:`scan_noise_dir`
- :func:`synth_room_rir` / :func:`apply_rir` / :func:`add_noise_at_snr` /
  :func:`add_gaussian_noise` / :class:`ReverbConfig`
- :class:`AudioEntry` / :class:`NoiseEntry`
"""

from .aishell import (
    AudioEntry,
    merge_speaker_dicts,
    scan_aishell1,
    scan_aishell3,
    scan_cnceleb,
    scan_magicdata,
)
from .augmentation import (
    ReverbConfig,
    add_gaussian_noise,
    add_noise_at_snr,
    apply_rir,
    synth_room_rir,
)
from .composer import ClipFamily, ComposerConfig, EventType
from .mixer import MixerConfig, WulfeniteMixer, collate_mixer_batch
from .noise import NoiseEntry, scan_noise_dir

__all__ = [
    "AudioEntry",
    "merge_speaker_dicts",
    "scan_aishell1",
    "scan_aishell3",
    "scan_magicdata",
    "scan_cnceleb",
    "ReverbConfig",
    "add_gaussian_noise",
    "add_noise_at_snr",
    "apply_rir",
    "synth_room_rir",
    "ClipFamily",
    "ComposerConfig",
    "EventType",
    "NoiseEntry",
    "scan_noise_dir",
    "MixerConfig",
    "WulfeniteMixer",
    "collate_mixer_batch",
]
