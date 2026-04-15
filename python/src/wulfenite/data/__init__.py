"""wulfenite.data — dataset, augmentation, and PSE mixing utilities.

Public entry points:
- :func:`scan_aishell1` / :func:`scan_aishell3` / :func:`scan_magicdata` /
  :func:`scan_cnceleb` / :func:`merge_speaker_dicts`
- :func:`scan_noise_dir` / :func:`scan_noise_dirs`
- :class:`PSEMixer`
- :func:`synth_room_rir` / :func:`apply_rir` / :func:`add_noise_at_snr` /
  :func:`add_gaussian_noise` / :func:`scale_noise_to_snr` /
  :class:`ReverbConfig`
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
    RoomPreset,
    ReverbConfig,
    add_gaussian_noise,
    add_noise_at_snr,
    apply_rir,
    scale_noise_to_snr,
    synth_room_rir,
)
from .noise import NoiseEntry, scan_noise_dir, scan_noise_dirs
from .pse_mixer import PSEMixer

__all__ = [
    "AudioEntry",
    "merge_speaker_dicts",
    "scan_aishell1",
    "scan_aishell3",
    "scan_magicdata",
    "scan_cnceleb",
    "RoomPreset",
    "ReverbConfig",
    "add_gaussian_noise",
    "add_noise_at_snr",
    "apply_rir",
    "scale_noise_to_snr",
    "synth_room_rir",
    "NoiseEntry",
    "scan_noise_dir",
    "scan_noise_dirs",
    "PSEMixer",
]
