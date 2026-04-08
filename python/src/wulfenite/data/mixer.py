"""On-the-fly 2-speaker TSE mixer.

Generates training samples for SpeakerBeam-SS with both target-present
and target-absent branches, as described in ``docs/architecture.md``
section 5 and 6. Each ``__getitem__`` call produces a dict suitable
for feeding into :class:`wulfenite.losses.combined.WulfeniteLoss`.

### Sample composition

**Target-present branch** (``target_present_prob`` fraction):
  - Pick two distinct speakers A and B
  - Pick TWO different utterances of A:
    - one becomes the clean target
    - the other becomes the enrollment (so the model cannot cheat
      by memorizing the exact target waveform)
  - Pick one utterance of B as the interferer
  - Scale interferer to a random SNR relative to target
  - Mix ``target + scaled_interferer``
  - Optional: reverb (different RIR per source), additive noise
  - Returns ``target_present = 1`` and the reverberated target

**Target-absent branch** (``1 - target_present_prob`` fraction):
  - Pick two distinct speakers A (the *claimed* target) and B
  - Pick one utterance of A → enrollment (A will NOT appear in the
    mixture)
  - Pick one utterance of B → the entire mixture content
  - Optional: reverb on B, additive noise
  - ``target = zeros`` (the correct model output is silence),
    ``target_present = 0``

### Returned dict

Every call yields a dict with:

- ``"mixture"``: ``[T]`` float tensor, 16 kHz mono input to the model
- ``"target"``: ``[T]`` float tensor, loss reference (zeros for absent)
- ``"enrollment"``: ``[T_enr]`` float tensor, fed to CAM++
- ``"target_present"``: scalar tensor, 1.0 or 0.0
- ``"snr_db"``: scalar tensor for logging
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Sequence

import soundfile as sf
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .aishell import AudioEntry
from .augmentation import (
    ReverbConfig,
    add_noise_at_snr,
    add_gaussian_noise,
    apply_rir,
    synth_room_rir,
)
from .dns_noise import NoiseEntry


SAMPLE_RATE = 16000


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class MixerConfig:
    """Configuration bundle for :class:`WulfeniteMixer`.

    Attributes are grouped by concern. All time-in-seconds fields are
    converted to samples internally using ``sample_rate``.
    """

    sample_rate: int = SAMPLE_RATE
    segment_seconds: float = 4.0
    enrollment_seconds: float = 4.0

    # --- Mixing ---
    snr_range_db: tuple[float, float] = (-5.0, 5.0)
    target_present_prob: float = 0.85  # fraction of target-present samples

    # --- Optional acoustic augmentation ---
    apply_reverb: bool = True
    reverb_prob: float = 0.85
    reverb: ReverbConfig = field(default_factory=ReverbConfig)
    reverb_enrollment: bool = True

    # --- Additive noise on the final mixture ---
    apply_noise: bool = True
    noise_prob: float = 0.80
    noise_snr_range_db: tuple[float, float] = (10.0, 25.0)
    # If True and dns_noises is provided, sample from DNS4;
    # otherwise fall back to synthetic Gaussian noise.
    use_dns_noise: bool = True

    # --- Per-source loudness target ---
    rms_target: float = 0.1

    # --- Safety ---
    peak_clip: float = 0.99  # rescale mixture if peak exceeds this


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rms(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return torch.sqrt(torch.mean(x * x) + eps)


def _load_chunk(
    entry: AudioEntry,
    target_len: int,
    rng: random.Random,
) -> torch.Tensor:
    """Load a random ``target_len``-sample window from an entry.

    Uses the cached ``entry.num_frames`` so only ONE filesystem call
    per sample is needed (``sf.read``). Pads with zeros if the file
    is shorter than the requested chunk.
    """
    n = entry.num_frames
    if n >= target_len:
        start = rng.randint(0, n - target_len)
        data, _ = sf.read(
            str(entry.path), start=start, stop=start + target_len,
            dtype="float32", always_2d=False,
        )
        return torch.from_numpy(data)
    data, _ = sf.read(str(entry.path), dtype="float32", always_2d=False)
    pad = target_len - data.shape[0]
    return F.pad(torch.from_numpy(data), (0, pad))


def _load_noise_chunk(
    entry: NoiseEntry,
    target_len: int,
    rng: random.Random,
) -> torch.Tensor:
    """Load a random ``target_len``-sample window from a noise file."""
    n = entry.num_frames
    if n >= target_len:
        start = rng.randint(0, n - target_len)
        data, _ = sf.read(
            str(entry.path), start=start, stop=start + target_len,
            dtype="float32", always_2d=False,
        )
        return torch.from_numpy(data)
    data, _ = sf.read(str(entry.path), dtype="float32", always_2d=False)
    # Loop short noise files to the target length
    reps = (target_len + data.shape[0] - 1) // data.shape[0]
    tiled = torch.from_numpy(data).repeat(reps)[:target_len]
    return tiled


def _rescale_to_snr(
    target: torch.Tensor,
    interferer: torch.Tensor,
    snr_db: float,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Scale ``interferer`` so that target-vs-scaled-interferer hits ``snr_db``."""
    target_rms = float(_rms(target, eps))
    interf_rms = float(_rms(interferer, eps))
    factor = 10.0 ** (snr_db / 20.0)
    if interf_rms < eps:
        return interferer
    k = target_rms / (interf_rms * factor + eps)
    return interferer * k


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class WulfeniteMixer(Dataset):
    """On-the-fly 2-speaker target-speaker-extraction mixer.

    This is the Wulfenite data source used by the training loop. It
    wraps an already-scanned speaker dict (see
    :func:`wulfenite.data.aishell.scan_aishell1` and
    :func:`scan_aishell3`) and an optional noise pool, and produces
    training samples on demand.

    The dataset is "virtual": ``__len__`` is the number of samples
    per virtual epoch, chosen independently of the underlying file
    count. Each ``__getitem__`` call draws fresh random mixtures.
    """

    def __init__(
        self,
        speakers: dict[str, list[AudioEntry]],
        noise_pool: Sequence[NoiseEntry] | None = None,
        config: MixerConfig | None = None,
        samples_per_epoch: int = 20000,
        seed: int | None = None,
    ) -> None:
        cfg = config or MixerConfig()
        self.cfg = cfg
        self.samples_per_epoch = samples_per_epoch
        self.segment_len = int(cfg.segment_seconds * cfg.sample_rate)
        self.enrollment_len = int(cfg.enrollment_seconds * cfg.sample_rate)

        # Drop speakers with < 2 utterances (present branch needs two;
        # absent branch needs one but we unify the requirement).
        self.speakers = {k: v for k, v in speakers.items() if len(v) >= 2}
        self.speaker_ids: list[str] = sorted(self.speakers.keys())
        if len(self.speaker_ids) < 2:
            raise RuntimeError(
                "Need at least 2 speakers with ≥ 2 utterances each."
            )

        self.noise_pool = list(noise_pool) if noise_pool else []
        # If caller disabled DNS noise or provided none, fall back
        # silently to synthetic Gaussian noise when noise is applied.
        self._has_dns = bool(self.noise_pool) and cfg.use_dns_noise

        self._base_seed = seed  # None ⇒ fresh entropy each call

        total_utts = sum(len(v) for v in self.speakers.values())
        print(
            f"[WulfeniteMixer] speakers={len(self.speaker_ids)} "
            f"utterances={total_utts} noise_files={len(self.noise_pool)} "
            f"epoch={samples_per_epoch} target_present_prob={cfg.target_present_prob}"
        )

    def __len__(self) -> int:
        return self.samples_per_epoch

    # ------------------------------------------------------------------
    # Sample construction
    # ------------------------------------------------------------------

    def _rng(self, index: int) -> random.Random:
        if self._base_seed is None:
            return random.Random()
        return random.Random(self._base_seed + index)

    def _make_present_sample(self, rng: random.Random) -> dict:
        cfg = self.cfg

        target_spk, interf_spk = rng.sample(self.speaker_ids, 2)
        target_utts = self.speakers[target_spk]
        interf_utts = self.speakers[interf_spk]

        target_entry, enroll_entry = rng.sample(target_utts, 2)
        interf_entry = rng.choice(interf_utts)

        target = _load_chunk(target_entry, self.segment_len, rng)
        interferer = _load_chunk(interf_entry, self.segment_len, rng)
        enrollment = _load_chunk(enroll_entry, self.enrollment_len, rng)

        # Normalize each source to the target RMS.
        target = target / (_rms(target) + 1e-8) * cfg.rms_target
        interferer = interferer / (_rms(interferer) + 1e-8) * cfg.rms_target
        enrollment = enrollment / (_rms(enrollment) + 1e-8) * cfg.rms_target

        # Reverb.
        if cfg.apply_reverb and rng.random() < cfg.reverb_prob:
            rir_t = synth_room_rir(cfg.reverb, rng)
            rir_i = synth_room_rir(cfg.reverb, rng)
            target = apply_rir(target, rir_t)
            interferer = apply_rir(interferer, rir_i)
            if cfg.reverb_enrollment:
                rir_e = synth_room_rir(cfg.reverb, rng)
                enrollment = apply_rir(enrollment, rir_e)
            # Reverb changes RMS, re-normalize.
            target = target / (_rms(target) + 1e-8) * cfg.rms_target
            interferer = interferer / (_rms(interferer) + 1e-8) * cfg.rms_target
            enrollment = enrollment / (_rms(enrollment) + 1e-8) * cfg.rms_target

        # Mix at a random SNR.
        snr_db = rng.uniform(*cfg.snr_range_db)
        interferer_scaled = _rescale_to_snr(target, interferer, snr_db)
        mixture = target + interferer_scaled

        # Additive noise on the mixture.
        if cfg.apply_noise and rng.random() < cfg.noise_prob:
            noise_snr = rng.uniform(*cfg.noise_snr_range_db)
            if self._has_dns:
                noise_entry = rng.choice(self.noise_pool)
                noise = _load_noise_chunk(noise_entry, self.segment_len, rng)
                mixture = add_noise_at_snr(mixture, noise, noise_snr)
            else:
                mixture = add_gaussian_noise(mixture, noise_snr)

        # Clip guard.
        peak = float(mixture.abs().max())
        if peak > cfg.peak_clip:
            scale = cfg.peak_clip / peak
            mixture = mixture * scale
            target = target * scale

        return {
            "mixture": mixture,
            "target": target,
            "enrollment": enrollment,
            "target_present": torch.tensor(1.0, dtype=torch.float32),
            "snr_db": torch.tensor(snr_db, dtype=torch.float32),
        }

    def _make_absent_sample(self, rng: random.Random) -> dict:
        cfg = self.cfg

        # Two speakers: A = claimed target (NOT in mixture),
        # B = the only speaker in the mixture.
        target_spk, interf_spk = rng.sample(self.speaker_ids, 2)

        # Enrollment of A — the speaker the model should try to
        # extract. A is not in the mixture so the correct output
        # is silence.
        enroll_entry = rng.choice(self.speakers[target_spk])
        # Mixture content: one of B's utterances.
        interf_entry = rng.choice(self.speakers[interf_spk])

        enrollment = _load_chunk(enroll_entry, self.enrollment_len, rng)
        mixture_source = _load_chunk(interf_entry, self.segment_len, rng)

        enrollment = enrollment / (_rms(enrollment) + 1e-8) * cfg.rms_target
        mixture_source = mixture_source / (_rms(mixture_source) + 1e-8) * cfg.rms_target

        # Reverb on the mixture content (and optionally the enrollment).
        if cfg.apply_reverb and rng.random() < cfg.reverb_prob:
            rir_m = synth_room_rir(cfg.reverb, rng)
            mixture_source = apply_rir(mixture_source, rir_m)
            mixture_source = mixture_source / (_rms(mixture_source) + 1e-8) * cfg.rms_target
            if cfg.reverb_enrollment:
                rir_e = synth_room_rir(cfg.reverb, rng)
                enrollment = apply_rir(enrollment, rir_e)
                enrollment = enrollment / (_rms(enrollment) + 1e-8) * cfg.rms_target

        mixture = mixture_source

        if cfg.apply_noise and rng.random() < cfg.noise_prob:
            noise_snr = rng.uniform(*cfg.noise_snr_range_db)
            if self._has_dns:
                noise_entry = rng.choice(self.noise_pool)
                noise = _load_noise_chunk(noise_entry, self.segment_len, rng)
                mixture = add_noise_at_snr(mixture, noise, noise_snr)
            else:
                mixture = add_gaussian_noise(mixture, noise_snr)

        peak = float(mixture.abs().max())
        if peak > cfg.peak_clip:
            mixture = mixture * (cfg.peak_clip / peak)

        return {
            "mixture": mixture,
            "target": torch.zeros(self.segment_len, dtype=torch.float32),
            "enrollment": enrollment,
            "target_present": torch.tensor(0.0, dtype=torch.float32),
            # SNR is not meaningful for absent samples; log as 0.
            "snr_db": torch.tensor(0.0, dtype=torch.float32),
        }

    def __getitem__(self, index: int) -> dict:
        rng = self._rng(index)
        if rng.random() < self.cfg.target_present_prob:
            return self._make_present_sample(rng)
        return self._make_absent_sample(rng)


# ---------------------------------------------------------------------------
# Batching
# ---------------------------------------------------------------------------


def collate_mixer_batch(batch: Sequence[dict]) -> dict:
    """Default DataLoader collate for :class:`WulfeniteMixer` samples.

    Stacks per-field into ``[B, *]`` tensors. Assumes all samples in
    the batch share the same segment / enrollment length, which is
    the case for a single ``WulfeniteMixer`` instance.
    """
    return {
        "mixture": torch.stack([b["mixture"] for b in batch], dim=0),
        "target": torch.stack([b["target"] for b in batch], dim=0),
        "enrollment": torch.stack([b["enrollment"] for b in batch], dim=0),
        "target_present": torch.stack([b["target_present"] for b in batch], dim=0),
        "snr_db": torch.stack([b["snr_db"] for b in batch], dim=0),
    }
