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

Every ``__getitem__`` call yields a dict with:

- ``"mixture"``: ``[T]`` float tensor, 16 kHz mono input to the model
- ``"target"``: ``[T]`` float tensor, loss reference (zeros for absent)
- ``"enrollment"``: ``[T_enr_max]`` float tensor, fed to the speaker
  encoder and optionally cropped batch-wise at collation time
- ``"enrollment_speech_len"``: scalar ``long`` tensor with the number
  of non-padding enrollment samples before any batch-wise crop
- ``"target_present"``: scalar tensor, 1.0 or 0.0
- ``"target_speaker_idx"``: scalar ``long`` tensor, stable speaker id
- ``"snr_db"``: scalar tensor for logging

The default collate function computes ``"enrollment_fbank"`` after the
final enrollment tensor is chosen, so the features always match any
batch-wise crop.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Sequence

import soundfile as sf
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from ..models.dvector import compute_fbank_batch
from .aishell import AudioEntry
from .augmentation import (
    ReverbConfig,
    add_noise_at_snr,
    add_gaussian_noise,
    apply_rir,
    synth_room_rir,
)
from .noise import NoiseEntry


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
    enrollment_seconds_range: tuple[float, float] = (1.5, 6.0)

    # --- Mixing ---
    snr_range_db: tuple[float, float] = (-5.0, 5.0)
    target_present_prob: float = 0.85  # fraction of target-present samples

    # --- Optional acoustic augmentation ---
    apply_reverb: bool = True
    reverb_prob: float = 0.85
    reverb: ReverbConfig = field(default_factory=ReverbConfig)
    reverb_enrollment: bool = True
    rir_pool_size: int = 1000

    # --- Additive noise on the final mixture ---
    apply_noise: bool = True
    noise_prob: float = 0.80
    noise_snr_range_db: tuple[float, float] = (10.0, 25.0)
    enrollment_noise_prob: float = 0.5
    enrollment_noise_snr_range_db: tuple[float, float] = (15.0, 30.0)
    # If True and a noise_pool is provided, sample real noise files;
    # otherwise (or if disabled here) fall back to synthetic Gaussian
    # noise. Accepts any corpus that can be scanned by
    # :func:`wulfenite.data.noise.scan_noise_dir` (MUSAN, DEMAND,
    # DNS4, custom recordings, ...).
    use_noise_pool: bool = True

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
        lo, hi = cfg.enrollment_seconds_range
        if lo <= 0.0 or hi <= 0.0 or lo > hi:
            raise ValueError(
                "enrollment_seconds_range must satisfy 0 < lo <= hi; got "
                f"{cfg.enrollment_seconds_range}"
            )
        self.enrollment_len = int(hi * cfg.sample_rate)

        # Drop speakers with < 2 utterances (present branch needs two;
        # absent branch needs one but we unify the requirement).
        self.speakers = {k: v for k, v in speakers.items() if len(v) >= 2}
        self.speaker_ids: list[str] = sorted(self.speakers.keys())
        self.speaker_to_idx: dict[str, int] = {
            sid: i for i, sid in enumerate(self.speaker_ids)
        }
        if len(self.speaker_ids) < 2:
            raise RuntimeError(
                "Need at least 2 speakers with ≥ 2 utterances each."
            )

        self.noise_pool = list(noise_pool) if noise_pool else []
        # If caller disabled the noise pool or provided none, fall back
        # silently to synthetic Gaussian noise when noise is applied.
        self._has_noise_pool = bool(self.noise_pool) and cfg.use_noise_pool

        self._base_seed = seed  # None ⇒ fresh entropy each call
        pool_seed = None if seed is None else seed + 1_000_003
        pool_rng = random.Random(pool_seed)
        pool_size = max(0, cfg.rir_pool_size)
        build_pool = cfg.apply_reverb and cfg.reverb_prob > 0.0 and pool_size > 0
        self._rir_pool = [
            synth_room_rir(cfg.reverb, pool_rng)
            for _ in range(pool_size)
        ] if build_pool else []

        total_utts = sum(len(v) for v in self.speakers.values())
        print(
            f"[WulfeniteMixer] speakers={len(self.speaker_ids)} "
            f"utterances={total_utts} noise_files={len(self.noise_pool)} "
            f"rir_pool={len(self._rir_pool)} "
            f"epoch={samples_per_epoch} target_present_prob={cfg.target_present_prob} "
            f"enrollment_seconds_range={cfg.enrollment_seconds_range}"
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

    def _sample_rir(self, rng: random.Random) -> torch.Tensor:
        if self._rir_pool:
            return rng.choice(self._rir_pool)
        return synth_room_rir(self.cfg.reverb, rng)

    def _maybe_add_enrollment_noise(
        self,
        enrollment: torch.Tensor,
        rng: random.Random,
    ) -> torch.Tensor:
        cfg = self.cfg
        if (
            cfg.enrollment_noise_prob <= 0.0
            or rng.random() >= cfg.enrollment_noise_prob
        ):
            return enrollment

        noise_snr = rng.uniform(*cfg.enrollment_noise_snr_range_db)
        if self._has_noise_pool:
            noise_entry = rng.choice(self.noise_pool)
            noise = _load_noise_chunk(noise_entry, enrollment.shape[-1], rng)
            return add_noise_at_snr(enrollment, noise, noise_snr)
        return add_gaussian_noise(enrollment, noise_snr)

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
        enrollment_speech_len = min(enroll_entry.num_frames, self.enrollment_len)

        # Normalize each source to the target RMS.
        target = target / (_rms(target) + 1e-8) * cfg.rms_target
        interferer = interferer / (_rms(interferer) + 1e-8) * cfg.rms_target
        enrollment = enrollment / (_rms(enrollment) + 1e-8) * cfg.rms_target

        # Reverb.
        if cfg.apply_reverb and rng.random() < cfg.reverb_prob:
            rir_t = self._sample_rir(rng)
            rir_i = self._sample_rir(rng)
            target = apply_rir(target, rir_t)
            interferer = apply_rir(interferer, rir_i)
            if cfg.reverb_enrollment:
                rir_e = self._sample_rir(rng)
                enrollment = apply_rir(enrollment, rir_e)
            # Reverb changes RMS, re-normalize.
            target = target / (_rms(target) + 1e-8) * cfg.rms_target
            interferer = interferer / (_rms(interferer) + 1e-8) * cfg.rms_target
            enrollment = enrollment / (_rms(enrollment) + 1e-8) * cfg.rms_target

        enrollment = self._maybe_add_enrollment_noise(enrollment, rng)

        # Mix at a random SNR.
        snr_db = rng.uniform(*cfg.snr_range_db)
        interferer_scaled = _rescale_to_snr(target, interferer, snr_db)
        mixture = target + interferer_scaled

        # Additive noise on the mixture.
        if cfg.apply_noise and rng.random() < cfg.noise_prob:
            noise_snr = rng.uniform(*cfg.noise_snr_range_db)
            if self._has_noise_pool:
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
            "enrollment_speech_len": torch.tensor(
                enrollment_speech_len, dtype=torch.long
            ),
            "target_present": torch.tensor(1.0, dtype=torch.float32),
            "target_speaker_idx": torch.tensor(
                self.speaker_to_idx[target_spk], dtype=torch.long,
            ),
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
        enrollment_speech_len = min(enroll_entry.num_frames, self.enrollment_len)

        enrollment = enrollment / (_rms(enrollment) + 1e-8) * cfg.rms_target
        mixture_source = mixture_source / (_rms(mixture_source) + 1e-8) * cfg.rms_target

        # Reverb on the mixture content (and optionally the enrollment).
        if cfg.apply_reverb and rng.random() < cfg.reverb_prob:
            rir_m = self._sample_rir(rng)
            mixture_source = apply_rir(mixture_source, rir_m)
            mixture_source = mixture_source / (_rms(mixture_source) + 1e-8) * cfg.rms_target
            if cfg.reverb_enrollment:
                rir_e = self._sample_rir(rng)
                enrollment = apply_rir(enrollment, rir_e)
                enrollment = enrollment / (_rms(enrollment) + 1e-8) * cfg.rms_target

        enrollment = self._maybe_add_enrollment_noise(enrollment, rng)

        mixture = mixture_source

        if cfg.apply_noise and rng.random() < cfg.noise_prob:
            noise_snr = rng.uniform(*cfg.noise_snr_range_db)
            if self._has_noise_pool:
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
            "enrollment_speech_len": torch.tensor(
                enrollment_speech_len, dtype=torch.long
            ),
            "target_present": torch.tensor(0.0, dtype=torch.float32),
            "target_speaker_idx": torch.tensor(
                self.speaker_to_idx[target_spk], dtype=torch.long,
            ),
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


def collate_mixer_batch(
    batch: Sequence[dict],
    enrollment_seconds_range: tuple[float, float] | None = None,
    sample_rate: int = SAMPLE_RATE,
) -> dict:
    """Default DataLoader collate for :class:`WulfeniteMixer` samples.

    Stacks per-field into ``[B, *]`` tensors. Assumes all samples in
    the batch share the same segment / enrollment length, which is
    the case for a single ``WulfeniteMixer`` instance.
    """
    if enrollment_seconds_range is None:
        enrollment = torch.stack([b["enrollment"] for b in batch], dim=0)
    else:
        lo, hi = enrollment_seconds_range
        if lo <= 0.0 or hi <= 0.0 or lo > hi:
            raise ValueError(
                "enrollment_seconds_range must satisfy 0 < lo <= hi; got "
                f"{enrollment_seconds_range}"
            )
        min_len = int(lo * sample_rate)
        max_len = int(hi * sample_rate)
        target_len = random.randint(min_len, max_len)
        enrollment_items: list[torch.Tensor] = []
        for sample in batch:
            wav = sample["enrollment"]
            speech_len = int(sample["enrollment_speech_len"].item())
            max_start = max(0, speech_len - target_len)
            start = random.randint(0, max_start) if max_start > 0 else 0
            if wav.numel() > target_len:
                wav = wav[start:start + target_len]
            if wav.numel() < target_len:
                wav = F.pad(wav, (0, target_len - wav.numel()))
            enrollment_items.append(wav)
        enrollment = torch.stack(enrollment_items, dim=0)
    enrollment_fbank = compute_fbank_batch(enrollment)

    return {
        "mixture": torch.stack([b["mixture"] for b in batch], dim=0),
        "target": torch.stack([b["target"] for b in batch], dim=0),
        "enrollment": enrollment,
        "enrollment_fbank": enrollment_fbank,
        "enrollment_speech_len": torch.stack(
            [b["enrollment_speech_len"] for b in batch], dim=0
        ),
        "target_present": torch.stack([b["target_present"] for b in batch], dim=0),
        "target_speaker_idx": torch.stack(
            [b["target_speaker_idx"] for b in batch], dim=0,
        ),
        "snr_db": torch.stack([b["snr_db"] for b in batch], dim=0),
    }
