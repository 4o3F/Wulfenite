"""On-the-fly TSE mixer.

Legacy mode still emits single-view present/absent samples. The
default ``clip_composer`` mode now generates one long coherent scene
and derives several enrollment-conditioned views from the same
mixture:

- target speaker ``A``
- alternate in-scene speaker ``B``
- outsider absent view

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
- ``"enrollment"``: ``[T_enr]`` float tensor, fed to the speaker encoder
- ``"target_present"``: scalar tensor, 1.0 or 0.0
- ``"target_speaker_idx"``: scalar ``long`` tensor, stable speaker id
- ``"snr_db"``: scalar tensor for logging

The default collate function computes ``"enrollment_fbank"`` after the
batch enrollment tensor is stacked.
"""

from __future__ import annotations

import random
import warnings
from dataclasses import dataclass, field, replace
from typing import Sequence

import soundfile as sf
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from ..audio_features import compute_fbank_batch
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
VIEW_ROLE_TO_ID = {
    "A": 0,
    "B": 1,
    "OUTSIDER": 2,
    "A_ALT": 3,
}


def _default_composer_config() -> "ComposerConfig":
    from .composer import ComposerConfig

    return ComposerConfig()


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
    segment_seconds: float = 8.0
    enrollment_seconds: float = 4.0
    # Deprecated compatibility shim for older inference utilities and
    # checkpoints that still pass or read a fixed "range".
    enrollment_seconds_range: tuple[float, float] | None = None

    # --- Mixing ---
    snr_range_db: tuple[float, float] = (-5.0, 5.0)
    composition_mode: str = "clip_composer"
    composer: "ComposerConfig" = field(default_factory=_default_composer_config)
    target_present_prob: float = 0.85  # fraction of target-present samples
    transition_prob: float = 0.0
    transition_min_fraction: float = 0.25
    transition_min_target_rms: float = 0.01

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

    def __post_init__(self) -> None:
        if self.enrollment_seconds_range is not None:
            lo, hi = self.enrollment_seconds_range
            if lo <= 0.0 or hi <= 0.0 or lo > hi:
                raise ValueError(
                    "enrollment_seconds_range must satisfy 0 < lo <= hi; got "
                    f"{self.enrollment_seconds_range}"
                )
            if abs(float(lo) - float(hi)) > 1e-9:
                raise ValueError(
                    "enrollment_seconds_range compatibility only supports fixed "
                    f"endpoints; got {self.enrollment_seconds_range}"
                )
            self.enrollment_seconds = float(hi)
        if self.enrollment_seconds <= 0.0:
            raise ValueError(
                "enrollment_seconds must be positive; got "
                f"{self.enrollment_seconds}"
            )
        if not 0.0 <= self.transition_prob <= 1.0:
            raise ValueError(
                "transition_prob must be in [0, 1]; got "
                f"{self.transition_prob}"
            )
        if not 0.0 < self.transition_min_fraction < 0.5:
            raise ValueError(
                "transition_min_fraction must be in (0, 0.5); got "
                f"{self.transition_min_fraction}"
            )
        if self.transition_min_target_rms <= 0.0:
            raise ValueError(
                "transition_min_target_rms must be positive; got "
                f"{self.transition_min_target_rms}"
            )
        if self.composition_mode not in ("clip_composer", "legacy_branch"):
            raise ValueError(
                "composition_mode must be 'clip_composer' or 'legacy_branch'; got "
                f"{self.composition_mode!r}"
            )
        self.enrollment_seconds_range = (
            float(self.enrollment_seconds),
            float(self.enrollment_seconds),
        )
        if self.composition_mode == "clip_composer":
            self.composer = replace(
                self.composer,
                sample_rate=self.sample_rate,
                segment_seconds=self.segment_seconds,
            )


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
    k = _snr_scale_factor(target, interferer, snr_db, eps=eps)
    return interferer * k


def _snr_scale_factor(
    target: torch.Tensor,
    interferer: torch.Tensor,
    snr_db: float,
    eps: float = 1e-8,
) -> float:
    """Return the scalar applied to ``interferer`` to hit ``snr_db``."""
    target_rms = float(_rms(target, eps))
    interf_rms = float(_rms(interferer, eps))
    factor = 10.0 ** (snr_db / 20.0)
    if interf_rms < eps:
        return 1.0
    return target_rms / (interf_rms * factor + eps)


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
        if cfg.enrollment_seconds <= 0.0:
            raise ValueError(
                "enrollment_seconds must be positive; got "
                f"{cfg.enrollment_seconds}"
            )
        self.enrollment_len = int(cfg.enrollment_seconds * cfg.sample_rate)

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
        self._speaker_mean_frames = {
            sid: sum(entry.num_frames for entry in entries) / len(entries)
            for sid, entries in self.speakers.items()
        }
        self._speaker_primary_dataset = {
            sid: max(
                {entry.dataset for entry in entries},
                key=lambda dataset: sum(
                    1 for entry in entries if entry.dataset == dataset
                ),
            )
            for sid, entries in self.speakers.items()
        }
        self._planner = None
        self._renderer = None
        if cfg.composition_mode == "clip_composer":
            from .composer import ClipPlanner, ClipRenderer

            if cfg.transition_prob > 0.0:
                warnings.warn(
                    "transition_prob is deprecated in clip_composer mode and will "
                    "be ignored",
                    DeprecationWarning,
                    stacklevel=2,
                )
            if abs(cfg.target_present_prob - 0.85) > 1e-6:
                warnings.warn(
                    "target_present_prob is ignored in clip_composer mode because "
                    "each unified long scene now emits both present and absent "
                    "views by construction",
                    DeprecationWarning,
                    stacklevel=2,
                )
            self._planner = ClipPlanner(
                cfg.composer,
                allow_third_speaker=len(self.speaker_ids) >= 4,
            )
            self._renderer = ClipRenderer(
                cfg,
                noise_pool=self.noise_pool,
                rir_pool=self._rir_pool,
            )

        total_utts = sum(len(v) for v in self.speakers.values())
        print(
            f"[WulfeniteMixer] speakers={len(self.speaker_ids)} "
            f"utterances={total_utts} noise_files={len(self.noise_pool)} "
            f"rir_pool={len(self._rir_pool)} "
            f"epoch={samples_per_epoch} composition_mode={cfg.composition_mode} "
            f"target_present_prob={cfg.target_present_prob}"
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

    def _sample_interferer_speaker(
        self,
        target_spk: str,
        used: set[str],
        rng: random.Random,
    ) -> str:
        candidates = [sid for sid in self.speaker_ids if sid not in used]
        if not candidates:
            raise RuntimeError("Need at least one unused interferer speaker.")
        target_dataset = self._speaker_primary_dataset[target_spk]
        same_dataset = [
            sid for sid in candidates
            if self._speaker_primary_dataset[sid] == target_dataset
        ]
        pool = same_dataset or candidates
        target_mean = self._speaker_mean_frames[target_spk]
        pool = sorted(
            pool,
            key=lambda sid: abs(self._speaker_mean_frames[sid] - target_mean),
        )
        head = pool[:max(1, min(len(pool), 8))]
        return rng.choice(head)

    def _sample_speaker_cast(
        self,
        plan: "ClipPlan",
        rng: random.Random,
    ) -> "SpeakerCast":
        from .composer import SpeakerCast

        required_samples_by_role = self._required_samples_by_role(plan)
        target_spk = rng.choice(self.speaker_ids)
        target_entry, enroll_entry = self._sample_role_entries(
            target_spk,
            required_source_frames=required_samples_by_role["A"],
            rng=rng,
        )
        used = {target_spk}
        source_speaker_ids = {"A": target_spk}
        source_entries: dict[str, AudioEntry] = {"A": target_entry}
        enrollment_entries: dict[str, AudioEntry] = {"A": enroll_entry}
        roles = ("B", "C") if plan.use_third_speaker else ("B",)
        for role in roles:
            spk_id = self._sample_interferer_speaker(target_spk, used, rng)
            used.add(spk_id)
            source_entry, enrollment_entry = self._sample_role_entries(
                spk_id,
                required_source_frames=required_samples_by_role[role],
                rng=rng,
            )
            source_speaker_ids[role] = spk_id
            source_entries[role] = source_entry
            enrollment_entries[role] = enrollment_entry
        outsider_spk = self._sample_interferer_speaker(target_spk, used, rng)
        return SpeakerCast(
            source_speaker_ids=source_speaker_ids,
            source_entries=source_entries,
            enrollment_entries=enrollment_entries,
            outsider_speaker_id=outsider_spk,
            outsider_enrollment_entry=rng.choice(self.speakers[outsider_spk]),
        )

    @staticmethod
    def _required_samples_by_role(plan: "ClipPlan") -> dict[str, int]:
        required = {
            role: 0 for role in plan.active_frames_by_role
        }
        for slot in plan.slots:
            length = (slot.end_frame - slot.start_frame) * plan.stride_samples
            for role in slot.active_roles:
                required[role] = required.get(role, 0) + length
        return required

    def _sample_role_entries(
        self,
        speaker_id: str,
        *,
        required_source_frames: int,
        rng: random.Random,
    ) -> tuple[AudioEntry, AudioEntry]:
        entries = self.speakers[speaker_id]
        source_candidates = [
            entry for entry in entries
            if entry.num_frames >= required_source_frames
        ]
        if not source_candidates:
            ranked = sorted(entries, key=lambda entry: entry.num_frames, reverse=True)
            source_candidates = ranked[:max(1, min(len(ranked), 4))]
        source_entry = rng.choice(source_candidates)
        enroll_candidates = [
            entry for entry in entries if entry.path != source_entry.path
        ]
        if not enroll_candidates:
            raise RuntimeError(
                f"speaker {speaker_id} does not have a distinct enrollment utterance"
            )
        enrollment_entry = rng.choice(enroll_candidates)
        return source_entry, enrollment_entry

    def _select_enrollment(
        self,
        pool: Sequence[torch.Tensor],
        rng: random.Random,
        *,
        avoid_index: int | None = None,
    ) -> torch.Tensor:
        if not pool:
            raise ValueError("enrollment pool must not be empty")
        weights = [0.35, 0.25, 0.15, 0.15, 0.10]
        indices = list(range(len(pool)))
        if avoid_index is not None and avoid_index in indices and len(indices) > 1:
            indices.remove(avoid_index)
        selected = rng.choices(
            indices,
            weights=[weights[i] if i < len(weights) else 1.0 for i in indices],
            k=1,
        )[0]
        return pool[selected].clone()

    @staticmethod
    def _union_active_frames(
        active_frames_by_role: dict[str, torch.Tensor],
        *,
        exclude_role: str | None = None,
    ) -> torch.Tensor:
        union = torch.zeros_like(next(iter(active_frames_by_role.values())))
        for role, active in active_frames_by_role.items():
            if role != exclude_role:
                union = union | active.bool()
        return union

    @staticmethod
    def _frame_mask_to_sample_mask(
        frame_mask: torch.Tensor,
        *,
        stride_samples: int,
        n_samples: int,
    ) -> torch.Tensor:
        return frame_mask.bool().repeat_interleave(stride_samples)[:n_samples]

    @staticmethod
    def _masked_rms(
        signal: torch.Tensor,
        sample_mask: torch.Tensor,
    ) -> float:
        if not bool(sample_mask.any().item()):
            return 0.0
        segment = signal[sample_mask]
        return float(torch.sqrt(torch.mean(segment * segment) + 1e-12))

    def _bundle_is_usable(
        self,
        plan: "ClipPlan",
        bundle: "SceneBundle",
    ) -> bool:
        min_activity_ratio = 0.15
        min_active_rms = self.cfg.rms_target * 0.02

        for role, planned_mask in plan.active_frames_by_role.items():
            actual_mask = bundle.active_frames_by_role.get(role)
            track = bundle.source_tracks.get(role)
            if actual_mask is None or track is None:
                return False
            planned_frames = int(planned_mask.sum().item())
            actual_frames = int(actual_mask.sum().item())
            min_frames = max(1, int(round(planned_frames * min_activity_ratio)))
            if actual_frames < min_frames:
                return False
            active_sample_mask = self._frame_mask_to_sample_mask(
                actual_mask,
                stride_samples=plan.stride_samples,
                n_samples=track.shape[-1],
            )
            if self._masked_rms(track, active_sample_mask) < min_active_rms:
                return False

        planned_overlap = int(plan.overlap_frames.sum().item())
        actual_overlap = int(bundle.overlap_frames.sum().item())
        min_overlap_frames = max(1, int(round(planned_overlap * min_activity_ratio)))
        if actual_overlap < min_overlap_frames:
            return False
        return bool(bundle.background_frames.any().item())

    def _scene_views_from_bundle(
        self,
        bundle: "SceneBundle",
        rng: random.Random,
    ) -> list[dict]:
        from .composer import SceneView

        mixture = bundle.mixture
        zeros_wave = torch.zeros_like(mixture)
        zeros_frames = torch.zeros_like(bundle.background_frames)
        speaker_indices = bundle.metadata["speaker_indices"]
        target_active_a = bundle.active_frames_by_role["A"]
        target_active_b = bundle.active_frames_by_role["B"]
        target_present_a = float(bool(target_active_a.any().item()))
        target_present_b = float(bool(target_active_b.any().item()))
        views = [
            SceneView(
                scene_id=bundle.scene_id,
                view_role="A",
                enrollment=self._select_enrollment(bundle.enrollment_pool["A"], rng),
                target=(
                    bundle.source_tracks["A"]
                    if target_present_a >= 0.5
                    else zeros_wave
                ),
                target_present=target_present_a,
                target_speaker_idx=speaker_indices["A"],
                target_active_frames=target_active_a,
                nontarget_active_frames=self._union_active_frames(
                    bundle.active_frames_by_role,
                    exclude_role="A",
                ),
                overlap_frames=(
                    target_active_a
                    & self._union_active_frames(
                        bundle.active_frames_by_role,
                        exclude_role="A",
                    )
                ),
                background_frames=bundle.background_frames,
                snr_db=float(bundle.metadata.get("snr_db", 0.0)),
            ),
            SceneView(
                scene_id=bundle.scene_id,
                view_role="B",
                enrollment=self._select_enrollment(bundle.enrollment_pool["B"], rng),
                target=(
                    bundle.source_tracks["B"]
                    if target_present_b >= 0.5
                    else zeros_wave
                ),
                target_present=target_present_b,
                target_speaker_idx=speaker_indices["B"],
                target_active_frames=target_active_b,
                nontarget_active_frames=self._union_active_frames(
                    bundle.active_frames_by_role,
                    exclude_role="B",
                ),
                overlap_frames=(
                    target_active_b
                    & self._union_active_frames(
                        bundle.active_frames_by_role,
                        exclude_role="B",
                    )
                ),
                background_frames=bundle.background_frames,
                snr_db=float(bundle.metadata.get("snr_db", 0.0)),
            ),
            SceneView(
                scene_id=bundle.scene_id,
                view_role="OUTSIDER",
                enrollment=self._select_enrollment(
                    bundle.enrollment_pool["OUTSIDER"], rng,
                ),
                target=zeros_wave,
                target_present=0.0,
                target_speaker_idx=bundle.metadata["outsider_speaker_idx"],
                target_active_frames=zeros_frames,
                nontarget_active_frames=self._union_active_frames(
                    bundle.active_frames_by_role,
                ),
                overlap_frames=zeros_frames,
                background_frames=bundle.background_frames,
                snr_db=0.0,
            ),
        ]

        samples: list[dict] = []
        for view in views:
            sample = view.to_sample()
            sample["mixture"] = mixture.clone()
            sample["view_role_id"] = torch.tensor(
                VIEW_ROLE_TO_ID[view.view_role], dtype=torch.long,
            )
            sample["view_role"] = view.view_role
            samples.append(sample)
        return samples

    def sample_scene(self, index: int) -> dict:
        """Sample one full long-scene bundle plus its derived views.

        Only valid in ``clip_composer`` mode. This is a utility for
        diagnostics and audio export; training still consumes
        ``__getitem__`` / ``collate_mixer_batch``.
        """
        if self.cfg.composition_mode != "clip_composer":
            raise RuntimeError("sample_scene is only available in clip_composer mode")
        if self._planner is None or self._renderer is None:
            raise RuntimeError("clip_composer mode requires planner + renderer")
        rng = self._rng(index)
        for _ in range(8):
            plan = self._planner.plan(rng=rng)
            cast = self._sample_speaker_cast(plan, rng)
            bundle = self._renderer.render_bundle(
                scene_id=index,
                plan=plan,
                cast=cast,
                speaker_to_idx=self.speaker_to_idx,
                rng=rng,
            )
            bundle.metadata["snr_db"] = plan.snr_db
            scene = {
                "scene_id": index,
                "plan": plan,
                "bundle": bundle,
                "views": self._scene_views_from_bundle(bundle, rng),
            }
            if self._bundle_is_usable(plan, bundle):
                return scene
        raise RuntimeError(
            f"failed to render a usable clip_composer scene for index={index} "
            f"after 8 attempts"
        )

    def _attach_legacy_frame_labels(self, sample: dict) -> dict:
        stride = self.cfg.composer.stride_samples
        target = sample["target"]
        n_frames = target.shape[-1] // stride
        usable = n_frames * stride
        target_active = (
            target[:usable]
            .reshape(n_frames, stride)
            .abs()
            .amax(dim=-1)
            .gt(1e-7)
        )
        if bool(sample["target_present"].item() >= 0.5):
            nontarget_active = torch.ones(n_frames, dtype=torch.bool)
        else:
            nontarget_active = torch.zeros(n_frames, dtype=torch.bool)
        sample["target_active_frames"] = target_active
        sample["nontarget_active_frames"] = nontarget_active
        sample["overlap_frames"] = target_active & nontarget_active
        return sample

    def _prepare_present_sources(
        self,
        rng: random.Random,
    ) -> tuple[str, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Load present-branch sources with the standard augmentation pipeline."""
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

        return target_spk, target, interferer, enrollment

    def _maybe_add_final_noise(
        self,
        mixture: torch.Tensor,
        rng: random.Random,
    ) -> torch.Tensor:
        cfg = self.cfg
        if cfg.apply_noise and rng.random() < cfg.noise_prob:
            noise_snr = rng.uniform(*cfg.noise_snr_range_db)
            if self._has_noise_pool:
                noise_entry = rng.choice(self.noise_pool)
                noise = _load_noise_chunk(noise_entry, self.segment_len, rng)
                mixture = add_noise_at_snr(mixture, noise, noise_snr)
            else:
                mixture = add_gaussian_noise(mixture, noise_snr)
        return mixture

    def _clip_present_like_pair(
        self,
        mixture: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        peak = float(mixture.abs().max())
        if peak > self.cfg.peak_clip:
            scale = self.cfg.peak_clip / peak
            mixture = mixture * scale
            target = target * scale
        return mixture, target

    def _make_present_sample(self, rng: random.Random) -> dict:
        cfg = self.cfg
        target_spk, target, interferer, enrollment = self._prepare_present_sources(rng)

        # Mix at a random SNR.
        snr_db = rng.uniform(*cfg.snr_range_db)
        interferer_scaled = _rescale_to_snr(target, interferer, snr_db)
        mixture = target + interferer_scaled

        # Additive noise on the mixture.
        mixture = self._maybe_add_final_noise(mixture, rng)

        # Clip guard.
        mixture, target = self._clip_present_like_pair(mixture, target)

        return {
            "mixture": mixture,
            "target": target,
            "enrollment": enrollment,
            "target_present": torch.tensor(1.0, dtype=torch.float32),
            "target_speaker_idx": torch.tensor(
                self.speaker_to_idx[target_spk], dtype=torch.long,
            ),
            "snr_db": torch.tensor(snr_db, dtype=torch.float32),
        }

    def _make_transition_sample(self, rng: random.Random) -> dict:
        cfg = self.cfg
        target_spk, target, interferer, enrollment = self._prepare_present_sources(rng)

        align = 160
        raw_min = int(cfg.transition_min_fraction * self.segment_len)
        # Align bounds to encoder stride so t0 always lands on a frame edge.
        min_len = ((raw_min + align - 1) // align) * align
        max_len = ((self.segment_len - raw_min) // align) * align
        if min_len > max_len:
            min_len = max_len
        t0 = rng.randint(min_len // align, max_len // align) * align

        roll = rng.random()
        if roll < 0.40:
            transition_type = "absent_to_present"
        elif roll < 0.70:
            transition_type = "silence_to_present"
        elif roll < 0.90:
            transition_type = "present_to_absent"
        else:
            transition_type = "present_to_silence"

        target_gate = torch.ones(self.segment_len, dtype=target.dtype)
        interferer_gate = torch.ones(self.segment_len, dtype=interferer.dtype)
        if transition_type in ("absent_to_present", "silence_to_present"):
            target_gate[:t0] = 0.0
            present_slice = slice(t0, self.segment_len)
        else:
            target_gate[t0:] = 0.0
            present_slice = slice(0, t0)
        if transition_type in ("silence_to_present", "present_to_silence"):
            if transition_type == "silence_to_present":
                interferer_gate[:t0] = 0.0
            else:
                interferer_gate[t0:] = 0.0

        gated_target = target * target_gate
        # Use epsilon-free RMS so the guard catches truly zero targets
        # even at very small thresholds (unlike _rms() which floors at 1e-4).
        gated_rms = float(torch.sqrt(torch.mean(gated_target * gated_target)))
        if gated_rms < cfg.transition_min_target_rms:
            return self._make_present_sample(rng)
        snr_db = rng.uniform(*cfg.snr_range_db)
        k = _snr_scale_factor(
            target[present_slice],
            interferer[present_slice],
            snr_db,
        )
        interferer_scaled = interferer * k * interferer_gate
        mixture = gated_target + interferer_scaled

        mixture = self._maybe_add_final_noise(mixture, rng)
        mixture, gated_target = self._clip_present_like_pair(mixture, gated_target)

        return {
            "mixture": mixture,
            "target": gated_target,
            "enrollment": enrollment,
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
            "target_present": torch.tensor(0.0, dtype=torch.float32),
            "target_speaker_idx": torch.tensor(
                self.speaker_to_idx[target_spk], dtype=torch.long,
            ),
            # SNR is not meaningful for absent samples; log as 0.
            "snr_db": torch.tensor(0.0, dtype=torch.float32),
        }

    def __getitem__(self, index: int) -> dict:
        if self.cfg.composition_mode == "clip_composer":
            scene = self.sample_scene(index)
            return {
                "scene_id": torch.tensor(index, dtype=torch.long),
                "views": scene["views"],
            }
        rng = self._rng(index)
        if rng.random() < self.cfg.transition_prob:
            return self._attach_legacy_frame_labels(self._make_transition_sample(rng))
        if rng.random() < self.cfg.target_present_prob:
            return self._attach_legacy_frame_labels(self._make_present_sample(rng))
        return self._attach_legacy_frame_labels(self._make_absent_sample(rng))


# ---------------------------------------------------------------------------
# Batching
# ---------------------------------------------------------------------------


def collate_mixer_batch(batch: Sequence[dict]) -> dict:
    """Default DataLoader collate for :class:`WulfeniteMixer` samples.

    Stacks per-field into ``[B, *]`` tensors. Assumes all samples in
    the batch share the same segment / enrollment length, which is
    the case for a single ``WulfeniteMixer`` instance.
    """
    if batch and "views" in batch[0]:
        flattened: list[dict] = []
        for scene in batch:
            flattened.extend(scene["views"])
        batch = flattened

    enrollment = torch.stack([b["enrollment"] for b in batch], dim=0)
    enrollment_fbank = compute_fbank_batch(enrollment)

    def _stack_frame_labels(key: str) -> torch.Tensor:
        if key not in batch[0]:
            return torch.zeros((len(batch), 0), dtype=torch.bool)
        labels = [b[key] for b in batch]
        if any(torch.is_floating_point(v) for v in labels):
            return torch.stack([v.to(torch.float32) for v in labels], dim=0)
        return torch.stack([v.to(torch.bool) for v in labels], dim=0)

    collated = {
        "mixture": torch.stack([b["mixture"] for b in batch], dim=0),
        "target": torch.stack([b["target"] for b in batch], dim=0),
        "enrollment": enrollment,
        "enrollment_fbank": enrollment_fbank,
        "target_present": torch.stack([b["target_present"] for b in batch], dim=0),
        "target_speaker_idx": torch.stack(
            [b["target_speaker_idx"] for b in batch], dim=0,
        ),
        "snr_db": torch.stack([b["snr_db"] for b in batch], dim=0),
        "target_active_frames": _stack_frame_labels("target_active_frames"),
        "nontarget_active_frames": _stack_frame_labels("nontarget_active_frames"),
        "overlap_frames": _stack_frame_labels("overlap_frames"),
        "background_frames": _stack_frame_labels("background_frames"),
    }
    if "scene_id" in batch[0]:
        collated["scene_id"] = torch.stack([b["scene_id"] for b in batch], dim=0)
    if "view_role_id" in batch[0]:
        collated["view_role_id"] = torch.stack(
            [b["view_role_id"] for b in batch], dim=0,
        )
        collated["view_role"] = [str(b["view_role"]) for b in batch]
    return collated
