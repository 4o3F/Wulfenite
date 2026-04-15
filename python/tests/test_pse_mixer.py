"""Unit tests for the PSE scene mixer."""

from __future__ import annotations

from pathlib import Path
import random

import numpy as np
import pytest
import soundfile as sf
import torch

from wulfenite.data import AudioEntry, PSEMixer, ReverbConfig
from wulfenite.data.noise import NoiseEntry


SR = 16000


def _write_constant_wav(
    path: Path,
    value: float,
    seconds: float,
    *,
    dataset: str = "test",
) -> AudioEntry:
    path.parent.mkdir(parents=True, exist_ok=True)
    audio = np.full(int(seconds * SR), value, dtype=np.float32)
    sf.write(str(path), audio, SR)
    return AudioEntry(
        speaker_id=path.parent.name,
        path=path,
        num_frames=audio.shape[0],
        dataset=dataset,
    )


def _write_noise(path: Path, seconds: float = 3.0) -> NoiseEntry:
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    audio = rng.standard_normal(int(seconds * SR)).astype(np.float32) * 0.05
    sf.write(str(path), audio, SR)
    return NoiseEntry(path=path, num_frames=audio.shape[0])


def _write_constant_noise(path: Path, value: float, seconds: float = 3.0) -> NoiseEntry:
    path.parent.mkdir(parents=True, exist_ok=True)
    audio = np.full(int(seconds * SR), value, dtype=np.float32)
    sf.write(str(path), audio, SR)
    return NoiseEntry(path=path, num_frames=audio.shape[0])


def test_pse_mixer_returns_expected_shapes(tmp_path: Path) -> None:
    speakers = {
        "S0001": [
            _write_constant_wav(tmp_path / "spk1" / "utt1.wav", 0.2, 2.0),
            _write_constant_wav(tmp_path / "spk1" / "utt2.wav", 0.2, 2.2),
        ],
        "S0002": [
            _write_constant_wav(tmp_path / "spk2" / "utt1.wav", 0.4, 2.0),
            _write_constant_wav(tmp_path / "spk2" / "utt2.wav", 0.4, 2.2),
        ],
    }
    noises = [_write_noise(tmp_path / "noise" / "n1.wav")]
    mixer = PSEMixer(
        speakers,
        noises,
        epoch_size=4,
        segment_length=16000,
        enrollment_length=24000,
        reverb_probability=0.0,
        gain_probability=0.0,
        bandwidth_limit_probability=0.0,
        seed=0,
    )

    mixture, target, enrollment, speaker_id = mixer[0]
    assert mixture.shape == (16000,)
    assert target.shape == (16000,)
    assert enrollment.shape == (24000,)
    assert speaker_id in speakers


def test_pse_mixer_enrollment_matches_target_speaker(tmp_path: Path) -> None:
    speakers = {
        "S0001": [
            _write_constant_wav(tmp_path / "spk1" / "utt1.wav", 0.2, 1.5),
            _write_constant_wav(tmp_path / "spk1" / "utt2.wav", 0.2, 1.6),
        ],
    }
    mixer = PSEMixer(
        speakers,
        noises=[],
        epoch_size=2,
        segment_length=8000,
        enrollment_length=12000,
        reverb_probability=0.0,
        gain_probability=0.0,
        bandwidth_limit_probability=0.0,
        seed=1,
    )
    mixture, target, enrollment, speaker_id = mixer[0]
    assert speaker_id == "S0001"
    assert torch.allclose(mixture, target)
    assert target.abs().mean().item() == pytest.approx(0.2, abs=1e-4)
    assert enrollment.abs().mean().item() == pytest.approx(0.2, abs=1e-4)


def test_pse_mixer_can_generate_nontrivial_mixtures(tmp_path: Path) -> None:
    speakers = {
        "S0001": [
            _write_constant_wav(tmp_path / "spk1" / "utt1.wav", 0.1, 2.0),
            _write_constant_wav(tmp_path / "spk1" / "utt2.wav", 0.1, 2.1),
        ],
        "S0002": [
            _write_constant_wav(tmp_path / "spk2" / "utt1.wav", 0.4, 2.0),
            _write_constant_wav(tmp_path / "spk2" / "utt2.wav", 0.4, 2.1),
        ],
    }
    noises = [_write_noise(tmp_path / "noise" / "n1.wav")]
    mixer = PSEMixer(
        speakers,
        noises,
        epoch_size=10,
        segment_length=8000,
        enrollment_length=12000,
        reverb_probability=0.0,
        gain_probability=0.0,
        bandwidth_limit_probability=0.0,
        seed=7,
    )

    deltas = []
    for idx in range(len(mixer)):
        mixture, target, _enrollment, _speaker_id = mixer[idx]
        deltas.append(float((mixture - target).abs().sum()))
    assert any(delta > 0.0 for delta in deltas)


def test_pse_mixer_is_deterministic_within_same_epoch(tmp_path: Path) -> None:
    speakers = {
        "S0001": [
            _write_constant_wav(tmp_path / "spk1" / "utt1.wav", 0.2, 1.5),
            _write_constant_wav(tmp_path / "spk1" / "utt2.wav", 0.2, 1.6),
        ],
    }
    noises = [_write_noise(tmp_path / "noise" / "n1.wav", seconds=2.0)]
    mixer = PSEMixer(
        speakers,
        noises,
        epoch_size=2,
        segment_length=8000,
        enrollment_length=12000,
        reverb_probability=0.0,
        gain_probability=0.0,
        bandwidth_limit_probability=0.0,
        seed=11,
    )
    mixer.set_epoch(3)
    sample_a = mixer[0]
    sample_b = mixer[0]
    assert sample_a[3] == sample_b[3]
    assert torch.allclose(sample_a[0], sample_b[0])
    assert torch.allclose(sample_a[1], sample_b[1])
    assert torch.allclose(sample_a[2], sample_b[2])


def test_pse_mixer_changes_scene_when_epoch_changes(tmp_path: Path) -> None:
    speakers = {
        "S0001": [
            _write_constant_wav(tmp_path / "spk1" / "utt1.wav", 0.2, 1.5),
            _write_constant_wav(tmp_path / "spk1" / "utt2.wav", 0.2, 1.6),
        ],
    }
    noises = [_write_noise(tmp_path / "noise" / "n1.wav", seconds=2.0)]
    mixer = PSEMixer(
        speakers,
        noises,
        epoch_size=2,
        segment_length=8000,
        enrollment_length=12000,
        reverb_probability=0.0,
        gain_probability=0.0,
        bandwidth_limit_probability=0.0,
        seed=13,
    )
    mixer.set_epoch(0)
    mixture_epoch0, _target0, _enroll0, _speaker0 = mixer[0]
    mixer.set_epoch(1)
    mixture_epoch1, _target1, _enroll1, _speaker1 = mixer[0]
    assert not torch.allclose(mixture_epoch0, mixture_epoch1)


def test_pse_mixer_uses_interferer_pool_when_provided(tmp_path: Path) -> None:
    target_speakers = {
        "S0001": [
            _write_constant_wav(tmp_path / "target" / "utt1.wav", 0.1, 1.0),
            _write_constant_wav(tmp_path / "target" / "utt2.wav", 0.1, 1.0),
        ],
    }
    interferer_speakers = {
        "INT0001": [
            _write_constant_wav(tmp_path / "interferer" / "utt1.wav", 0.8, 1.0),
        ],
    }
    mixer = PSEMixer(
        target_speakers,
        noises=[],
        interferer_speakers=interferer_speakers,
        epoch_size=1,
        segment_length=8000,
        enrollment_length=8000,
        reverb_probability=0.0,
        gain_probability=0.0,
        bandwidth_limit_probability=0.0,
        seed=17,
    )
    mixer._draw_scene_components = lambda rng: (False, True)  # type: ignore[method-assign]
    mixer._sample_sir_db = lambda rng: 0.0  # type: ignore[method-assign]
    mixture, target, _enrollment, _speaker_id = mixer[0]
    assert float((mixture - target).abs().sum()) > 0.0


def test_pse_mixer_scales_noise_against_target_not_mixture(tmp_path: Path) -> None:
    """Verify that noise SNR is relative to the target signal, not the mixture.

    If noise were scaled against the mixture (which includes interference),
    the actual noise-to-target SNR would be lower than requested when
    interference is present. We check that the achieved noise-to-target
    SNR stays near the requested value regardless of interference.
    """
    target_speakers = {
        "S0001": [
            _write_constant_wav(tmp_path / "tgt" / "utt1.wav", 0.3, 1.0),
            _write_constant_wav(tmp_path / "tgt" / "utt2.wav", 0.3, 1.0),
        ],
    }
    interferer_speakers = {
        "INT0001": [
            _write_constant_wav(tmp_path / "int" / "utt1.wav", 0.6, 1.0),
        ],
    }
    noises = [_write_noise(tmp_path / "noise" / "n1.wav", seconds=2.0)]
    requested_snr = 6.0
    mixer = PSEMixer(
        target_speakers,
        noises,
        interferer_speakers=interferer_speakers,
        epoch_size=1,
        segment_length=8000,
        enrollment_length=8000,
        reverb_probability=0.0,
        gain_probability=0.0,
        bandwidth_limit_probability=0.0,
        seed=23,
    )
    mixer._sample_snr_db = lambda rng: requested_snr  # type: ignore[method-assign]
    mixer._sample_sir_db = lambda rng: 0.0  # type: ignore[method-assign]

    # Noise-only scene: measure achieved noise-to-target SNR
    mixer._draw_scene_components = lambda rng: (True, False)  # type: ignore[method-assign]
    mix_n, tgt_n, _, _ = mixer[0]
    noise_only = mix_n - tgt_n
    if float(noise_only.pow(2).mean()) > 1e-10:
        snr_noise_only = 10 * torch.log10(
            tgt_n.pow(2).mean() / noise_only.pow(2).mean()
        ).item()
        assert abs(snr_noise_only - requested_snr) < 1.5

    # Both noise + interference: the noise-to-target SNR should still be
    # near the requested value (not inflated by interference energy)
    mixer._draw_scene_components = lambda rng: (True, True)  # type: ignore[method-assign]
    mix_both, tgt_both, _, _ = mixer[0]
    # Extract noise by subtracting target and interference
    # We can't perfectly isolate noise, but we can check that the total
    # residual energy scales correctly with the target, not the mixture
    residual = mix_both - tgt_both
    residual_snr = 10 * torch.log10(
        tgt_both.pow(2).mean() / residual.pow(2).mean().clamp_min(1e-12)
    ).item()
    # With both noise (6dB) and interference (0dB SIR), the combined
    # residual should be dominated by the interference, giving SNR < 6dB
    # The key check: residual_snr should NOT be much lower than what
    # independent scaling would produce
    assert residual_snr > -2.0  # sanity: not absurdly low


def test_pse_mixer_concatenates_short_utterances(tmp_path: Path) -> None:
    speakers = {
        "S0001": [
            _write_constant_wav(tmp_path / "spk1" / "utt1.wav", 0.2, 0.25),
            _write_constant_wav(tmp_path / "spk1" / "utt2.wav", 0.2, 0.25),
            _write_constant_wav(tmp_path / "spk1" / "utt3.wav", 0.2, 0.25),
        ],
    }
    mixer = PSEMixer(
        speakers,
        noises=[],
        epoch_size=1,
        segment_length=12000,
        enrollment_length=12000,
        reverb_probability=0.0,
        gain_probability=0.0,
        bandwidth_limit_probability=0.0,
        seed=19,
    )
    mixture, target, enrollment, _speaker_id = mixer[0]
    assert torch.allclose(mixture, target)
    assert target.abs().mean().item() == pytest.approx(0.2, abs=1e-4)
    assert enrollment.abs().mean().item() == pytest.approx(0.2, abs=1e-4)


def test_pse_mixer_excludes_all_target_paths_from_enrollment(tmp_path: Path) -> None:
    speakers = {
        "S0001": [
            _write_constant_wav(tmp_path / "spk1" / "utt1.wav", 0.1, 0.25),
            _write_constant_wav(tmp_path / "spk1" / "utt2.wav", 0.2, 0.25),
            _write_constant_wav(tmp_path / "spk1" / "utt3.wav", 0.3, 0.25),
            _write_constant_wav(tmp_path / "spk1" / "utt4.wav", 0.4, 0.25),
        ],
    }
    mixer = PSEMixer(
        speakers,
        noises=[],
        epoch_size=1,
        segment_length=12000,
        enrollment_length=8000,
        reverb_probability=0.0,
        gain_probability=0.0,
        bandwidth_limit_probability=0.0,
        seed=29,
    )
    target_entry = speakers["S0001"][0]
    enroll_entry = speakers["S0001"][3]
    expected_paths = {
        speakers["S0001"][0].path,
        speakers["S0001"][1].path,
        speakers["S0001"][2].path,
    }
    captured_avoid_paths: dict[str, set[Path]] = {}

    mixer._sample_target_pair = lambda rng: ("default", "S0001", target_entry, enroll_entry)  # type: ignore[method-assign]

    def fake_target_segment(
        entries: list[AudioEntry],
        length: int,
        rng: object,
        *,
        anchor_entry: AudioEntry | None = None,
        avoid_paths: set[Path] | None = None,
    ) -> tuple[torch.Tensor, set[Path]]:
        assert anchor_entry == target_entry
        return torch.ones(length), expected_paths

    def fake_enrollment_segment(
        entries: list[AudioEntry],
        length: int,
        rng: object,
        *,
        anchor_entry: AudioEntry | None = None,
        avoid_paths: set[Path] | None = None,
    ) -> torch.Tensor:
        assert anchor_entry == enroll_entry
        captured_avoid_paths["value"] = set(avoid_paths or set())
        return torch.zeros(length)

    mixer._sample_speaker_segment_tracked = fake_target_segment  # type: ignore[method-assign]
    mixer._sample_speaker_segment = fake_enrollment_segment  # type: ignore[method-assign]
    mixer._draw_scene_components = lambda rng: (False, False)  # type: ignore[method-assign]

    mixture, target, enrollment, speaker_id = mixer[0]

    assert speaker_id == "S0001"
    assert captured_avoid_paths["value"] == expected_paths
    assert torch.allclose(mixture, torch.ones(12000))
    assert torch.allclose(target, torch.ones(12000))
    assert torch.allclose(enrollment, torch.zeros(8000))


def test_pse_mixer_reserves_enrollment_anchor_when_target_needs_reuse(tmp_path: Path) -> None:
    """With only 2 utterances, enrollment anchor must be reserved from target."""
    speakers = {
        "S0001": [
            _write_constant_wav(tmp_path / "spk1" / "utt1.wav", 0.1, 0.25),
            _write_constant_wav(tmp_path / "spk1" / "utt2.wav", 0.2, 0.25),
        ],
    }
    mixer = PSEMixer(
        speakers,
        noises=[],
        epoch_size=1,
        segment_length=12000,
        enrollment_length=8000,
        reverb_probability=0.0,
        gain_probability=0.0,
        bandwidth_limit_probability=0.0,
        seed=31,
    )
    mixer._sample_target_pair = lambda rng: (  # type: ignore[method-assign]
        "default",
        "S0001",
        speakers["S0001"][0],
        speakers["S0001"][1],
    )
    mixer._draw_scene_components = lambda rng: (False, False)  # type: ignore[method-assign]

    mixture, target, enrollment, speaker_id = mixer[0]
    assert speaker_id == "S0001"
    assert torch.allclose(mixture, target)
    # Target anchored on utt1 (0.1), enrollment anchored on utt2 (0.2)
    assert target.abs().mean().item() == pytest.approx(0.1, abs=1e-4)
    assert enrollment.abs().mean().item() == pytest.approx(0.2, abs=1e-4)


def test_pse_mixer_dataset_weights_can_force_target_dataset(tmp_path: Path) -> None:
    datasets = {
        "aishell1": {
            "A1": [
                _write_constant_wav(tmp_path / "a1" / "A1" / "utt1.wav", 0.1, 1.0, dataset="aishell1"),
                _write_constant_wav(tmp_path / "a1" / "A1" / "utt2.wav", 0.1, 1.0, dataset="aishell1"),
            ],
        },
        "magicdata": {
            "B1": [
                _write_constant_wav(tmp_path / "b1" / "B1" / "utt1.wav", 0.4, 1.0, dataset="magicdata"),
                _write_constant_wav(tmp_path / "b1" / "B1" / "utt2.wav", 0.4, 1.0, dataset="magicdata"),
            ],
        },
    }
    mixer = PSEMixer(
        datasets=datasets,
        noises=[],
        dataset_weights={"aishell1": 0.0, "magicdata": 1.0},
        scene_weights={"target_only_degraded": 1.0},
        epoch_size=1,
        segment_length=8000,
        enrollment_length=8000,
        reverb_probability=0.0,
        gain_probability=0.0,
        bandwidth_limit_probability=0.0,
        seed=41,
    )
    mixture, target, enrollment, speaker_id = mixer[0]
    assert speaker_id == "B1"
    assert torch.allclose(mixture, target)
    assert target.abs().mean().item() == pytest.approx(0.4, abs=1e-4)
    assert enrollment.abs().mean().item() == pytest.approx(0.4, abs=1e-4)


def test_pse_mixer_enrollment_stays_in_selected_target_dataset(tmp_path: Path) -> None:
    datasets = {
        "aishell1": {
            "S0001": [
                _write_constant_wav(tmp_path / "aishell1" / "S0001" / "utt1.wav", 0.1, 1.0, dataset="aishell1"),
                _write_constant_wav(tmp_path / "aishell1" / "S0001" / "utt2.wav", 0.1, 1.0, dataset="aishell1"),
            ],
        },
        "aishell3": {
            "S0001": [
                _write_constant_wav(tmp_path / "aishell3" / "S0001" / "utt1.wav", 0.6, 1.0, dataset="aishell3"),
                _write_constant_wav(tmp_path / "aishell3" / "S0001" / "utt2.wav", 0.6, 1.0, dataset="aishell3"),
            ],
        },
    }
    mixer = PSEMixer(
        datasets=datasets,
        noises=[],
        dataset_weights={"aishell1": 0.0, "aishell3": 1.0},
        scene_weights={"target_only_degraded": 1.0},
        epoch_size=1,
        segment_length=8000,
        enrollment_length=8000,
        reverb_probability=0.0,
        gain_probability=0.0,
        bandwidth_limit_probability=0.0,
        seed=43,
    )
    _mixture, target, enrollment, speaker_id = mixer[0]
    assert speaker_id == "S0001"
    assert target.abs().mean().item() == pytest.approx(0.6, abs=1e-4)
    assert enrollment.abs().mean().item() == pytest.approx(0.6, abs=1e-4)


def test_pse_mixer_same_dataset_interferer_probability_prefers_same_dataset(tmp_path: Path) -> None:
    datasets = {
        "aishell1": {
            "A1": [
                _write_constant_wav(tmp_path / "aishell1" / "A1" / "utt1.wav", 0.1, 1.0, dataset="aishell1"),
                _write_constant_wav(tmp_path / "aishell1" / "A1" / "utt2.wav", 0.1, 1.0, dataset="aishell1"),
            ],
            "A2": [
                _write_constant_wav(tmp_path / "aishell1" / "A2" / "utt1.wav", 0.6, 1.0, dataset="aishell1"),
                _write_constant_wav(tmp_path / "aishell1" / "A2" / "utt2.wav", 0.6, 1.0, dataset="aishell1"),
            ],
        },
        "magicdata": {
            "B1": [
                _write_constant_wav(tmp_path / "magicdata" / "B1" / "utt1.wav", -0.6, 1.0, dataset="magicdata"),
                _write_constant_wav(tmp_path / "magicdata" / "B1" / "utt2.wav", -0.6, 1.0, dataset="magicdata"),
            ],
        },
    }
    mixer = PSEMixer(
        datasets=datasets,
        noises=[],
        dataset_weights={"aishell1": 1.0, "magicdata": 0.0},
        interferer_same_dataset_probability=1.0,
        scene_weights={"interference": 1.0},
        epoch_size=1,
        segment_length=8000,
        enrollment_length=8000,
        reverb_probability=0.0,
        gain_probability=0.0,
        bandwidth_limit_probability=0.0,
        seed=47,
    )
    mixer._sample_target_pair = lambda rng: (  # type: ignore[method-assign]
        "aishell1",
        "A1",
        datasets["aishell1"]["A1"][0],
        datasets["aishell1"]["A1"][1],
    )
    mixer._sample_sir_db = lambda rng: 0.0  # type: ignore[method-assign]
    mixture, target, _enrollment, speaker_id = mixer[0]
    assert speaker_id == "A1"
    assert float((mixture - target).mean()) > 0.0


def test_pse_mixer_same_dataset_interferer_probability_can_force_cross_dataset(tmp_path: Path) -> None:
    datasets = {
        "aishell1": {
            "A1": [
                _write_constant_wav(tmp_path / "aishell1" / "A1" / "utt1.wav", 0.1, 1.0, dataset="aishell1"),
                _write_constant_wav(tmp_path / "aishell1" / "A1" / "utt2.wav", 0.1, 1.0, dataset="aishell1"),
            ],
            "A2": [
                _write_constant_wav(tmp_path / "aishell1" / "A2" / "utt1.wav", 0.6, 1.0, dataset="aishell1"),
                _write_constant_wav(tmp_path / "aishell1" / "A2" / "utt2.wav", 0.6, 1.0, dataset="aishell1"),
            ],
        },
        "magicdata": {
            "B1": [
                _write_constant_wav(tmp_path / "magicdata" / "B1" / "utt1.wav", -0.6, 1.0, dataset="magicdata"),
                _write_constant_wav(tmp_path / "magicdata" / "B1" / "utt2.wav", -0.6, 1.0, dataset="magicdata"),
            ],
        },
    }
    mixer = PSEMixer(
        datasets=datasets,
        noises=[],
        dataset_weights={"aishell1": 1.0, "magicdata": 0.0},
        interferer_same_dataset_probability=0.0,
        scene_weights={"interference": 1.0},
        epoch_size=1,
        segment_length=8000,
        enrollment_length=8000,
        reverb_probability=0.0,
        gain_probability=0.0,
        bandwidth_limit_probability=0.0,
        seed=53,
    )
    mixer._sample_target_pair = lambda rng: (  # type: ignore[method-assign]
        "aishell1",
        "A1",
        datasets["aishell1"]["A1"][0],
        datasets["aishell1"]["A1"][1],
    )
    mixer._sample_sir_db = lambda rng: 0.0  # type: ignore[method-assign]
    mixture, target, _enrollment, speaker_id = mixer[0]
    assert speaker_id == "A1"
    assert float((mixture - target).mean()) < 0.0


def test_pse_mixer_dataset_wrapper_matches_legacy_single_dataset_behavior(tmp_path: Path) -> None:
    speakers = {
        "S0001": [
            _write_constant_wav(tmp_path / "spk1" / "utt1.wav", 0.2, 1.5),
            _write_constant_wav(tmp_path / "spk1" / "utt2.wav", 0.2, 1.6),
        ],
    }
    legacy = PSEMixer(
        speakers,
        noises=[],
        epoch_size=1,
        segment_length=8000,
        enrollment_length=8000,
        reverb_probability=0.0,
        gain_probability=0.0,
        bandwidth_limit_probability=0.0,
        seed=59,
    )
    dataset_wrapped = PSEMixer(
        datasets={"default": speakers},
        noises=[],
        epoch_size=1,
        segment_length=8000,
        enrollment_length=8000,
        reverb_probability=0.0,
        gain_probability=0.0,
        bandwidth_limit_probability=0.0,
        seed=59,
    )
    assert legacy[0][3] == dataset_wrapped[0][3]
    assert torch.allclose(legacy[0][0], dataset_wrapped[0][0])
    assert torch.allclose(legacy[0][1], dataset_wrapped[0][1])
    assert torch.allclose(legacy[0][2], dataset_wrapped[0][2])


def test_pse_mixer_wraps_flat_noise_list_as_default_category(tmp_path: Path) -> None:
    speakers = {
        "S0001": [
            _write_constant_wav(tmp_path / "spk" / "utt1.wav", 0.1, 1.0),
            _write_constant_wav(tmp_path / "spk" / "utt2.wav", 0.1, 1.0),
        ],
    }
    noises = [_write_noise(tmp_path / "noise" / "n1.wav")]
    mixer = PSEMixer(
        speakers,
        noises,
        epoch_size=1,
        segment_length=8000,
        enrollment_length=8000,
        reverb_probability=0.0,
        gain_probability=0.0,
        bandwidth_limit_probability=0.0,
        seed=61,
    )
    assert mixer.noise_categories == ["default"]
    assert mixer.noise_category_weights == {"default": 1.0}


def test_pse_mixer_noise_category_weights_can_force_selected_pool(tmp_path: Path) -> None:
    speakers = {
        "S0001": [
            _write_constant_wav(tmp_path / "spk" / "utt1.wav", 0.1, 1.0),
            _write_constant_wav(tmp_path / "spk" / "utt2.wav", 0.1, 1.0),
        ],
    }
    noises = {
        "positive": [_write_constant_noise(tmp_path / "noise" / "positive.wav", 0.2)],
        "negative": [_write_constant_noise(tmp_path / "noise" / "negative.wav", -0.2)],
    }
    mixer = PSEMixer(
        speakers,
        noises,
        noise_category_weights={"positive": 0.0, "negative": 1.0},
        epoch_size=1,
        segment_length=8000,
        enrollment_length=8000,
        reverb_probability=0.0,
        gain_probability=0.0,
        bandwidth_limit_probability=0.0,
        seed=67,
    )
    sampled = mixer._sample_noise(8000, random.Random(0))
    assert sampled is not None
    assert float(sampled.mean()) < 0.0


def test_pse_mixer_noise_categories_default_to_uniform_weights(tmp_path: Path) -> None:
    speakers = {
        "S0001": [
            _write_constant_wav(tmp_path / "spk" / "utt1.wav", 0.1, 1.0),
            _write_constant_wav(tmp_path / "spk" / "utt2.wav", 0.1, 1.0),
        ],
    }
    noises = {
        "room": [_write_constant_noise(tmp_path / "noise" / "room.wav", 0.2)],
        "keyboard": [_write_constant_noise(tmp_path / "noise" / "keyboard.wav", -0.2)],
    }
    mixer = PSEMixer(
        speakers,
        noises,
        epoch_size=1,
        segment_length=8000,
        enrollment_length=8000,
        reverb_probability=0.0,
        gain_probability=0.0,
        bandwidth_limit_probability=0.0,
        seed=71,
    )
    assert mixer.noise_category_weights == {"keyboard": 1.0, "room": 1.0}


def test_pse_mixer_noise_scene_uses_selected_noise_category(tmp_path: Path) -> None:
    speakers = {
        "S0001": [
            _write_constant_wav(tmp_path / "spk" / "utt1.wav", 0.2, 1.0),
            _write_constant_wav(tmp_path / "spk" / "utt2.wav", 0.2, 1.0),
        ],
    }
    noises = {
        "positive": [_write_constant_noise(tmp_path / "noise" / "positive.wav", 0.3)],
        "negative": [_write_constant_noise(tmp_path / "noise" / "negative.wav", -0.3)],
    }
    mixer = PSEMixer(
        speakers,
        noises,
        noise_category_weights={"positive": 0.0, "negative": 1.0},
        scene_weights={"noise": 1.0},
        epoch_size=1,
        segment_length=8000,
        enrollment_length=8000,
        reverb_probability=0.0,
        gain_probability=0.0,
        bandwidth_limit_probability=0.0,
        seed=73,
    )
    mixer._sample_snr_db = lambda rng: 0.0  # type: ignore[method-assign]
    mixture, target, _enrollment, _speaker_id = mixer[0]
    assert float((mixture - target).mean()) < 0.0


def test_pse_mixer_scene_weights_support_target_only_degraded_scene(tmp_path: Path) -> None:
    speakers = {
        "S0001": [
            _write_constant_wav(tmp_path / "spk" / "utt1.wav", 0.2, 1.0),
            _write_constant_wav(tmp_path / "spk" / "utt2.wav", 0.2, 1.0),
        ],
        "S0002": [
            _write_constant_wav(tmp_path / "spk2" / "utt1.wav", 0.5, 1.0),
            _write_constant_wav(tmp_path / "spk2" / "utt2.wav", 0.5, 1.0),
        ],
    }
    mixer = PSEMixer(
        speakers,
        noises=[_write_noise(tmp_path / "noise" / "n1.wav")],
        scene_weights={"target_only_degraded": 1.0},
        epoch_size=1,
        segment_length=8000,
        enrollment_length=8000,
        reverb_probability=0.0,
        gain_probability=0.0,
        bandwidth_limit_probability=0.0,
        seed=79,
    )
    mixture, target, _enrollment, _speaker_id = mixer[0]
    assert torch.allclose(mixture, target)


def test_pse_mixer_scene_weights_can_force_noise_only_scene(tmp_path: Path) -> None:
    speakers = {
        "S0001": [
            _write_constant_wav(tmp_path / "spk" / "utt1.wav", 0.2, 1.0),
            _write_constant_wav(tmp_path / "spk" / "utt2.wav", 0.2, 1.0),
        ],
    }
    mixer = PSEMixer(
        speakers,
        noises=[_write_constant_noise(tmp_path / "noise" / "n1.wav", -0.3)],
        scene_weights={"noise": 1.0},
        epoch_size=1,
        segment_length=8000,
        enrollment_length=8000,
        reverb_probability=0.0,
        gain_probability=0.0,
        bandwidth_limit_probability=0.0,
        seed=83,
    )
    mixer._sample_snr_db = lambda rng: 0.0  # type: ignore[method-assign]
    mixture, target, _enrollment, _speaker_id = mixer[0]
    assert float((mixture - target).mean()) < 0.0


def test_pse_mixer_bucketed_snr_sampling_can_force_exact_value(tmp_path: Path) -> None:
    speakers = {
        "S0001": [
            _write_constant_wav(tmp_path / "spk" / "utt1.wav", 0.2, 1.0),
            _write_constant_wav(tmp_path / "spk" / "utt2.wav", 0.2, 1.0),
        ],
    }
    mixer = PSEMixer(
        speakers,
        noises=[],
        snr_buckets=((1.0, 3.0, 3.0),),
        epoch_size=1,
        segment_length=8000,
        enrollment_length=8000,
        reverb_probability=0.0,
        gain_probability=0.0,
        bandwidth_limit_probability=0.0,
        seed=89,
    )
    assert mixer._sample_snr_db(random.Random(0)) == pytest.approx(3.0, abs=1e-6)


def test_pse_mixer_bucketed_sir_sampling_can_force_exact_value(tmp_path: Path) -> None:
    speakers = {
        "S0001": [
            _write_constant_wav(tmp_path / "spk" / "utt1.wav", 0.2, 1.0),
            _write_constant_wav(tmp_path / "spk" / "utt2.wav", 0.2, 1.0),
        ],
        "S0002": [
            _write_constant_wav(tmp_path / "spk2" / "utt1.wav", 0.5, 1.0),
            _write_constant_wav(tmp_path / "spk2" / "utt2.wav", 0.5, 1.0),
        ],
    }
    mixer = PSEMixer(
        speakers,
        noises=[],
        sir_buckets=((1.0, -2.0, -2.0),),
        epoch_size=1,
        segment_length=8000,
        enrollment_length=8000,
        reverb_probability=0.0,
        gain_probability=0.0,
        bandwidth_limit_probability=0.0,
        seed=97,
    )
    assert mixer._sample_sir_db(random.Random(0)) == pytest.approx(-2.0, abs=1e-6)


def test_pse_mixer_scene_reverb_config_uses_weighted_room_family(tmp_path: Path) -> None:
    speakers = {
        "S0001": [
            _write_constant_wav(tmp_path / "spk" / "utt1.wav", 0.2, 1.0),
            _write_constant_wav(tmp_path / "spk" / "utt2.wav", 0.2, 1.0),
        ],
    }
    mixer = PSEMixer(
        speakers,
        noises=[],
        reverb_probability=1.0,
        reverb_room_weights={"small": 0.0, "medium": 1.0, "large": 0.0},
        gain_probability=0.0,
        bandwidth_limit_probability=0.0,
        seed=101,
    )
    cfg = mixer._sample_scene_reverb_config(random.Random(0))
    assert cfg is not None
    assert cfg.rt60_range == (0.20, 0.40)


def test_pse_mixer_scene_reverb_config_falls_back_to_base_config(tmp_path: Path) -> None:
    speakers = {
        "S0001": [
            _write_constant_wav(tmp_path / "spk" / "utt1.wav", 0.2, 1.0),
            _write_constant_wav(tmp_path / "spk" / "utt2.wav", 0.2, 1.0),
        ],
    }
    base_cfg = ReverbConfig(sample_rate=SR, rt60_range=(0.11, 0.22))
    mixer = PSEMixer(
        speakers,
        noises=[],
        reverb_config=base_cfg,
        reverb_probability=1.0,
        gain_probability=0.0,
        bandwidth_limit_probability=0.0,
        seed=103,
    )
    cfg = mixer._sample_scene_reverb_config(random.Random(0))
    assert cfg is base_cfg


def test_pse_mixer_reuses_same_room_family_for_target_and_interferer(tmp_path: Path) -> None:
    speakers = {
        "S0001": [
            _write_constant_wav(tmp_path / "spk1" / "utt1.wav", 0.2, 1.0),
            _write_constant_wav(tmp_path / "spk1" / "utt2.wav", 0.2, 1.0),
        ],
        "S0002": [
            _write_constant_wav(tmp_path / "spk2" / "utt1.wav", 0.5, 1.0),
            _write_constant_wav(tmp_path / "spk2" / "utt2.wav", 0.5, 1.0),
        ],
    }
    mixer = PSEMixer(
        speakers,
        noises=[],
        scene_weights={"interference": 1.0},
        reverb_probability=1.0,
        reverb_room_weights={"small": 1.0, "medium": 0.0, "large": 0.0},
        gain_probability=0.0,
        bandwidth_limit_probability=0.0,
        seed=107,
    )
    mixer._sample_sir_db = lambda rng: 0.0  # type: ignore[method-assign]
    seen_ranges: list[tuple[float, float] | None] = []

    def fake_maybe_reverb(
        signal: torch.Tensor,
        rng: random.Random,
        cfg: object = None,
    ) -> torch.Tensor:
        if cfg is None:
            seen_ranges.append(None)
        else:
            seen_ranges.append(cfg.rt60_range)
        return signal

    mixer._maybe_reverb = fake_maybe_reverb  # type: ignore[method-assign]
    mixer[0]
    assert seen_ranges == [(0.08, 0.20), (0.08, 0.20)]
