"""Smoke tests for the inference CLIs.

These tests do NOT exercise the full ``run_whole`` / ``run_streaming``
entry points. Instead, they test:

1. The whole-utterance vs streaming numerical equivalence on random
   weights end-to-end through ``WulfeniteTSE`` — same guarantee as
   the SpeakerBeam-SS separator test, but including the TSE wrapper.
2. The checkpoint round-trip preserves output bit-for-bit.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from wulfenite.inference.utils import build_model_from_checkpoint
from wulfenite.models import SpeakerBeamSSConfig, WulfeniteTSE
from wulfenite.training.checkpoint import load_checkpoint, save_checkpoint
from wulfenite.training.config import TrainingConfig


SR = 16000


def _tiny_separator_config() -> SpeakerBeamSSConfig:
    return SpeakerBeamSSConfig(
        enc_channels=16,
        bottleneck_channels=16,
        num_repeats=1,
        r1_blocks=1,
        r2_blocks=1,
        hidden_channels=16,
        s4d_state_dim=8,
    )


def _separator_checkpoint_config(
    separator_config: SpeakerBeamSSConfig,
    *,
    encoder_type: str,
) -> dict[str, int | str]:
    return {
        "encoder_type": encoder_type,
        "enc_channels": separator_config.enc_channels,
        "bottleneck_channels": separator_config.bottleneck_channels,
        "hidden_channels": separator_config.hidden_channels,
        "num_repeats": separator_config.num_repeats,
        "r1_blocks": separator_config.r1_blocks,
        "r2_blocks": separator_config.r2_blocks,
        "s4d_state_dim": separator_config.s4d_state_dim,
    }


def _tiny_tse() -> WulfeniteTSE:
    cfg = _tiny_separator_config()
    return WulfeniteTSE.from_learnable_dvector(
        num_speakers=4,
        separator_config=cfg,
        dvector_kwargs={
            "tdnn_channels": 16,
            "stats_dim": 16,
            "spec_augment": False,
        },
    ).eval()


def _checkpoint_tse(num_speakers: int = 4) -> WulfeniteTSE:
    return WulfeniteTSE.from_learnable_dvector(
        num_speakers=num_speakers,
        dvector_kwargs={"spec_augment": False},
    ).eval()


def _tiny_campplus_tse(freeze: bool) -> WulfeniteTSE:
    return WulfeniteTSE.from_campplus(
        campplus_checkpoint=None,
        separator_config=_tiny_separator_config(),
        freeze_backbone=freeze,
    ).eval()


def test_whole_vs_streaming_end_to_end() -> None:
    """End-to-end TSE forward vs streaming_step must agree numerically."""
    torch.manual_seed(0)
    tse = _tiny_tse()

    # Random enrollment / mixture audio.
    T = 8 * tse.separator.config.enc_stride  # 8 encoder frames
    mixture = torch.randn(1, T)
    enrollment_wav = torch.randn(SR)  # 1 second

    with torch.no_grad():
        # Whole-utterance path.
        whole_outputs = tse(mixture, enrollment_wav)
        whole_clean = whole_outputs["clean"]

        # Streaming path, manually threaded.
        emb = tse.encode_enrollment(enrollment_wav)
        state = tse.separator.initial_streaming_state(batch_size=1)
        chunk_size = 2 * tse.separator.config.enc_stride  # 2 frames per chunk
        pieces = []
        for start in range(0, T, chunk_size):
            chunk = mixture[..., start:start + chunk_size]
            if chunk.shape[-1] != chunk_size:
                continue
            out, state = tse.separator.streaming_step(chunk, emb, state)
            pieces.append(out)
        stream_clean = torch.cat(pieces, dim=-1)

    assert whole_clean.shape == stream_clean.shape
    diff = (whole_clean - stream_clean).abs().max().item()
    assert diff < 1e-4, (
        f"whole vs streaming TSE disagree by {diff:.2e} (>1e-4)"
    )


def test_checkpoint_preserves_inference_output(tmp_path: Path) -> None:
    """Saving and re-loading a checkpoint must not change the model output."""
    torch.manual_seed(1)
    tse = _tiny_tse()
    T = 4 * tse.separator.config.enc_stride
    mixture = torch.randn(1, T)
    enrollment = torch.randn(SR)

    with torch.no_grad():
        before = tse(mixture, enrollment)["clean"]

    ckpt_path = tmp_path / "ckpt.pt"
    save_checkpoint(ckpt_path, model=tse, config=TrainingConfig())

    tse2 = _tiny_tse()
    load_checkpoint(ckpt_path, model=tse2)
    tse2.eval()
    with torch.no_grad():
        after = tse2(mixture, enrollment)["clean"]

    assert torch.allclose(before, after, atol=1e-6), (
        "inference output diverged after checkpoint round-trip"
    )


def test_build_model_from_checkpoint_learnable_path(tmp_path: Path) -> None:
    """Inference checkpoints should rebuild through the classifier-free path."""
    torch.manual_seed(3)
    tse = _checkpoint_tse(num_speakers=4)
    T = 4 * tse.separator.config.enc_stride
    mixture = torch.randn(1, T)
    enrollment = torch.randn(SR)

    with torch.no_grad():
        before = tse(mixture, enrollment)["clean"]

    ckpt_path = tmp_path / "learnable.pt"
    save_checkpoint(
        ckpt_path,
        model=tse,
        config=TrainingConfig(),
    )

    loaded, info = build_model_from_checkpoint(ckpt_path)
    assert loaded.speaker_encoder.classifier is None
    assert info["skipped_classifier_keys"]
    assert all(
        key.startswith("speaker_encoder.classifier.")
        for key in info["skipped_classifier_keys"]
    )

    with torch.no_grad():
        after = loaded(mixture, enrollment)["clean"]

    assert torch.allclose(before, after, atol=1e-6), (
        "learnable inference output diverged after checkpoint rebuild"
    )


@pytest.mark.parametrize("encoder_type", ("campplus-frozen", "campplus-finetune"))
def test_build_model_from_checkpoint_campplus_roundtrip(
    tmp_path: Path,
    encoder_type: str,
) -> None:
    """CAM++ checkpoints should rebuild through the strict inference path."""
    torch.manual_seed(4)
    tse = _tiny_campplus_tse(freeze=encoder_type == "campplus-frozen")
    T = 4 * tse.separator.config.enc_stride
    mixture = torch.randn(1, T)
    enrollment = torch.randn(SR)

    with torch.no_grad():
        before = tse(mixture, enrollment)["clean"]

    ckpt_path = tmp_path / f"{encoder_type}.pt"
    save_checkpoint(
        ckpt_path,
        model=tse,
        config=_separator_checkpoint_config(
            tse.separator.config,
            encoder_type=encoder_type,
        ),
    )

    loaded, info = build_model_from_checkpoint(ckpt_path)
    assert loaded.separator.config == _tiny_separator_config()
    assert info["config"]["encoder_type"] == encoder_type

    with torch.no_grad():
        after = loaded(mixture, enrollment)["clean"]

    assert torch.allclose(before, after, atol=1e-6), (
        "CAM++ inference output diverged after checkpoint rebuild"
    )


def test_campplus_whole_vs_streaming_end_to_end() -> None:
    """End-to-end CAM++ TSE forward vs streaming_step must agree numerically."""
    torch.manual_seed(5)
    tse = _tiny_campplus_tse(freeze=True)

    T = 8 * tse.separator.config.enc_stride
    mixture = torch.randn(1, T)
    enrollment_wav = torch.randn(SR)

    with torch.no_grad():
        whole_outputs = tse(mixture, enrollment_wav)
        whole_clean = whole_outputs["clean"]

        emb = tse.encode_enrollment(enrollment_wav)
        state = tse.separator.initial_streaming_state(batch_size=1)
        chunk_size = 2 * tse.separator.config.enc_stride
        pieces = []
        for start in range(0, T, chunk_size):
            chunk = mixture[..., start:start + chunk_size]
            if chunk.shape[-1] != chunk_size:
                continue
            out, state = tse.separator.streaming_step(chunk, emb, state)
            pieces.append(out)
        stream_clean = torch.cat(pieces, dim=-1)

    assert whole_clean.shape == stream_clean.shape
    diff = (whole_clean - stream_clean).abs().max().item()
    assert diff < 1e-4, (
        f"whole vs streaming CAM++ TSE disagree by {diff:.2e} (>1e-4)"
    )


def test_build_model_from_checkpoint_campplus_rejects_incompatible(
    tmp_path: Path,
) -> None:
    """CAM++ inference loading should reject learnable checkpoints."""
    torch.manual_seed(6)
    tse = _tiny_tse()

    ckpt_path = tmp_path / "campplus_incompatible.pt"
    save_checkpoint(
        ckpt_path,
        model=tse,
        config=_separator_checkpoint_config(
            tse.separator.config,
            encoder_type="campplus-frozen",
        ),
    )

    with pytest.raises(RuntimeError, match="CAM\\+\\+ TSE pipeline"):
        build_model_from_checkpoint(ckpt_path)


def test_presence_head_output_shape() -> None:
    """The whole-utterance forward should emit presence_logit of shape [B]."""
    torch.manual_seed(2)
    tse = _tiny_tse()
    T = 4 * tse.separator.config.enc_stride
    mixture = torch.randn(3, T)
    enrollment = torch.randn(SR)

    with torch.no_grad():
        outputs = tse(mixture, enrollment)

    assert "presence_logit" in outputs
    assert outputs["presence_logit"].shape == (3,)
    assert torch.isfinite(outputs["presence_logit"]).all()


def test_streaming_state_initializes_to_zero() -> None:
    """Initial state tensors should all be zero."""
    tse = _tiny_tse()
    state = tse.separator.initial_streaming_state(batch_size=2)
    assert torch.all(state["encoder_buffer"] == 0)
    assert torch.all(state["decoder_overlap"] == 0)
    # block_states is a list of block-specific zero tensors
    for bs in state["block_states"]:
        if torch.is_tensor(bs):
            assert torch.all(bs == 0)
