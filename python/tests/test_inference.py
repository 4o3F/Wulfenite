"""Smoke tests for the inference CLIs.

Because CAM++ weights are not available locally, these tests do NOT
exercise the full ``run_whole`` / ``run_streaming`` entry points (they
require a real CAM++ .bin file). Instead, they test:

1. The whole-utterance vs streaming numerical equivalence on random
   weights end-to-end through ``WulfeniteTSE`` — same guarantee as
   the SpeakerBeam-SS separator test, but including the CAM++
   encoder path and the TSE wrapper.
2. The checkpoint round-trip preserves output bit-for-bit.
"""

from __future__ import annotations

from pathlib import Path

import torch

from wulfenite.models import (
    CAMPPlus,
    SpeakerBeamSS,
    SpeakerBeamSSConfig,
    WulfeniteTSE,
)
from wulfenite.training.checkpoint import load_checkpoint, save_checkpoint
from wulfenite.training.config import TrainingConfig


SR = 16000


def _tiny_tse() -> WulfeniteTSE:
    cfg = SpeakerBeamSSConfig(
        enc_channels=16,
        bottleneck_channels=192,
        num_repeats=1,
        num_blocks_per_repeat=2,
        hidden_channels=16,
        s4d_state_dim=8,
    )
    encoder = CAMPPlus(feat_dim=80, embedding_size=192).eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    separator = SpeakerBeamSS(cfg)
    return WulfeniteTSE(speaker_encoder=encoder, separator=separator).eval()


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
