"""Shape / plumbing tests for the SpeakerBeam-SS separator.

These are smoke tests that verify the model assembles, runs a forward
pass without errors, and produces outputs of the expected shape. They
do NOT test separation quality — that requires training.
"""

from __future__ import annotations

import torch

from wulfenite.models.speakerbeam_ss import SpeakerBeamSS, SpeakerBeamSSConfig


def _small_config() -> SpeakerBeamSSConfig:
    """A tiny config to keep the tests fast."""
    return SpeakerBeamSSConfig(
        enc_channels=32,
        bottleneck_channels=16,   # keep in sync with the embedding in tests
        num_repeats=1,
        num_blocks_per_repeat=2,
        hidden_channels=32,
        s4d_state_dim=8,
    )


def test_speakerbeam_forward_shape() -> None:
    torch.manual_seed(0)
    cfg = _small_config()
    model = SpeakerBeamSS(cfg).eval()

    batch = 2
    # T must be a multiple of enc_stride (160). Use 5 frames = 800 samples.
    t = 5 * cfg.enc_stride + cfg.enc_kernel_size - cfg.enc_stride
    mixture = torch.randn(batch, t)
    embedding = torch.randn(batch, cfg.bottleneck_channels)
    embedding = torch.nn.functional.normalize(embedding, p=2, dim=-1)

    with torch.no_grad():
        out = model(mixture, embedding)

    clean = out["clean"]
    assert clean.dim() == 2
    assert clean.size(0) == batch
    # Decoder output length may differ from input length by a few
    # samples depending on stride arithmetic; assert it is within one
    # encoder stride of the input length.
    assert abs(clean.size(-1) - t) <= cfg.enc_stride
    assert torch.isfinite(clean).all()

    assert "presence_logit" in out
    assert out["presence_logit"].shape == (batch,)


def test_speakerbeam_initial_state_shapes() -> None:
    cfg = _small_config()
    model = SpeakerBeamSS(cfg).eval()
    state = model.initial_state(batch_size=1)
    assert len(state) == len(model.blocks)


def test_speakerbeam_streaming_matches_forward() -> None:
    """streaming_step on any chunking must match forward() exactly.

    This is the central guarantee that lets us train with forward()
    and deploy with streaming_step()/ONNX. The test feeds the same
    utterance through both paths and asserts the outputs are equal
    up to floating-point rounding.
    """
    torch.manual_seed(0)
    cfg = _small_config()
    model = SpeakerBeamSS(cfg).eval()

    stride = cfg.enc_stride
    T = 8 * stride  # 8 encoder frames
    batch = 2
    mixture = torch.randn(batch, T)
    embedding = torch.nn.functional.normalize(
        torch.randn(batch, cfg.bottleneck_channels), p=2, dim=-1,
    )

    with torch.no_grad():
        # Whole-utterance reference.
        out_whole = model(mixture, embedding)["clean"]

        # Streaming with varying chunk sizes.
        for chunk_size in (stride, 2 * stride, 4 * stride):
            state = model.initial_streaming_state(batch_size=batch)
            pieces = []
            for start in range(0, T, chunk_size):
                chunk = mixture[..., start:start + chunk_size]
                if chunk.shape[-1] != chunk_size:
                    continue  # skip trailing partial
                y, state = model.streaming_step(chunk, embedding, state)
                pieces.append(y)
            out_stream = torch.cat(pieces, dim=-1)
            assert out_stream.shape == out_whole.shape, (
                f"chunk_size={chunk_size}: stream {tuple(out_stream.shape)} "
                f"vs whole {tuple(out_whole.shape)}"
            )
            diff = (out_stream - out_whole).abs().max().item()
            assert diff < 1e-4, (
                f"chunk_size={chunk_size}: streaming / whole-utterance "
                f"disagree by {diff:.2e}"
            )
