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
