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
        bottleneck_channels=16,
        num_repeats=1,
        r1_blocks=1,
        r2_blocks=1,
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


def test_speaker_film_initializes_to_plan_b_point() -> None:
    """Plan C2 invariant: at init, FiLM exactly reproduces Plan B's output.

    - speaker_gamma.weight == sqrt(d) * I  (Plan B's operator)
    - speaker_beta.weight  == 0            (no additive effect)
    - neither layer has bias (bias=False)
    - old speaker_proj attribute is gone
    """
    torch.manual_seed(0)
    cfg = _small_config()  # bottleneck_channels = 16
    model = SpeakerBeamSS(cfg).eval()

    d = cfg.bottleneck_channels
    expected_gamma = (d ** 0.5) * torch.eye(d)
    expected_beta = torch.zeros(d, d)

    assert torch.allclose(
        model.speaker_gamma.weight, expected_gamma, atol=1e-6,
    ), "speaker_gamma.weight should be sqrt(d) * I at init"
    assert torch.allclose(
        model.speaker_beta.weight, expected_beta, atol=1e-8,
    ), "speaker_beta.weight should be zeros at init"

    assert model.speaker_gamma.bias is None
    assert model.speaker_beta.bias is None

    # Plan B's speaker_proj should be gone.
    assert not hasattr(model, "speaker_proj"), \
        "speaker_proj must be replaced by speaker_gamma + speaker_beta"


def test_speaker_film_helper_matches_plan_b_math_at_init() -> None:
    """At init, _apply_speaker_conditioning must equal Plan B's math.

    Plan B computed ``feat * sqrt(d) * e`` at init. Plan C2's FiLM
    helper computes ``feat * gamma(e) + beta(e)``. With the prescribed
    init, these are mathematically identical. This test catches any
    broadcast / shape bug in the helper.
    """
    torch.manual_seed(0)
    cfg = _small_config()
    model = SpeakerBeamSS(cfg).eval()

    d = cfg.bottleneck_channels
    B, T = 2, 32
    feat = torch.randn(B, d, T)
    embedding = torch.nn.functional.normalize(
        torch.randn(B, d), p=2, dim=-1,
    )

    with torch.no_grad():
        actual = model._apply_speaker_conditioning(feat, embedding)
        # Plan B reference: feat * sqrt(d) * embedding
        scale = d ** 0.5
        expected = feat * (scale * embedding).unsqueeze(-1)

    diff = (actual - expected).abs().max().item()
    assert diff < 1e-5, (
        f"FiLM at init should reproduce Plan B's feat * sqrt(d) * e "
        f"exactly; max abs diff = {diff:.2e}"
    )
