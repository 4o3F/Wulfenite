"""Shape / plumbing tests for the SpeakerBeam-SS separator."""

from __future__ import annotations

import torch

from wulfenite.models.speakerbeam_ss import (
    SpeakerBeamSS,
    SpeakerBeamSSConfig,
)


def _small_config() -> SpeakerBeamSSConfig:
    """A tiny config to keep the tests fast."""
    return SpeakerBeamSSConfig(
        enc_channels=32,
        bottleneck_channels=16,
        speaker_embed_dim=192,
        r1_repeats=1,
        r2_repeats=1,
        conv_blocks_per_repeat=1,
        hidden_channels=32,
        s4d_state_dim=8,
        target_presence_head=True,
    )


def _s4d_state_norm(model: SpeakerBeamSS, state: dict) -> float:
    total = torch.zeros(())
    for block_state in state["block_states"]:
        _, s4d_state = block_state
        total = total + s4d_state.pow(2).sum().cpu()
    return float(total.sqrt().item())


def test_speakerbeam_forward_shape() -> None:
    torch.manual_seed(0)
    cfg = _small_config()
    model = SpeakerBeamSS(cfg).eval()

    batch = 2
    t = 5 * cfg.enc_stride + cfg.enc_kernel_size - cfg.enc_stride
    mixture = torch.randn(batch, t)
    embedding = torch.randn(batch, cfg.speaker_embed_dim)
    embedding = torch.nn.functional.normalize(embedding, p=2, dim=-1)

    with torch.no_grad():
        out = model(mixture, embedding)

    clean = out["clean"]
    assert clean.dim() == 2
    assert clean.size(0) == batch
    assert abs(clean.size(-1) - t) <= cfg.enc_stride
    assert torch.isfinite(clean).all()

    assert "mask" in out
    assert torch.all(out["mask"] >= 0)
    assert "presence_logit" in out
    assert out["presence_logit"].shape == (batch,)


def test_speakerbeam_initial_state_shapes() -> None:
    cfg = _small_config()
    model = SpeakerBeamSS(cfg).eval()
    state = model.initial_state(batch_size=1)
    assert len(state) == len(model._all_blocks())


def test_speakerbeam_streaming_matches_forward() -> None:
    torch.manual_seed(0)
    cfg = _small_config()
    model = SpeakerBeamSS(cfg).eval()

    stride = cfg.enc_stride
    t = 8 * stride
    batch = 2
    mixture = torch.randn(batch, t)
    embedding = torch.nn.functional.normalize(
        torch.randn(batch, cfg.speaker_embed_dim), p=2, dim=-1,
    )

    with torch.no_grad():
        out_whole = model(mixture, embedding)["clean"]

        for chunk_size in (stride, 2 * stride, 4 * stride):
            state = model.initial_streaming_state(batch_size=batch)
            pieces = []
            for start in range(0, t, chunk_size):
                chunk = mixture[..., start:start + chunk_size]
                if chunk.shape[-1] != chunk_size:
                    continue
                y, state = model.streaming_step(
                    chunk, embedding, state, s4d_state_decay=1.0,
                )
                pieces.append(y)
            out_stream = torch.cat(pieces, dim=-1)
            assert out_stream.shape == out_whole.shape
            diff = (out_stream - out_whole).abs().max().item()
            assert diff < 1e-4


def test_streaming_s4d_decay_bounds_state() -> None:
    torch.manual_seed(2)
    cfg = _small_config()
    model = SpeakerBeamSS(cfg).eval()

    batch = 1
    steps = 64
    chunk = torch.randn(batch, cfg.enc_stride)
    embedding = torch.nn.functional.normalize(
        torch.randn(batch, cfg.speaker_embed_dim), p=2, dim=-1,
    )
    decay_state = model.initial_streaming_state(batch_size=batch)
    no_decay_state = model.initial_streaming_state(batch_size=batch)
    decay_norms = []
    no_decay_norms = []

    with torch.no_grad():
        for _ in range(steps):
            _, decay_state = model.streaming_step(
                chunk, embedding, decay_state, s4d_state_decay=0.99,
            )
            _, no_decay_state = model.streaming_step(
                chunk, embedding, no_decay_state, s4d_state_decay=1.0,
            )
            decay_norms.append(_s4d_state_norm(model, decay_state))
            no_decay_norms.append(_s4d_state_norm(model, no_decay_state))

    decay_norms_t = torch.tensor(decay_norms)
    no_decay_norms_t = torch.tensor(no_decay_norms)

    assert torch.isfinite(decay_norms_t).all()
    assert decay_norms_t[-1] < no_decay_norms_t[-1]
    assert decay_norms_t.max() < no_decay_norms_t.max()


def test_streaming_s4d_decay_default_matches_explicit() -> None:
    """Default s4d_state_decay=1.0 matches passing it explicitly."""
    torch.manual_seed(3)
    cfg = _small_config()
    model = SpeakerBeamSS(cfg).eval()

    batch = 2
    chunk_size = 2 * cfg.enc_stride
    chunks = [torch.randn(batch, chunk_size) for _ in range(4)]
    embedding = torch.nn.functional.normalize(
        torch.randn(batch, cfg.speaker_embed_dim), p=2, dim=-1,
    )
    default_state = model.initial_streaming_state(batch_size=batch)
    explicit_state = model.initial_streaming_state(batch_size=batch)

    with torch.no_grad():
        for chunk in chunks:
            default_out, default_state = model.streaming_step(
                chunk, embedding, default_state,
            )
            explicit_out, explicit_state = model.streaming_step(
                chunk, embedding, explicit_state, s4d_state_decay=1.0,
            )
            assert torch.equal(default_out, explicit_out)


def test_speaker_modulation_initializes_to_identity() -> None:
    cfg = _small_config()
    model = SpeakerBeamSS(cfg).eval()

    expected_weight = torch.zeros(cfg.bottleneck_channels, cfg.speaker_embed_dim)
    expected_bias = torch.ones(cfg.bottleneck_channels)

    assert torch.allclose(model.speaker_projection.weight, expected_weight, atol=1e-8)
    assert torch.allclose(model.speaker_projection.bias, expected_bias, atol=1e-8)


def test_speaker_modulation_helper_is_noop_at_init() -> None:
    torch.manual_seed(1)
    cfg = _small_config()
    model = SpeakerBeamSS(cfg).eval()

    batch, frames = 2, 32
    feat = torch.randn(batch, cfg.bottleneck_channels, frames)
    embedding = torch.nn.functional.normalize(
        torch.randn(batch, cfg.speaker_embed_dim), p=2, dim=-1,
    )

    with torch.no_grad():
        actual = model._apply_speaker_modulation(feat, embedding)

    diff = (actual - feat).abs().max().item()
    assert diff < 1e-6
