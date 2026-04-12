"""Shape / plumbing tests for the SpeakerBeam-SS separator."""

from __future__ import annotations

import torch

from wulfenite.models.speakerbeam_ss import SpeakerBeamSS, SpeakerBeamSSConfig


def _small_config() -> SpeakerBeamSSConfig:
    """A tiny config to keep the tests fast."""
    return SpeakerBeamSSConfig(
        enc_channels=32,
        bottleneck_channels=16,
        speaker_embed_dim=192,
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

    assert "presence_logit" in out
    assert out["presence_logit"].shape == (batch,)


def test_speakerbeam_initial_state_shapes() -> None:
    cfg = _small_config()
    model = SpeakerBeamSS(cfg).eval()
    state = model.initial_state(batch_size=1)
    assert len(state) == len(model.blocks)


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
                y, state = model.streaming_step(chunk, embedding, state)
                pieces.append(y)
            out_stream = torch.cat(pieces, dim=-1)
            assert out_stream.shape == out_whole.shape
            diff = (out_stream - out_whole).abs().max().item()
            assert diff < 1e-4


def test_speaker_film_initializes_to_identity() -> None:
    cfg = _small_config()
    model = SpeakerBeamSS(cfg).eval()

    expected_gamma = torch.zeros(cfg.bottleneck_channels, cfg.speaker_embed_dim)
    expected_beta = torch.zeros(cfg.bottleneck_channels, cfg.speaker_embed_dim)

    assert torch.allclose(model.speaker_gamma.weight, expected_gamma, atol=1e-8)
    assert torch.allclose(model.speaker_beta.weight, expected_beta, atol=1e-8)
    assert model.speaker_gamma.bias is None
    assert model.speaker_beta.bias is None


def test_speaker_film_helper_is_noop_at_init() -> None:
    torch.manual_seed(1)
    cfg = _small_config()
    model = SpeakerBeamSS(cfg).eval()

    batch, frames = 2, 32
    feat = torch.randn(batch, cfg.bottleneck_channels, frames)
    embedding = torch.nn.functional.normalize(
        torch.randn(batch, cfg.speaker_embed_dim), p=2, dim=-1,
    )

    with torch.no_grad():
        actual = model._apply_speaker_conditioning(feat, embedding)

    diff = (actual - feat).abs().max().item()
    assert diff < 1e-6
