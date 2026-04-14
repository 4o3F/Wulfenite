"""Smoke tests for checkpoint-aware inference paths."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from wulfenite.inference.utils import build_model_from_checkpoint
from wulfenite.models import SpeakerBeamSSConfig, WulfeniteTSE
from wulfenite.training.checkpoint import load_checkpoint, save_checkpoint


SR = 16000


def _tiny_separator_config() -> SpeakerBeamSSConfig:
    return SpeakerBeamSSConfig(
        enc_channels=16,
        bottleneck_channels=16,
        speaker_embed_dim=192,
        r1_repeats=1,
        r2_repeats=1,
        conv_blocks_per_repeat=1,
        hidden_channels=16,
        s4d_state_dim=8,
        target_presence_head=True,
    )


def _separator_checkpoint_config(
    separator_config: SpeakerBeamSSConfig,
) -> dict[str, int | str | bool]:
    return {
        "enc_channels": separator_config.enc_channels,
        "bottleneck_channels": separator_config.bottleneck_channels,
        "speaker_embed_dim": separator_config.speaker_embed_dim,
        "hidden_channels": separator_config.hidden_channels,
        "r1_repeats": separator_config.r1_repeats,
        "r2_repeats": separator_config.r2_repeats,
        "conv_blocks_per_repeat": separator_config.conv_blocks_per_repeat,
        "s4d_state_dim": separator_config.s4d_state_dim,
        "s4d_ffn_multiplier": separator_config.s4d_ffn_multiplier,
        "separator_lookahead_frames": separator_config.separator_lookahead_frames,
        "lookahead_policy": separator_config.lookahead_policy,
        "target_presence_head": separator_config.target_presence_head,
        "mask_activation": separator_config.mask_activation,
    }


def _tiny_tse() -> WulfeniteTSE:
    return WulfeniteTSE.from_campplus(
        campplus_checkpoint=None,
        separator_config=_tiny_separator_config(),
    ).eval()


def test_whole_vs_streaming_end_to_end() -> None:
    torch.manual_seed(0)
    tse = _tiny_tse()

    t = 8 * tse.separator.config.enc_stride
    mixture = torch.randn(1, t)
    enrollment_wav = torch.randn(SR)

    with torch.no_grad():
        whole_outputs = tse(mixture, enrollment_wav)
        whole_clean = whole_outputs["clean"]

        emb = tse.encode_enrollment(enrollment_wav)
        state = tse.separator.initial_streaming_state(batch_size=1)
        chunk_size = 2 * tse.separator.config.enc_stride
        pieces = []
        for start in range(0, t, chunk_size):
            chunk = mixture[..., start:start + chunk_size]
            if chunk.shape[-1] != chunk_size:
                continue
            out, state = tse.separator.streaming_step(
                chunk, emb, state, s4d_state_decay=1.0,
            )
            pieces.append(out)
        stream_clean = torch.cat(pieces, dim=-1)

    assert whole_clean.shape == stream_clean.shape
    diff = (whole_clean - stream_clean).abs().max().item()
    assert diff < 1e-4


def test_checkpoint_preserves_inference_output(tmp_path: Path) -> None:
    torch.manual_seed(1)
    tse = _tiny_tse()
    t = 4 * tse.separator.config.enc_stride
    mixture = torch.randn(1, t)
    enrollment = torch.randn(SR)

    with torch.no_grad():
        before = tse(mixture, enrollment)["clean"]

    ckpt_path = tmp_path / "ckpt.pt"
    save_checkpoint(
        ckpt_path,
        model=tse,
        config=_separator_checkpoint_config(tse.separator.config),
    )

    tse2 = _tiny_tse()
    load_checkpoint(ckpt_path, model=tse2)
    tse2.eval()
    with torch.no_grad():
        after = tse2(mixture, enrollment)["clean"]

    assert torch.allclose(before, after, atol=1e-6)


def test_build_model_from_checkpoint_roundtrip(tmp_path: Path) -> None:
    torch.manual_seed(2)
    tse = _tiny_tse()
    t = 4 * tse.separator.config.enc_stride
    mixture = torch.randn(1, t)
    enrollment = torch.randn(SR)

    with torch.no_grad():
        before = tse(mixture, enrollment)["clean"]

    ckpt_path = tmp_path / "campplus.pt"
    save_checkpoint(
        ckpt_path,
        model=tse,
        config=_separator_checkpoint_config(tse.separator.config),
    )

    loaded, info = build_model_from_checkpoint(ckpt_path)

    assert loaded.separator.config == _tiny_separator_config()
    assert info["skipped_legacy_keys"] == []
    assert info["skipped_incompatible_keys"] == []

    with torch.no_grad():
        after = loaded(mixture, enrollment)["clean"]

    assert torch.allclose(before, after, atol=1e-6)


def test_build_model_from_checkpoint_skips_legacy_keys(tmp_path: Path) -> None:
    torch.manual_seed(3)
    tse = _tiny_tse()
    ckpt_path = tmp_path / "legacy_camplus.pt"
    save_checkpoint(
        ckpt_path,
        model=tse,
        config=_separator_checkpoint_config(tse.separator.config),
    )

    payload = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    payload["model_state_dict"]["speaker_encoder.to_separator.weight"] = torch.randn(16, 192)
    payload["model_state_dict"]["speaker_encoder.classifier.weight"] = torch.randn(4, 16)
    payload["model_state_dict"]["separator.speaker_gamma.weight"] = torch.randn(16, 16)
    payload["model_state_dict"]["separator.speaker_beta.weight"] = torch.randn(16, 16)
    torch.save(payload, ckpt_path)

    loaded, info = build_model_from_checkpoint(ckpt_path)

    assert isinstance(loaded, WulfeniteTSE)
    assert "speaker_encoder.to_separator.weight" in info["skipped_legacy_keys"]
    assert "speaker_encoder.classifier.weight" in info["skipped_legacy_keys"]
    assert "separator.speaker_gamma.weight" in info["skipped_incompatible_keys"]
    assert "separator.speaker_beta.weight" in info["skipped_incompatible_keys"]


def test_build_model_from_checkpoint_rejects_legacy_learnable(tmp_path: Path) -> None:
    ckpt_path = tmp_path / "learnable.pt"
    torch.save(
        {
            "model_state_dict": {
                "speaker_encoder.frame.0.linear.weight": torch.randn(4, 4, 1),
            },
            "config": {"encoder_type": "learnable"},
        },
        ckpt_path,
    )

    with pytest.raises(RuntimeError, match="learnable d-vector"):
        build_model_from_checkpoint(ckpt_path)


def test_presence_head_output_shape() -> None:
    torch.manual_seed(4)
    tse = _tiny_tse()
    t = 4 * tse.separator.config.enc_stride
    mixture = torch.randn(3, t)
    enrollment = torch.randn(SR)

    with torch.no_grad():
        outputs = tse(mixture, enrollment)

    assert "presence_logit" in outputs
    assert outputs["presence_logit"].shape == (3,)
    assert torch.isfinite(outputs["presence_logit"]).all()


def test_streaming_state_initializes_to_zero() -> None:
    tse = _tiny_tse()
    state = tse.separator.initial_streaming_state(batch_size=2)
    assert torch.all(state["encoder_buffer"] == 0)
    assert torch.all(state["decoder_overlap"] == 0)
    assert torch.all(state["skip_buffer"] == 0)
    assert torch.all(state["output_buffer"] == 0)
    assert state["block_startup_remaining"] == [
        block.tcn_right_context for block in tse.separator._all_blocks()
    ]
    for block_state in state["block_states"]:
        if torch.is_tensor(block_state):
            assert torch.all(block_state == 0)
        elif isinstance(block_state, tuple):
            for part in block_state:
                assert torch.all(part == 0)


def test_checkpoint_roundtrip_preserves_separator_lookahead(tmp_path: Path) -> None:
    cfg = _tiny_separator_config()
    cfg.separator_lookahead_frames = 1
    tse = WulfeniteTSE.from_campplus(
        campplus_checkpoint=None,
        separator_config=cfg,
    ).eval()

    ckpt_path = tmp_path / "lookahead.pt"
    save_checkpoint(
        ckpt_path,
        model=tse,
        config=_separator_checkpoint_config(cfg),
    )

    loaded, _ = build_model_from_checkpoint(ckpt_path)
    assert loaded.separator.config.separator_lookahead_frames == 1
    assert loaded.separator.config.lookahead_policy == "post_fusion_frontloaded"
