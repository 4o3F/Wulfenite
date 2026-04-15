"""Unit tests for the pDFNet2 model stack."""

from __future__ import annotations

import pytest
import torch
from torch import nn
from typing import Callable

from wulfenite.inference import Enhancer
from wulfenite.models import (
    Conv2dNormAct,
    ConvTranspose2dNormAct,
    DfNet,
    DfOp,
    ECAPA_TDNN_GLOB_c512,
    ECAPA_TDNN_GLOB_c1024,
    GroupedGRU,
    GroupedLinear,
    PDfNet2,
    PDfNet2Plus,
    SpeakerEncoder,
    TinyECAPA,
    erb_fb,
    erb_fb_inverse,
)


def _num_params(module: nn.Module) -> int:
    return sum(param.numel() for param in module.parameters())


def _save_checkpoint(
    tmp_path,
    model_factory: Callable[[], nn.Module],
    *,
    wrapped: bool = False,
    add_projection: bool = False,
    add_module_prefix: bool = False,
    add_frontend: bool = False,
):
    model = model_factory()
    state_dict = model.state_dict()
    if add_projection:
        state_dict["projection.weight"] = torch.randn(4, 192)
        state_dict["projection.bias"] = torch.randn(4)
    if add_module_prefix:
        state_dict = {f"module.{key}": value for key, value in state_dict.items()}
    if add_frontend:
        state_dict["frontend.conv.weight"] = torch.randn(8, 1, 3)
    payload = {"model": state_dict} if wrapped else state_dict
    checkpoint_path = tmp_path / "speaker.pt"
    torch.save(payload, checkpoint_path)
    return checkpoint_path


def test_erb_filterbank_shapes_and_inverse() -> None:
    fb = erb_fb()
    inv = erb_fb_inverse(fb)
    assert fb.shape == (24, 161)
    assert inv.shape == (24, 161)
    assert torch.all(fb >= 0)
    assert torch.allclose(fb.sum(dim=1), torch.ones(24), atol=1e-4)


def test_dfop_alpha_zero_is_identity() -> None:
    spec = torch.randn(2, 1, 8, 161, 2)
    coefs = torch.randn(2, 8, 5, 96, 2)
    alpha = torch.zeros(2, 8, 1)
    df = DfOp()
    out = df(spec, coefs, alpha)
    assert torch.allclose(out, spec)


def test_grouped_modules_shapes() -> None:
    linear = GroupedLinear(8, 12, groups=2)
    x = torch.randn(2, 5, 8)
    y = linear(x)
    assert y.shape == (2, 5, 12)

    gru = GroupedGRU(12, 16, num_layers=2, groups=1)
    out, state = gru(y)
    assert out.shape == (2, 5, 16)
    assert state.shape == (2, 2, 16)


def test_conv_modules_shapes() -> None:
    conv = Conv2dNormAct(1, 4, kernel_size=(3, 3), separable=True, bias=False)
    x = torch.randn(2, 1, 6, 24)
    y = conv(x)
    assert y.shape == (2, 4, 6, 24)

    convt = ConvTranspose2dNormAct(4, 4, kernel_size=(1, 3), fstride=2, bias=False)
    z = convt(torch.randn(2, 4, 6, 12))
    assert z.size(0) == 2
    assert z.size(1) == 4
    assert z.size(2) == 6
    assert z.size(3) >= 23


def test_dfnet2_forward_and_reconstruction_shapes() -> None:
    model = DfNet().eval()
    waveform = torch.randn(2, 3200)
    spec, _ = model.waveform_to_spec(waveform)
    enhanced_spec, gains, lsnr, alpha = model(spec)
    enhanced_waveform = model.spec_to_waveform(enhanced_spec, length=waveform.size(-1))
    assert enhanced_spec.shape == spec.shape
    assert gains.shape == (2, 1, spec.size(2), 24)
    assert lsnr.shape == (2, spec.size(2), 1)
    assert alpha.shape == (2, spec.size(2), 1)
    assert enhanced_waveform.shape == waveform.shape


def test_pdfnet2_forward_shapes() -> None:
    model = PDfNet2().eval()
    waveform = torch.randn(2, 3200)
    spec, _ = model.waveform_to_spec(waveform)
    speaker_emb = torch.randn(2, 192)
    enhanced_spec, gains, lsnr, alpha = model(spec, speaker_emb)
    assert enhanced_spec.shape == spec.shape
    assert gains.shape[-1] == 24
    assert lsnr.shape == (2, spec.size(2), 1)
    assert alpha.shape == (2, spec.size(2), 1)


def test_ecapa_tdnn_c512_forward_shapes() -> None:
    model = ECAPA_TDNN_GLOB_c512(feat_dim=80, embed_dim=192, pooling_func="ASTP").eval()
    out4, emb = model(torch.randn(2, 100, 80))
    assert out4.shape == (2, 512, 100)
    assert emb.shape == (2, 192)


def test_ecapa_tdnn_c1024_forward_shapes() -> None:
    model = ECAPA_TDNN_GLOB_c1024(feat_dim=80, embed_dim=192, pooling_func="ASTP").eval()
    out4, emb = model(torch.randn(2, 100, 80))
    assert out4.shape == (2, 1024, 100)
    assert emb.shape == (2, 192)


def test_tiny_ecapa_embeddings_chunks_and_param_count() -> None:
    model = TinyECAPA().eval()
    waveform = torch.randn(2, 16000)
    emb = model(waveform)
    chunks = model.forward_chunks(waveform, chunk_seconds=1.0, overlap=0.5)
    assert emb.shape == (2, 192)
    assert chunks.shape[0] == 2
    assert chunks.shape[1] == 192
    assert chunks.shape[2] >= 1
    assert 120_000 <= model.num_parameters <= 200_000


def test_speaker_encoder_rejects_dim_mismatch() -> None:
    class DummyBackbone(nn.Module):
        def forward(self, waveform: torch.Tensor) -> torch.Tensor:
            return waveform.mean(dim=-1, keepdim=True).repeat(1, 128)

    with pytest.raises(ValueError, match="embedding_dim"):
        SpeakerEncoder(backend=DummyBackbone(), embedding_dim=128, output_dim=192)


def test_speaker_encoder_loads_raw_checkpoint(tmp_path) -> None:
    checkpoint_path = _save_checkpoint(
        tmp_path,
        lambda: ECAPA_TDNN_GLOB_c512(feat_dim=80, embed_dim=192, pooling_func="ASTP"),
    )
    model = SpeakerEncoder(checkpoint_path=checkpoint_path).eval()
    emb = model(torch.randn(2, 16000))
    assert emb.shape == (2, 192)


def test_speaker_encoder_loads_wrapped_checkpoint(tmp_path) -> None:
    checkpoint_path = _save_checkpoint(
        tmp_path,
        lambda: ECAPA_TDNN_GLOB_c512(feat_dim=80, embed_dim=192, pooling_func="ASTP"),
        wrapped=True,
    )
    model = SpeakerEncoder(checkpoint_path=checkpoint_path).eval()
    emb = model(torch.randn(2, 16000))
    assert emb.shape == (2, 192)


def test_speaker_encoder_strips_projection_keys(tmp_path) -> None:
    checkpoint_path = _save_checkpoint(
        tmp_path,
        lambda: ECAPA_TDNN_GLOB_c512(feat_dim=80, embed_dim=192, pooling_func="ASTP"),
        wrapped=True,
        add_projection=True,
    )
    model = SpeakerEncoder(checkpoint_path=checkpoint_path).eval()
    emb = model(torch.randn(2, 16000))
    assert emb.shape == (2, 192)


def test_speaker_encoder_auto_detects_variant(tmp_path) -> None:
    checkpoint_path = _save_checkpoint(
        tmp_path,
        lambda: ECAPA_TDNN_GLOB_c1024(feat_dim=80, embed_dim=192, pooling_func="ASTP"),
    )
    model = SpeakerEncoder(checkpoint_path=checkpoint_path).eval()
    emb = model(torch.randn(2, 16000))
    assert emb.shape == (2, 192)
    assert model.backend.layer1.conv.weight.shape[0] == 1024


def test_speaker_encoder_strips_module_prefix(tmp_path) -> None:
    checkpoint_path = _save_checkpoint(
        tmp_path,
        lambda: ECAPA_TDNN_GLOB_c512(feat_dim=80, embed_dim=192, pooling_func="ASTP"),
        wrapped=True,
        add_module_prefix=True,
    )
    model = SpeakerEncoder(checkpoint_path=checkpoint_path).eval()
    emb = model(torch.randn(2, 16000))
    assert emb.shape == (2, 192)


def test_speaker_encoder_rejects_frontend_checkpoint(tmp_path) -> None:
    checkpoint_path = _save_checkpoint(
        tmp_path,
        lambda: ECAPA_TDNN_GLOB_c512(feat_dim=80, embed_dim=192, pooling_func="ASTP"),
        add_frontend=True,
    )
    with pytest.raises(ValueError, match="frontend"):
        SpeakerEncoder(checkpoint_path=checkpoint_path)


def test_pdfnet2_plus_forward_shapes() -> None:
    model = PDfNet2Plus(tiny_ecapa=TinyECAPA()).eval()
    waveform = torch.randn(2, 16000)
    speaker_emb = torch.randn(2, 192)
    spec, _ = model.pdfnet2.waveform_to_spec(waveform)
    conditioning = model.refine_conditioning(waveform, speaker_emb, spec.size(2))
    enhanced_spec, gains, lsnr, alpha = model(waveform, speaker_emb)
    assert conditioning.shape == (2, spec.size(2), 193)
    assert enhanced_spec.shape == spec.shape
    assert gains.shape[-1] == 24
    assert lsnr.shape == (2, spec.size(2), 1)
    assert alpha.shape == (2, spec.size(2), 1)


def test_pdfnet2_plus_sigmoid_gate_is_smooth() -> None:
    model = PDfNet2Plus(
        tiny_ecapa=TinyECAPA(),
        similarity_activation="sigmoid",
        similarity_threshold=1.0 / 6.0,
    )
    cosine = torch.tensor([-1.0, 0.0, 1.0 / 6.0, 0.5, 1.0])
    gate = model.similarity_to_gate(cosine)
    assert torch.all(gate[1:] > gate[:-1])
    assert torch.all(gate > 0.0)
    assert torch.all(gate < 1.0)
    assert torch.isclose(gate[2], torch.tensor(0.5), atol=1e-6)


def test_pdfnet2_plus_clamp_gate_backward_compat() -> None:
    model = PDfNet2Plus(
        tiny_ecapa=TinyECAPA(),
        similarity_activation="clamp",
        alpha_scale=6.0,
    )
    cosine = torch.tensor([-1.0, 0.0, 1.0 / 12.0, 1.0 / 6.0, 0.5])
    expected = torch.tensor([0.0, 0.0, 0.5, 1.0, 1.0])
    assert torch.allclose(model.similarity_to_gate(cosine), expected)


def test_model_parameter_counts_are_reasonable() -> None:
    assert 1_800_000 <= _num_params(DfNet()) <= 2_600_000
    assert 120_000 <= _num_params(TinyECAPA()) <= 200_000


def test_pdfnet2_plus_keeps_tiny_ecapa_frozen_in_train_mode() -> None:
    model = PDfNet2Plus(tiny_ecapa=TinyECAPA())
    model.train()
    assert not model.tiny_ecapa.training


def test_pdfnet2_plus_rejects_missing_tiny_ecapa() -> None:
    with pytest.raises(ValueError, match="explicit TinyECAPA"):
        PDfNet2Plus()


def test_enhancer_batch_and_streaming_smoke() -> None:
    model = DfNet().eval()
    enhancer = Enhancer(model, device="cpu")
    waveform = torch.randn(1, 3200)
    full = enhancer.enhance(waveform)
    chunks: list[torch.Tensor] = []
    for start in range(0, waveform.size(-1), 160):
        end = min(waveform.size(-1), start + 160)
        chunks.append(
            enhancer.enhance_streaming(
                waveform[:, start:end],
                finalize=end == waveform.size(-1),
            )
        )
    streamed = torch.cat(chunks, dim=-1)
    assert full.shape == waveform.shape
    assert streamed.shape == waveform.shape
    assert torch.allclose(streamed, full, atol=2e-3, rtol=1e-3)


def test_pdfnet2_plus_streaming_matches_batch() -> None:
    torch.manual_seed(0)
    model = PDfNet2Plus(
        tiny_ecapa=TinyECAPA(),
        conditioning_mode="causal",
        conditioning_update_interval_frames=1,
        similarity_ema_decay=0.0,
        conditioning_energy_threshold=0.0,
    ).eval()
    enhancer = Enhancer(model, device="cpu")
    waveform = torch.randn(1, 3200)
    speaker_emb = torch.nn.functional.normalize(torch.randn(1, 192), dim=-1)
    full = enhancer.enhance(waveform, speaker_emb=speaker_emb, conditioning_mode="causal")
    chunks: list[torch.Tensor] = []
    for start in range(0, waveform.size(-1), 320):
        end = min(waveform.size(-1), start + 320)
        chunks.append(
            enhancer.enhance_streaming(
                waveform[:, start:end],
                speaker_emb=speaker_emb,
                finalize=end == waveform.size(-1),
            )
        )
    streamed = torch.cat(chunks, dim=-1)
    assert full.shape == waveform.shape
    assert streamed.shape == waveform.shape
    assert torch.allclose(streamed, full, atol=2e-3, rtol=1e-3)
