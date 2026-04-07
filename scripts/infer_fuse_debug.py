"""Debug inference: run BSRNN+CAM++ with optional RMS normalization and
print internal stats to diagnose the "uniform attenuation" failure on
out-of-distribution real audio.

Usage:
    uv run python infer_fuse_debug.py \
        --fuse-ckpt ../assets/campplus/train_phase0a/best.pt \
        --mixture   ../assets/your_real_mixture.wav \
        --enrollment ../assets/your_real_enrollment.wav \
        --output /tmp/debug_out.wav \
        --normalize-rms 0.1

With ``--normalize-rms 0.1`` the mixture and enrollment are rescaled to
match the training distribution (AISHELL-1 mixer also normalizes to 0.1).

The script runs the forward pass in two halves so we can peek at the
speaker embedding, the fuse-layer output, and the mask statistics.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import soundfile as sf
import torch

from bsrnn_campplus import build_bsrnn_campplus, compute_kaldi_fbank


def _load_mono(path: Path, sr: int = 16000) -> torch.Tensor:
    data, file_sr = sf.read(str(path), dtype="float32", always_2d=False)
    if file_sr != sr:
        raise ValueError(f"{path}: sample_rate={file_sr}, expected {sr}")
    if data.ndim == 2:
        data = data.mean(axis=1)
    return torch.from_numpy(data)


def _rms(x: torch.Tensor) -> float:
    return float(torch.sqrt((x * x).mean() + 1e-12))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bsrnn-ckpt", type=Path,
                        default=Path(__file__).resolve().parent.parent
                        / "assets" / "avg_model.pt")
    parser.add_argument("--campplus-ckpt", type=Path,
                        default=Path(__file__).resolve().parent.parent
                        / "assets" / "campplus" / "campplus_cn_common.bin")
    parser.add_argument("--fuse-ckpt", type=Path, default=None)
    parser.add_argument("--mixture", type=Path, required=True)
    parser.add_argument("--enrollment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--normalize-rms", type=float, default=None,
                        help="If set, rescale both inputs to this target RMS "
                             "(AISHELL training used 0.1). Default: no rescaling.")
    parser.add_argument("--rescale-output-to-input", action="store_true",
                        help="Post-processing: rescale the output waveform so "
                             "its RMS matches the mixture RMS. Fixes the "
                             "SI-SDR scale-invariance level drift at inference "
                             "time (a common hack — the separation quality is "
                             "not changed, only the playback level).")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    model = build_bsrnn_campplus(args.bsrnn_ckpt, args.campplus_ckpt, device=device)

    if args.fuse_ckpt is not None:
        ckpt = torch.load(args.fuse_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["state_dict"], strict=True)
        print(f"[load] {args.fuse_ckpt} (epoch={ckpt.get('epoch')})")
    model.eval()

    mixture = _load_mono(args.mixture)
    enrollment = _load_mono(args.enrollment)
    print(f"[input] mixture len={mixture.shape[0]} RMS={_rms(mixture):.4f} "
          f"peak={mixture.abs().max():.4f}")
    print(f"[input] enrollment len={enrollment.shape[0]} RMS={_rms(enrollment):.4f} "
          f"peak={enrollment.abs().max():.4f}")

    if args.normalize_rms is not None:
        scale_mix = args.normalize_rms / (_rms(mixture) + 1e-9)
        scale_enr = args.normalize_rms / (_rms(enrollment) + 1e-9)
        mixture = mixture * scale_mix
        enrollment = enrollment * scale_enr
        print(f"[norm] mixture scaled by {scale_mix:.3f} -> RMS={_rms(mixture):.4f}")
        print(f"[norm] enrollment scaled by {scale_enr:.3f} -> RMS={_rms(enrollment):.4f}")

    mixture = mixture.unsqueeze(0).to(device)       # [1, T_mix]
    enrollment = enrollment.unsqueeze(0).to(device)  # [1, T_enr]

    # -------------- split forward pass for diagnostics --------------
    with torch.no_grad():
        # 1. CAM++ FBank + embedding
        fbank = compute_kaldi_fbank(enrollment[0], sample_rate=16000,
                                    num_mel_bins=80, dither=0.0).unsqueeze(0)
        print(f"[cam++] fbank shape={tuple(fbank.shape)} "
              f"mean={fbank.mean().item():.4f} std={fbank.std().item():.4f} "
              f"min={fbank.min().item():.4f} max={fbank.max().item():.4f}")
        spk_embedding = model.spk_model(fbank)  # [1, 192]
        emb_norm = spk_embedding.norm(dim=-1).item()
        emb_mean = spk_embedding.mean().item()
        emb_std = spk_embedding.std().item()
        print(f"[cam++] embedding norm={emb_norm:.3f} mean={emb_mean:+.4f} "
              f"std={emb_std:.4f}")

        spk_embedding_full = spk_embedding.unsqueeze(1).unsqueeze(3)  # [1,1,192,1]

        # 2. BSRNN spectral front-end
        window = torch.hann_window(model.win, dtype=mixture.dtype, device=mixture.device)
        spec = torch.stft(mixture, n_fft=model.win, hop_length=model.stride,
                          window=window, return_complex=True)
        spec_ri = torch.stack([spec.real, spec.imag], dim=1)

        subband_spec = []
        subband_mix_spec = []
        band_idx = 0
        for band_width in model.band_width:
            subband_spec.append(spec_ri[:, :, band_idx:band_idx + band_width].contiguous())
            subband_mix_spec.append(spec[:, band_idx:band_idx + band_width])
            band_idx += band_width

        batch_size = 1
        subband_feature = []
        for band, bn in zip(subband_spec, model.BN):
            subband_feature.append(
                bn(band.view(batch_size, band.size(1) * band.size(2), -1))
            )
        subband_feature = torch.stack(subband_feature, dim=1)

        # 3. Fuse layer alone — inspect the conditioning signal
        fuse_layer = model.separator.separation[0]
        # Reproduce fuse-layer forward manually so we can capture the
        # transformed conditioning tensor before it multiplies x.
        embed_t = spk_embedding_full.expand(-1, subband_feature.size(1), -1,
                                            subband_feature.size(3))
        embed_t = torch.transpose(embed_t, 2, 3)
        transformed = torch.transpose(fuse_layer.fc(embed_t), 2, 3)
        print(f"[fuse] cond signal mean={transformed.mean().item():+.4f} "
              f"std={transformed.std().item():.4f} "
              f"abs_mean={transformed.abs().mean().item():.4f}")

        # Now continue the full forward pass via the model
        sep_output = model.separator(subband_feature, spk_embedding_full)

        # 4. Inspect the mask stats per band — full complex mask, not just gate.
        # mask = band_out[:, 0] * sigmoid(band_out[:, 1])  is a REAL scalar per
        # (real, imag, band_freq, T) slot. The first dim (=2) splits mask_real
        # vs mask_imag so the effective complex mask per bin is
        #     (mask_real + 1j * mask_imag)
        # and the actual suppression is |mask_complex|.
        all_gates = []
        all_raw_scale = []       # band_out[:, 0]
        all_mask_real = []
        all_mask_imag = []
        for idx, mask_head in enumerate(model.mask):
            band_out = mask_head(sep_output[:, idx]).view(
                batch_size, 2, 2, model.band_width[idx], -1
            )
            raw_scale = band_out[:, 0]            # [B, 2, bw, T] real-valued scale
            gate = torch.sigmoid(band_out[:, 1])  # [B, 2, bw, T] in [0, 1]
            mask = raw_scale * gate               # [B, 2, bw, T]
            all_gates.append(gate.flatten())
            all_raw_scale.append(raw_scale.flatten())
            all_mask_real.append(mask[:, 0].flatten())
            all_mask_imag.append(mask[:, 1].flatten())

        gates = torch.cat(all_gates)
        raw_scale = torch.cat(all_raw_scale)
        mask_real = torch.cat(all_mask_real)
        mask_imag = torch.cat(all_mask_imag)
        mask_mag = torch.sqrt(mask_real * mask_real + mask_imag * mask_imag)

        print(f"[mask ] gate       mean={gates.mean().item():.4f} "
              f"std={gates.std().item():.4f}  (healthy ~0.4-0.7)")
        print(f"[mask ] raw_scale  mean={raw_scale.mean().item():+.4f} "
              f"abs_mean={raw_scale.abs().mean().item():.4f} "
              f"std={raw_scale.std().item():.4f}")
        print(f"[mask ] mask_real  mean={mask_real.mean().item():+.4f} "
              f"abs_mean={mask_real.abs().mean().item():.4f}")
        print(f"[mask ] mask_imag  mean={mask_imag.mean().item():+.4f} "
              f"abs_mean={mask_imag.abs().mean().item():.4f}")
        print(f"[mask ] |mask|     mean={mask_mag.mean().item():.4f} "
              f"median={mask_mag.median().item():.4f} "
              f"p90={mask_mag.kthvalue(int(mask_mag.numel() * 0.9)).values.item():.4f} "
              f"(if this is ~0.02 the model is effectively zeroing the output)")

        # 5. Full forward to get output
        estimate = model(mixture, enrollment)

    # Post-processing: rescale-to-input-RMS if requested
    out_tensor = estimate[0].cpu()
    rms_in = _rms(mixture[0].cpu())
    rms_out_raw = _rms(out_tensor)
    if args.rescale_output_to_input:
        scale = rms_in / (rms_out_raw + 1e-9)
        out_tensor = out_tensor * scale
        # Soft-clip protection: if peak > 0.99 after rescaling, back off
        new_peak = float(out_tensor.abs().max())
        if new_peak > 0.99:
            backoff = 0.99 / new_peak
            out_tensor = out_tensor * backoff
            print(f"[post ] rescale x{scale:.1f} with peak backoff x{backoff:.3f} "
                  f"(would have clipped at {new_peak:.2f})")
        else:
            print(f"[post ] rescale x{scale:.1f} to match input RMS")

    out = out_tensor.numpy()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(args.output), out, 16000)

    peak = float(abs(out).max())
    rms_out = float((out * out).mean() ** 0.5)
    print(f"[output] peak={peak:.4f} rms={rms_out:.4f} "
          f"rms_ratio_out/in={rms_out / (rms_in + 1e-9):.3f} "
          f"peak_dBFS={20 * math.log10(max(peak, 1e-9)):.1f}")
    print(f"[output] saved to {args.output}")


if __name__ == "__main__":
    main()
