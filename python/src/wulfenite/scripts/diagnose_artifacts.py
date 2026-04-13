"""Diagnostic tool for analyzing model output artifacts.

Runs three checks to diagnose electronic/metallic noise in model output:

1. **Identity reconstruction** — encoder/decoder bypass with all-ones mask
   to test whether the learned basis itself introduces artifacts.
2. **Mask distribution** — statistics on the separation mask to detect
   musical-noise signatures (high sparsity + temporal discontinuity).
3. **Output amplitude** — checks whether output exceeds [-1, 1] and would
   be clipped by PCM_16 writes.

Usage:

    # With checkpoint (full diagnosis):
    uv run --directory python python -m wulfenite.scripts.diagnose_artifacts \\
        --checkpoint ../assets/checkpoints/best.pt \\
        --mixture ../assets/samples/real_mixture.wav \\
        --enrollment ../assets/samples/real_enrollment.wav \\
        --out-dir ../assets/diagnostics

    # Without checkpoint (identity reconstruction only):
    uv run --directory python python -m wulfenite.scripts.diagnose_artifacts \\
        --out-dir ../assets/diagnostics
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import pathlib
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

from ..models.speakerbeam_ss import SpeakerBeamSS, SpeakerBeamSSConfig

if platform.system() == "Windows":
    pathlib.PosixPath = pathlib.WindowsPath  # type: ignore[misc, assignment]

SAMPLE_RATE = 16000


# ---------------------------------------------------------------------------
# Check 1: Identity reconstruction
# ---------------------------------------------------------------------------


def check_identity_reconstruction(
    out_dir: Path,
    checkpoint: Path | None = None,
    device: str = "cpu",
) -> dict[str, Any]:
    """Test encoder → identity-mask → decoder reconstruction quality."""
    dev = torch.device(device)

    if checkpoint is not None:
        from ..inference.utils import build_model_from_checkpoint

        tse, info = build_model_from_checkpoint(checkpoint, device=dev)
        model = tse.separator
        source = f"checkpoint (epoch={info.get('epoch')})"
    else:
        model = SpeakerBeamSS(SpeakerBeamSSConfig()).to(dev).eval()
        source = "random weights"

    cfg = model.config
    print(f"\n=== Identity Reconstruction ({source}) ===")
    print(f"  enc_channels={cfg.enc_channels}, kernel={cfg.enc_kernel_size}, "
          f"stride={cfg.enc_stride}")

    # Test signals
    dur = 2.0
    n = int(SAMPLE_RATE * dur)
    t = torch.linspace(0, dur, n, device=dev)
    signals = {
        "sine_440hz": (torch.sin(2 * torch.pi * 440 * t) * 0.5).unsqueeze(0),
        "speech_like": _synth_speech_like(n, dev),
        "chirp": (torch.sin(2 * torch.pi * (100 + 3900 * t / dur) * t) * 0.4).unsqueeze(0),
    }

    results: dict[str, Any] = {"source": source}
    with torch.no_grad():
        for name, signal in signals.items():
            x = signal.unsqueeze(1)
            total_samples = signal.shape[-1]

            pad_left = cfg.enc_kernel_size - cfg.enc_stride
            x_padded = F.pad(x, (pad_left, 0))

            enc = torch.relu(model.encoder(x_padded))
            mask = torch.ones_like(enc)
            recon = model.decoder(enc * mask).squeeze(1)
            recon = recon[..., :total_samples]

            # Metrics
            error = signal - recon
            sig_pow = (signal ** 2).sum()
            err_pow = (error ** 2).sum() + 1e-12
            snr_db = float(10 * torch.log10(sig_pow / err_pow))
            peak_out = float(recon.abs().max())

            # Spectral analysis for hop-rate artifacts
            hop_artifact_ratio = _hop_rate_artifact_ratio(
                signal.squeeze(0).cpu(), recon.squeeze(0).cpu(),
                cfg.enc_stride, SAMPLE_RATE,
            )

            results[name] = {
                "snr_db": round(snr_db, 2),
                "peak_output": round(peak_out, 4),
                "exceeds_1": peak_out > 1.0,
                "hop_artifact_ratio": round(hop_artifact_ratio, 3),
            }
            print(f"  {name}: SNR={snr_db:.1f} dB, peak={peak_out:.4f}, "
                  f"hop_artifact={hop_artifact_ratio:.2f}x, "
                  f"exceeds_1={peak_out > 1.0}")

            # Save reconstructed audio
            wav_path = out_dir / f"identity_recon_{name}.wav"
            sf.write(str(wav_path), recon.squeeze(0).cpu().numpy(),
                     SAMPLE_RATE, subtype="FLOAT")

    return results


def _synth_speech_like(n: int, device: torch.device) -> torch.Tensor:
    """Synthesize a multi-frequency signal mimicking speech spectral shape."""
    t = torch.linspace(0, n / SAMPLE_RATE, n, device=device)
    signal = torch.zeros(n, device=device)
    for freq, amp in [(150, 0.3), (300, 0.2), (600, 0.15),
                      (1200, 0.1), (2400, 0.05), (3600, 0.03)]:
        signal = signal + amp * torch.sin(2 * torch.pi * freq * t)
    # AM envelope
    envelope = 0.5 + 0.5 * torch.sin(2 * torch.pi * 3 * t)
    return (signal * envelope).unsqueeze(0)


def _hop_rate_artifact_ratio(
    orig: torch.Tensor,
    recon: torch.Tensor,
    stride: int,
    sr: int,
) -> float:
    """Ratio of spectral error at hop-rate harmonics vs. overall."""
    n_fft = 2048
    hop = 128
    win = torch.hann_window(n_fft)

    spec_o = torch.stft(orig, n_fft=n_fft, hop_length=hop,
                        win_length=n_fft, window=win,
                        return_complex=True, center=True)
    spec_r = torch.stft(recon, n_fft=n_fft, hop_length=hop,
                        win_length=n_fft, window=win,
                        return_complex=True, center=True)
    diff = (spec_r.abs() - spec_o.abs()).abs()
    avg_all = float(diff.mean())

    hop_freq = sr / stride
    hop_bin = int(round(hop_freq * n_fft / sr))
    artifact_bins = [hop_bin * k for k in range(1, 8)
                     if hop_bin * k < n_fft // 2]
    if not artifact_bins or avg_all < 1e-12:
        return 1.0
    avg_artifact = float(diff[artifact_bins, :].mean())
    return avg_artifact / avg_all


# ---------------------------------------------------------------------------
# Check 2: Mask distribution
# ---------------------------------------------------------------------------


def check_mask_distribution(
    checkpoint: Path,
    mixture_path: Path,
    enrollment_path: Path,
    out_dir: Path,
    device: str = "cpu",
) -> dict[str, Any]:
    """Analyze separation mask statistics for musical-noise signatures."""
    dev = torch.device(device)

    from ..inference.utils import build_model_from_checkpoint

    model, info = build_model_from_checkpoint(checkpoint, device=dev)
    print(f"\n=== Mask Distribution Analysis ===")
    print(f"  Checkpoint: epoch={info.get('epoch')}")

    # Load audio
    mix_wav = _load_mono(mixture_path).unsqueeze(0).to(dev)
    enr_wav = _load_mono(enrollment_path).unsqueeze(0).to(dev)

    with torch.no_grad():
        outputs = model(mix_wav, enr_wav)

    clean = outputs["clean"]
    mask = outputs["mask"]

    # Basic mask stats
    mask_np = mask.cpu().numpy().squeeze(0)  # [C, L]
    stats: dict[str, Any] = {
        "shape": list(mask_np.shape),
        "mean": float(mask_np.mean()),
        "std": float(mask_np.std()),
        "median": float(np.median(mask_np)),
        "min": float(mask_np.min()),
        "max": float(mask_np.max()),
        "sparsity_pct_lt_001": float((mask_np < 0.01).mean() * 100),
        "sparsity_pct_lt_01": float((mask_np < 0.1).mean() * 100),
        "fraction_gt_2": float((mask_np > 2.0).mean() * 100),
        "fraction_gt_5": float((mask_np > 5.0).mean() * 100),
    }

    # Temporal smoothness (mean absolute diff between adjacent frames)
    temporal_diff = np.abs(np.diff(mask_np, axis=1))
    stats["temporal_smoothness_mean"] = float(temporal_diff.mean())
    stats["temporal_smoothness_std"] = float(temporal_diff.std())

    # Per-channel analysis
    ch_means = mask_np.mean(axis=1)
    stats["channel_mean_min"] = float(ch_means.min())
    stats["channel_mean_max"] = float(ch_means.max())
    stats["channels_near_zero_pct"] = float((ch_means < 0.01).mean() * 100)
    stats["channels_above_1_pct"] = float((ch_means > 1.0).mean() * 100)

    # Output amplitude check
    clean_np = clean.cpu().numpy().squeeze(0)
    stats["output_peak"] = float(np.abs(clean_np).max())
    stats["output_exceeds_1"] = bool(np.abs(clean_np).max() > 1.0)
    stats["output_rms"] = float(np.sqrt((clean_np ** 2).mean()))

    # Musical noise signature detection
    # High sparsity + low temporal smoothness = musical noise
    is_sparse = stats["sparsity_pct_lt_001"] > 30.0
    is_temporally_rough = stats["temporal_smoothness_mean"] > 0.3
    stats["musical_noise_risk"] = (
        "HIGH" if is_sparse and is_temporally_rough
        else "MEDIUM" if is_sparse or is_temporally_rough
        else "LOW"
    )

    print(f"  Mask shape: {mask_np.shape}")
    print(f"  Mean: {stats['mean']:.4f}, Std: {stats['std']:.4f}")
    print(f"  Min: {stats['min']:.4f}, Max: {stats['max']:.4f}")
    print(f"  Sparsity (<0.01): {stats['sparsity_pct_lt_001']:.1f}%")
    print(f"  Values >2.0: {stats['fraction_gt_2']:.1f}%")
    print(f"  Values >5.0: {stats['fraction_gt_5']:.1f}%")
    print(f"  Temporal smoothness: {stats['temporal_smoothness_mean']:.4f}")
    print(f"  Channels near-zero: {stats['channels_near_zero_pct']:.1f}%")
    print(f"  Output peak: {stats['output_peak']:.4f} (exceeds 1.0: {stats['output_exceeds_1']})")
    print(f"  Musical noise risk: {stats['musical_noise_risk']}")

    # Save outputs
    sf.write(str(out_dir / "model_output.wav"), clean_np, SAMPLE_RATE,
             subtype="FLOAT")
    sf.write(str(out_dir / "model_output_pcm16.wav"), clean_np, SAMPLE_RATE)

    # Save mask stats JSON
    with open(out_dir / "mask_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  Saved mask_stats.json and audio to {out_dir}")

    # Try visualization
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Mask heatmap (subsample channels for visibility)
        step = max(1, mask_np.shape[0] // 256)
        axes[0, 0].imshow(mask_np[::step, :], aspect="auto", origin="lower",
                          vmin=0, vmax=2.0, cmap="viridis")
        axes[0, 0].set_title("Mask heatmap (channels × time)")
        axes[0, 0].set_xlabel("Frame")
        axes[0, 0].set_ylabel(f"Channel (stride={step})")
        axes[0, 0].figure.colorbar(axes[0, 0].images[0], ax=axes[0, 0])

        # Mask histogram
        axes[0, 1].hist(mask_np.flatten(), bins=100, range=(0, 3),
                        density=True, alpha=0.7)
        axes[0, 1].set_title("Mask value histogram")
        axes[0, 1].set_xlabel("Mask value")
        axes[0, 1].axvline(1.0, color="r", linestyle="--", label="unity")
        axes[0, 1].legend()

        # Temporal smoothness per frame
        td_mean = temporal_diff.mean(axis=0)
        axes[1, 0].plot(td_mean, linewidth=0.5)
        axes[1, 0].set_title("Mean temporal roughness per frame")
        axes[1, 0].set_xlabel("Frame")
        axes[1, 0].set_ylabel("|mask[t] - mask[t-1]| avg")

        # Waveform comparison
        t_axis = np.arange(len(clean_np)) / SAMPLE_RATE
        mix_np = mix_wav.cpu().numpy().squeeze(0)
        axes[1, 1].plot(t_axis, mix_np[:len(clean_np)], alpha=0.3,
                        label="mixture", linewidth=0.5)
        axes[1, 1].plot(t_axis, clean_np, alpha=0.7,
                        label="output", linewidth=0.5)
        axes[1, 1].axhline(1.0, color="r", linestyle="--", alpha=0.5)
        axes[1, 1].axhline(-1.0, color="r", linestyle="--", alpha=0.5)
        axes[1, 1].set_title("Waveform (mixture vs output)")
        axes[1, 1].legend()

        plt.tight_layout()
        fig.savefig(str(out_dir / "mask_diagnosis.png"), dpi=150)
        plt.close(fig)
        print(f"  Saved mask_diagnosis.png")
    except ImportError:
        print("  matplotlib not available, skipping visualization")

    return stats


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_mono(path: Path) -> torch.Tensor:
    data, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if sr != SAMPLE_RATE:
        raise ValueError(
            f"{path} has sample_rate={sr}, expected {SAMPLE_RATE}."
        )
    if data.ndim == 2:
        data = data.mean(axis=1)
    return torch.from_numpy(data)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnose electronic noise artifacts in model output",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="Trained checkpoint (.pt)")
    parser.add_argument("--mixture", type=Path, default=None,
                        help="Mixture wav for mask analysis")
    parser.add_argument("--enrollment", type=Path, default=None,
                        help="Enrollment wav for mask analysis")
    parser.add_argument("--out-dir", type=Path, required=True,
                        help="Directory for diagnostic outputs")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    all_results: dict[str, Any] = {}

    # Check 1: Identity reconstruction
    recon_results = check_identity_reconstruction(
        args.out_dir, checkpoint=args.checkpoint, device=args.device,
    )
    all_results["identity_reconstruction"] = recon_results

    # Check 2: Mask distribution (needs checkpoint + audio)
    if args.checkpoint and args.mixture and args.enrollment:
        mask_results = check_mask_distribution(
            args.checkpoint, args.mixture, args.enrollment,
            args.out_dir, device=args.device,
        )
        all_results["mask_distribution"] = mask_results
    elif args.checkpoint:
        print("\n⚠ Mask analysis skipped: --mixture and --enrollment required")
    else:
        print("\n⚠ Mask analysis skipped: --checkpoint required")

    # Save full report
    with open(args.out_dir / "diagnosis_report.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nFull report saved to {args.out_dir / 'diagnosis_report.json'}")


if __name__ == "__main__":
    main()
