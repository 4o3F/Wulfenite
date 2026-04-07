"""Sanity-check the Phase 0a checkpoint on a freshly sampled AISHELL-1 mixture.

If the reported val SI-SDR-loss (~ -9.4 dB) is real, running the checkpoint on
an in-domain (AISHELL-1 dev) 2-speaker mixture should produce audibly clean
target-speaker output, and the measured SI-SDR of the output vs the clean
target should land near -loss. If the output sounds bad OR the measured SI-SDR
is much lower than -9 dB, something is wrong with the checkpoint loading or
the inference path.

Usage:
    uv run python diagnose_phase0a.py \
        --aishell-root /path/to/aishell \
        --fuse-ckpt ../assets/campplus/train_phase0a/best.pt \
        --out-dir /tmp/phase0a_diag

Outputs three wavs so you can listen:
    /tmp/phase0a_diag/mixture.wav        <- the synthetic mixture (what goes in)
    /tmp/phase0a_diag/target_clean.wav   <- what we want out
    /tmp/phase0a_diag/estimate.wav       <- what the model produces
And prints the measured SI-SDR of estimate vs target_clean.
"""

from __future__ import annotations

import argparse
import math
import pathlib
import platform
from pathlib import Path

import soundfile as sf
import torch

from aishell_mixer import AishellMixDataset
from bsrnn_campplus import build_bsrnn_campplus


# Cross-platform checkpoint compatibility shim — see infer_fuse_debug.py.
if platform.system() == "Windows":
    pathlib.PosixPath = pathlib.WindowsPath  # type: ignore[misc, assignment]


def si_sdr_db(estimate: torch.Tensor, target: torch.Tensor,
              eps: float = 1e-8) -> float:
    estimate = estimate - estimate.mean()
    target = target - target.mean()
    scale = (estimate * target).sum() / ((target * target).sum() + eps)
    projection = scale * target
    noise = estimate - projection
    ratio = (projection * projection).sum() / ((noise * noise).sum() + eps)
    return float(10.0 * math.log10(float(ratio) + eps))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aishell-root", type=Path, required=True)
    parser.add_argument("--fuse-ckpt", type=Path, required=True)
    parser.add_argument("--bsrnn-ckpt", type=Path,
                        default=Path(__file__).resolve().parent.parent
                        / "assets" / "avg_model.pt")
    parser.add_argument("--campplus-ckpt", type=Path,
                        default=Path(__file__).resolve().parent.parent
                        / "assets" / "campplus" / "campplus_cn_common.bin")
    parser.add_argument("--out-dir", type=Path, default=Path("/tmp/phase0a_diag"))
    parser.add_argument("--num-samples", type=int, default=5,
                        help="How many diagnostic samples to produce")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    model = build_bsrnn_campplus(args.bsrnn_ckpt, args.campplus_ckpt, device=device)

    ckpt = torch.load(args.fuse_ckpt, map_location=device, weights_only=False)
    if "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"], strict=True)
        print(f"[load] state_dict from {args.fuse_ckpt} (epoch={ckpt.get('epoch')})")
    else:
        raise KeyError("ckpt missing state_dict")
    model.eval()

    # Inspect fuse layer weights — a degenerate/near-zero matrix would be a red flag.
    fuse_w = model.separator.separation[0].fc.linear.weight
    fuse_b = model.separator.separation[0].fc.linear.bias
    print(
        f"[fuse] weight shape={tuple(fuse_w.shape)} "
        f"norm={fuse_w.norm().item():.3f} "
        f"mean_abs={fuse_w.abs().mean().item():.4f} "
        f"std={fuse_w.std().item():.4f}"
    )
    if fuse_b is not None:
        print(f"[fuse] bias norm={fuse_b.norm().item():.4f}")

    ds = AishellMixDataset(
        aishell_root=args.aishell_root,
        split="dev",
        samples_per_epoch=args.num_samples,
        segment_seconds=4.0,
        enrollment_seconds=4.0,
        snr_range_db=(-2.0, 2.0),  # moderate SNR for clearer A/B
        seed=42,
    )

    sdr_values = []
    with torch.no_grad():
        for i in range(args.num_samples):
            sample = ds[i]
            mixture = sample["mixture"].unsqueeze(0).to(device)
            target = sample["target"].unsqueeze(0).to(device)
            enrollment = sample["enrollment"].unsqueeze(0).to(device)

            estimate = model(mixture, enrollment)
            sdr = si_sdr_db(estimate[0].cpu(), target[0].cpu())
            sdr_mix = si_sdr_db(mixture[0].cpu(), target[0].cpu())
            improvement = sdr - sdr_mix
            print(
                f"[sample {i}] input SI-SDR={sdr_mix:+.2f} dB  "
                f"output SI-SDR={sdr:+.2f} dB  improvement={improvement:+.2f} dB"
            )
            sdr_values.append((sdr_mix, sdr, improvement))

            sf.write(args.out_dir / f"sample{i}_mixture.wav",
                     mixture[0].cpu().numpy(), 16000)
            sf.write(args.out_dir / f"sample{i}_target_clean.wav",
                     target[0].cpu().numpy(), 16000)
            sf.write(args.out_dir / f"sample{i}_estimate.wav",
                     estimate[0].cpu().numpy(), 16000)

    avg_input = sum(s[0] for s in sdr_values) / len(sdr_values)
    avg_output = sum(s[1] for s in sdr_values) / len(sdr_values)
    avg_impr = sum(s[2] for s in sdr_values) / len(sdr_values)
    print(
        f"\n[summary] avg input SI-SDR={avg_input:+.2f} dB  "
        f"avg output SI-SDR={avg_output:+.2f} dB  "
        f"avg improvement={avg_impr:+.2f} dB"
    )
    print(
        "\nExpected: improvement >= +7 dB if the Phase 0a checkpoint is "
        "legitimately separating on AISHELL-1 (matching val SI-SDR-loss ~-9.4 dB)."
    )
    print(f"\nListen to files in: {args.out_dir}")


if __name__ == "__main__":
    main()
