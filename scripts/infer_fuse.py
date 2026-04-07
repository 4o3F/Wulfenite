"""Inference with the BSRNN+CAM++ wrapper.

Mirrors the CLI of ``assets/wesep_bsrnn_ecapa_pytorch_only.py`` so you can
A/B listen to the English baseline vs the Chinese fuse-layer-fine-tuned
model on the same mixture + enrollment pair.

Usage:
    uv run python infer_fuse.py \
        --bsrnn-ckpt ../assets/avg_model.pt \
        --campplus-ckpt ../assets/campplus/campplus_cn_common.bin \
        --fuse-ckpt ../assets/campplus/train_phase0a/best.pt \
        --mixture ../assets/mixture2.wav \
        --enrollment ../assets/enrollment2.wav \
        --output /tmp/bsrnn_campplus_out.wav

If ``--fuse-ckpt`` is omitted, the wrapper runs with a fresh random fuse
layer — use that as a sanity check to verify the pipeline at least does not
explode (the output will sound bad because the fuse layer is untrained).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import soundfile as sf
import torch

from bsrnn_campplus import build_bsrnn_campplus


def _load_mono(path: Path, sample_rate: int = 16000) -> torch.Tensor:
    data, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if sr != sample_rate:
        raise ValueError(
            f"{path} has sample_rate={sr}, expected {sample_rate}. "
            "Resample it first (e.g. with sox)."
        )
    if data.ndim == 2:
        data = data.mean(axis=1)
    return torch.from_numpy(data)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bsrnn-ckpt", type=Path, required=True)
    parser.add_argument("--campplus-ckpt", type=Path, required=True)
    parser.add_argument("--fuse-ckpt", type=Path, default=None,
                        help="Optional fine-tuned checkpoint from train_fuse.py")
    parser.add_argument("--mixture", type=Path, required=True)
    parser.add_argument("--enrollment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"[init] device={device}")

    model = build_bsrnn_campplus(args.bsrnn_ckpt, args.campplus_ckpt, device=device)

    if args.fuse_ckpt is not None:
        ckpt = torch.load(args.fuse_ckpt, map_location=device, weights_only=False)
        # Prefer the full state_dict; fall back to fuse_state_dict.
        if "state_dict" in ckpt:
            model.load_state_dict(ckpt["state_dict"], strict=True)
            print(f"[load] full state_dict from {args.fuse_ckpt} (epoch={ckpt.get('epoch')})")
        elif "fuse_state_dict" in ckpt:
            model.separator.separation[0].fc.linear.load_state_dict(
                ckpt["fuse_state_dict"], strict=True
            )
            print(f"[load] fuse-only state_dict from {args.fuse_ckpt}")
        else:
            raise KeyError("fuse ckpt missing both 'state_dict' and 'fuse_state_dict'")

    model.eval()

    mixture = _load_mono(args.mixture).unsqueeze(0).to(device)  # [1, T]
    enrollment = _load_mono(args.enrollment).unsqueeze(0).to(device)  # [1, T]

    t0 = time.time()
    with torch.no_grad():
        estimate = model(mixture, enrollment)
    elapsed = time.time() - t0
    duration = mixture.shape[-1] / 16000.0
    print(
        f"[infer] audio={duration:.2f}s wall={elapsed:.2f}s RTF={elapsed / duration:.3f}"
    )

    import math

    out = estimate[0].cpu().numpy()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(args.output), out, 16000)
    peak = float(abs(out).max())
    rms = float((out * out).mean() ** 0.5)
    peak_dbfs = 20 * math.log10(max(peak, 1e-9))
    print(
        f"[write] {args.output} samples={out.shape[0]} peak={peak:.3f} "
        f"rms={rms:.4f} peak_dbfs={peak_dbfs:.1f}"
    )


if __name__ == "__main__":
    main()
