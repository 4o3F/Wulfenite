"""Low-latency streaming inference with sliding-window + linear-crossfade.

Simulates exactly the control flow the Rust deployment (``src/streaming``)
will use, so you can verify chunk-boundary artifacts and RTF *before*
touching the Rust side. Uses the same BSRNN+CAM++ model as the
whole-utterance scripts, no retraining or model changes needed.

Key design choices for a BiLSTM separator:

- **Per-chunk forward pass**: each chunk is processed as a complete
  sequence. The BiLSTM sees the full chunk's future context. This sets
  an algorithmic latency floor of ``chunk_ms``.
- **Short chunks (default 500 ms)** + modest overlap (default 150 ms)
  give a 500 ms algorithmic latency with ~35 % recompute overhead.
- **Linear crossfade in the overlap region** smooths boundary artifacts
  without requiring a perfect-reconstruction window (Hann with 50%
  overlap would be cleaner but wastes more compute).
- **Enrollment processed once**: CAM++ runs on the enrollment a single
  time at startup. All chunks reuse the cached 192-dim conditioning
  signal via ``model.separate_with_cond``. Without this, CAM++ FBank +
  forward would dominate per-chunk cost at 500 ms chunks.

Usage:

    uv run python infer_fuse_streaming.py \\
        --fuse-ckpt ../assets/campplus/train_phase0b/best.pt \\
        --mixture ../assets/mixture3.wav \\
        --enrollment ../assets/enrollment3.wav \\
        --output ../assets/streaming_out.wav \\
        --chunk-ms 500 --overlap-ms 150 --device cpu

Output wav is head-aligned with the input. Timing report at end:
average per-chunk latency, p50/p95/p99 latency, aggregate RTF, and the
theoretical algorithmic latency floor.
"""

from __future__ import annotations

import argparse
import math
import pathlib
import platform
import time
from pathlib import Path

import soundfile as sf
import torch

from bsrnn_campplus import build_bsrnn_campplus


# Cross-platform checkpoint compatibility shim — see infer_fuse_debug.py.
if platform.system() == "Windows":
    pathlib.PosixPath = pathlib.WindowsPath  # type: ignore[misc, assignment]


SAMPLE_RATE = 16000


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------


def _load_mono(path: Path, sr: int = SAMPLE_RATE) -> torch.Tensor:
    data, file_sr = sf.read(str(path), dtype="float32", always_2d=False)
    if file_sr != sr:
        raise ValueError(f"{path}: sample_rate={file_sr}, expected {sr}")
    if data.ndim == 2:
        data = data.mean(axis=1)
    return torch.from_numpy(data)


# ---------------------------------------------------------------------------
# Streaming loop
# ---------------------------------------------------------------------------


def run_streaming(
    model,
    mixture: torch.Tensor,
    enrollment: torch.Tensor,
    chunk_size: int,
    overlap: int,
    device: torch.device,
) -> tuple[torch.Tensor, dict]:
    """Process ``mixture`` chunk-by-chunk and crossfade the outputs.

    Returns the full output waveform (head-aligned, same length as input)
    plus a dict of timing statistics.
    """
    if overlap <= 0 or overlap >= chunk_size:
        raise ValueError(
            f"overlap ({overlap}) must be in (0, chunk_size={chunk_size})"
        )

    hop = chunk_size - overlap  # samples between consecutive chunk starts
    total_samples = mixture.shape[0]

    # 1. Pre-compute the speaker conditioning signal once.
    with torch.no_grad():
        enrollment_batched = enrollment.unsqueeze(0).to(device)  # [1, T_enr]
        cond = model.compute_cond(enrollment_batched)  # [1, 1, 192, 1]

    # 2. Pad mixture at the tail so the last chunk is full length.
    # We emit exactly ``total_samples`` samples at the end by trimming.
    #   number of chunks = ceil((T - overlap) / hop)
    n_chunks = max(1, math.ceil((total_samples - overlap) / hop))
    padded_length = overlap + n_chunks * hop
    if padded_length < chunk_size:
        padded_length = chunk_size
        n_chunks = 1
    pad_len = max(0, padded_length - total_samples)
    if pad_len > 0:
        mixture_padded = torch.nn.functional.pad(mixture, (0, pad_len))
    else:
        mixture_padded = mixture
    output = torch.zeros(mixture_padded.shape[0], dtype=torch.float32)

    # Linear crossfade ramps for the overlap region.
    ramp_up = torch.linspace(0.0, 1.0, steps=overlap)
    ramp_down = 1.0 - ramp_up

    latencies_ms: list[float] = []

    with torch.no_grad():
        for i in range(n_chunks):
            start = i * hop
            end = start + chunk_size
            if end > mixture_padded.shape[0]:
                # Shouldn't happen after padding, but guard anyway.
                break
            chunk = mixture_padded[start:end].unsqueeze(0).to(device)

            t0 = time.perf_counter()
            est = model.separate_with_cond(chunk, cond)[0].cpu()
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            latencies_ms.append(elapsed_ms)

            if i == 0:
                # First chunk: write everything verbatim.
                output[start:end] = est
            else:
                # Crossfade the overlap region:
                #   new_output[start:start+overlap] =
                #       existing * ramp_down + est[:overlap] * ramp_up
                # The "existing" samples come from the previous chunk's tail
                # which was already written to `output[start:start+overlap]`.
                existing = output[start:start + overlap].clone()
                output[start:start + overlap] = (
                    existing * ramp_down + est[:overlap] * ramp_up
                )
                # Non-overlap region: plain copy.
                output[start + overlap:end] = est[overlap:]

    # 3. Trim trailing pad.
    output = output[:total_samples]

    # 4. Timing summary.
    latencies = torch.tensor(latencies_ms)
    audio_seconds = total_samples / SAMPLE_RATE
    total_compute_ms = float(latencies.sum())
    stats = {
        "n_chunks": n_chunks,
        "chunk_ms": chunk_size * 1000 / SAMPLE_RATE,
        "overlap_ms": overlap * 1000 / SAMPLE_RATE,
        "hop_ms": hop * 1000 / SAMPLE_RATE,
        "per_chunk_mean_ms": float(latencies.mean()),
        "per_chunk_p50_ms": float(latencies.median()),
        "per_chunk_p95_ms": float(latencies.kthvalue(max(1, int(0.95 * len(latencies)))).values) if len(latencies) > 1 else float(latencies[0]),
        "per_chunk_max_ms": float(latencies.max()),
        "total_compute_ms": total_compute_ms,
        "audio_seconds": audio_seconds,
        "rtf": total_compute_ms / (audio_seconds * 1000.0) if audio_seconds > 0 else 0.0,
        "algo_latency_ms": chunk_size * 1000 / SAMPLE_RATE,
    }
    return output, stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bsrnn-ckpt", type=Path,
                        default=Path(__file__).resolve().parent.parent
                        / "assets" / "avg_model.pt")
    parser.add_argument("--campplus-ckpt", type=Path,
                        default=Path(__file__).resolve().parent.parent
                        / "assets" / "campplus" / "campplus_cn_common.bin")
    parser.add_argument("--fuse-ckpt", type=Path, default=None,
                        help="Trained Phase 0a / 0b checkpoint")
    parser.add_argument("--mixture", type=Path, required=True)
    parser.add_argument("--enrollment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)

    parser.add_argument("--chunk-ms", type=float, default=500.0,
                        help="Chunk length in milliseconds. Sets the "
                             "algorithmic latency floor for BiLSTM models.")
    parser.add_argument("--overlap-ms", type=float, default=150.0,
                        help="Overlap between consecutive chunks in ms. "
                             "Must be < chunk-ms. Larger = smoother "
                             "boundaries but more recompute.")

    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--rescale-output-to-input", action="store_true",
                        help="Rescale output RMS to match input RMS at the "
                             "very end. Useful for Phase 0a checkpoints that "
                             "trained with scale-invariant loss; Phase 0b "
                             "should not need this.")

    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"[init] device={device}")

    chunk_size = int(args.chunk_ms * SAMPLE_RATE / 1000)
    overlap = int(args.overlap_ms * SAMPLE_RATE / 1000)
    if overlap >= chunk_size:
        raise SystemExit(
            f"--overlap-ms ({args.overlap_ms}) must be < --chunk-ms ({args.chunk_ms})"
        )
    hop = chunk_size - overlap
    print(
        f"[stream] chunk={chunk_size} samples ({args.chunk_ms:.0f} ms) "
        f"overlap={overlap} samples ({args.overlap_ms:.0f} ms) "
        f"hop={hop} samples ({hop * 1000 / SAMPLE_RATE:.0f} ms)"
    )

    model = build_bsrnn_campplus(args.bsrnn_ckpt, args.campplus_ckpt, device=device)
    if args.fuse_ckpt is not None:
        ckpt = torch.load(args.fuse_ckpt, map_location=device, weights_only=False)
        if "state_dict" in ckpt:
            model.load_state_dict(ckpt["state_dict"], strict=True)
            print(f"[load] state_dict from {args.fuse_ckpt} (epoch={ckpt.get('epoch')})")
        else:
            raise KeyError(f"{args.fuse_ckpt} has no 'state_dict' key")
    model.eval()

    mixture = _load_mono(args.mixture)
    enrollment = _load_mono(args.enrollment)
    print(
        f"[input] mixture {mixture.shape[0]} samples "
        f"({mixture.shape[0] / SAMPLE_RATE:.2f} s) "
        f"RMS={float(torch.sqrt((mixture * mixture).mean() + 1e-12)):.4f}"
    )
    print(
        f"[input] enrollment {enrollment.shape[0]} samples "
        f"({enrollment.shape[0] / SAMPLE_RATE:.2f} s) "
        f"RMS={float(torch.sqrt((enrollment * enrollment).mean() + 1e-12)):.4f}"
    )

    output, stats = run_streaming(
        model, mixture, enrollment, chunk_size, overlap, device,
    )

    if args.rescale_output_to_input:
        in_rms = float(torch.sqrt((mixture * mixture).mean() + 1e-12))
        out_rms = float(torch.sqrt((output * output).mean() + 1e-12))
        if out_rms > 1e-9:
            scale = in_rms / out_rms
            peak_after = float((output * scale).abs().max())
            if peak_after > 0.99:
                scale *= 0.99 / peak_after
            output = output * scale
            print(f"[post ] rescale x{scale:.3f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(args.output), output.numpy(), SAMPLE_RATE)

    peak = float(output.abs().max())
    rms = float(torch.sqrt((output * output).mean() + 1e-12))
    peak_dbfs = 20 * math.log10(max(peak, 1e-9))

    print()
    print(f"[output] {args.output}")
    print(f"[output] peak={peak:.4f} rms={rms:.4f} peak_dBFS={peak_dbfs:.1f}")
    print()
    print("==== streaming timing ====")
    print(f"  audio length       : {stats['audio_seconds']:.2f} s")
    print(f"  chunks processed   : {stats['n_chunks']}")
    print(
        f"  chunk / hop / ovlp : "
        f"{stats['chunk_ms']:.0f} / {stats['hop_ms']:.0f} / {stats['overlap_ms']:.0f} ms"
    )
    print(f"  algo latency floor : {stats['algo_latency_ms']:.0f} ms (chunk length)")
    print()
    print(f"  per-chunk compute  :")
    print(f"    mean             : {stats['per_chunk_mean_ms']:.1f} ms")
    print(f"    p50              : {stats['per_chunk_p50_ms']:.1f} ms")
    print(f"    p95              : {stats['per_chunk_p95_ms']:.1f} ms")
    print(f"    max              : {stats['per_chunk_max_ms']:.1f} ms")
    print()
    print(f"  aggregate RTF      : {stats['rtf']:.3f}")
    print(f"    (< 1.0 means real-time is feasible; lower is better)")
    # Realtime feasibility check: per-chunk compute must fit in the hop window,
    # otherwise the next chunk's audio arrives before the current one finishes.
    hop_ms = stats["hop_ms"]
    if stats["per_chunk_p95_ms"] > hop_ms:
        print(
            f"  ⚠️  p95 chunk compute ({stats['per_chunk_p95_ms']:.1f} ms) "
            f"exceeds hop window ({hop_ms:.0f} ms) — real-time will stutter."
        )
    else:
        headroom = hop_ms - stats["per_chunk_p95_ms"]
        print(
            f"  ✓  p95 chunk compute fits in hop window with "
            f"{headroom:.0f} ms headroom."
        )


if __name__ == "__main__":
    main()
