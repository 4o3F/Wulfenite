"""Streaming inference CLI.

Runs the trained Wulfenite model frame-by-frame using
:meth:`SpeakerBeamSS.streaming_step`, simulating the exact
control flow the Rust runtime will use. Serves two purposes:

1. **Quality validation** — streaming output should match
   whole-utterance output to within floating-point rounding, verifying
   the stateful implementation is numerically equivalent to the
   training path. If they differ, something is wrong.
2. **Latency measurement** — reports per-chunk compute time and
   aggregate RTF, so you can verify real-time feasibility on the
   target CPU before porting to Rust.

Usage:

    uv run --directory python python -m wulfenite.inference.streaming \\
        --checkpoint ./checkpoints/phase3_paper_magicdata/best.pt \\
        --mixture ./samples/real_mixture.wav \\
        --enrollment ./samples/real_enrollment.wav \\
        --output ./output_stream.wav \\
        --chunk-ms 20
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
from tqdm.auto import tqdm

from .utils import build_model_from_checkpoint


if platform.system() == "Windows":
    pathlib.PosixPath = pathlib.WindowsPath  # type: ignore[misc, assignment]


SAMPLE_RATE = 16000


def _load_mono(path: Path) -> torch.Tensor:
    data, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if sr != SAMPLE_RATE:
        raise ValueError(
            f"{path} has sample_rate={sr}, expected {SAMPLE_RATE}."
        )
    if data.ndim == 2:
        data = data.mean(axis=1)
    return torch.from_numpy(data)


def run_streaming(
    checkpoint: Path,
    mixture: Path,
    enrollment: Path,
    output: Path,
    chunk_ms: float = 20.0,
    device: str = "cpu",
) -> dict:
    """Run streaming inference and write the clean wav.

    Returns a dict of metrics including per-chunk latency stats,
    aggregate RTF, and realtime feasibility verdict.
    """
    dev = torch.device(device)

    model, info = build_model_from_checkpoint(
        checkpoint,
        device=dev,
    )
    print(
        f"[load] checkpoint {checkpoint} epoch={info.get('epoch')}"
    )

    separator = model.separator
    stride = separator.config.enc_stride

    chunk_size = int(chunk_ms * SAMPLE_RATE / 1000)
    if chunk_size % stride != 0 or chunk_size <= 0:
        raise ValueError(
            f"--chunk-ms {chunk_ms} → {chunk_size} samples, which must be a "
            f"positive multiple of enc_stride={stride} "
            f"({stride * 1000 / SAMPLE_RATE:.0f} ms). Try 10, 20, 40, 80 ms."
        )

    mix_wav = _load_mono(mixture)
    enr_wav = _load_mono(enrollment).unsqueeze(0).to(dev)

    # --- Encode enrollment once ---
    with torch.no_grad():
        speaker_embedding = model.encode_enrollment(enr_wav[0])

    # --- Pad mixture so its length is a multiple of chunk_size ---
    pad_len = (chunk_size - mix_wav.shape[0] % chunk_size) % chunk_size
    if pad_len:
        mix_wav = torch.nn.functional.pad(mix_wav, (0, pad_len))
    mix_wav = mix_wav.unsqueeze(0).to(dev)  # [1, T_padded]

    # --- Streaming loop ---
    state = separator.initial_streaming_state(batch_size=1, device=dev)
    n_chunks = mix_wav.shape[-1] // chunk_size
    latencies_ms: list[float] = []
    clean_pieces = []

    with torch.no_grad(), tqdm(
        total=n_chunks, desc="streaming", unit="chunk", dynamic_ncols=True,
    ) as pbar:
        for i in range(n_chunks):
            start = i * chunk_size
            chunk = mix_wav[..., start:start + chunk_size]
            t0 = time.perf_counter()
            clean_chunk, state = separator.streaming_step(
                chunk, speaker_embedding, state,
            )
            latency = (time.perf_counter() - t0) * 1000.0
            latencies_ms.append(latency)
            clean_pieces.append(clean_chunk.cpu())
            pbar.update(1)
            if i % 10 == 0:
                pbar.set_postfix(
                    ms=f"{latency:.1f}",
                    refresh=False,
                )

    clean = torch.cat(clean_pieces, dim=-1)[0].numpy()
    # Trim any tail padding so output length matches original mixture.
    original_len = clean.shape[0] - pad_len
    clean = clean[:original_len]

    output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output), clean, SAMPLE_RATE)

    # --- Metrics ---
    lat = torch.tensor(latencies_ms)
    audio_seconds = original_len / SAMPLE_RATE
    total_compute_ms = float(lat.sum())
    hop_ms = chunk_size * 1000.0 / SAMPLE_RATE

    metrics = {
        "audio_seconds": audio_seconds,
        "n_chunks": n_chunks,
        "chunk_ms": hop_ms,
        "mean_chunk_ms": float(lat.mean()),
        "p50_chunk_ms": float(lat.median()),
        "p95_chunk_ms": float(
            lat.kthvalue(max(1, int(0.95 * len(lat)))).values
        ) if len(lat) > 1 else float(lat[0]),
        "max_chunk_ms": float(lat.max()),
        "rtf": total_compute_ms / (audio_seconds * 1000.0) if audio_seconds > 0 else 0.0,
        "output": str(output),
    }

    print()
    print(f"[output] {output}")
    print(f"[output] {audio_seconds:.2f} s, {original_len} samples")
    print()
    print("==== streaming timing ====")
    print(f"  n_chunks          : {n_chunks}")
    print(f"  chunk / hop       : {hop_ms:.0f} ms")
    print(f"  per-chunk compute :")
    print(f"    mean            : {metrics['mean_chunk_ms']:.2f} ms")
    print(f"    p50             : {metrics['p50_chunk_ms']:.2f} ms")
    print(f"    p95             : {metrics['p95_chunk_ms']:.2f} ms")
    print(f"    max             : {metrics['max_chunk_ms']:.2f} ms")
    print(f"  aggregate RTF     : {metrics['rtf']:.3f}")
    if metrics["p95_chunk_ms"] > hop_ms:
        print(
            f"  ⚠  p95 chunk compute ({metrics['p95_chunk_ms']:.1f} ms) "
            f"exceeds hop window ({hop_ms:.0f} ms) — real-time would stutter."
        )
        metrics["realtime_ok"] = False
    else:
        headroom = hop_ms - metrics["p95_chunk_ms"]
        print(
            f"  ✓  p95 chunk compute fits in the hop window with "
            f"{headroom:.1f} ms headroom."
        )
        metrics["realtime_ok"] = True

    return metrics


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--mixture", type=Path, required=True)
    parser.add_argument("--enrollment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunk-ms", type=float, default=20.0,
                        help="Chunk size in milliseconds. Must be a positive "
                             "multiple of 10 ms (enc_stride).")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_streaming(
        checkpoint=args.checkpoint,
        mixture=args.mixture,
        enrollment=args.enrollment,
        output=args.output,
        chunk_ms=args.chunk_ms,
        device=args.device,
    )


if __name__ == "__main__":
    main()
