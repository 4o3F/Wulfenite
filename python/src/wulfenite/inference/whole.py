"""Whole-utterance inference CLI.

Loads a trained Wulfenite checkpoint and runs the full separator
forward pass on a single (mixture, enrollment) pair, writing the
clean output to a wav file.

Usage:

    uv run --directory python python -m wulfenite.inference.whole \\
        --checkpoint ./checkpoints/phase5b_cnceleb/best.pt \\
        --mixture ./samples/real_mixture.wav \\
        --enrollment ./samples/real_enrollment.wav \\
        --output ./output.wav

For legacy frozen-CAM++ checkpoints also pass::

    --campplus-checkpoint ~/datasets/campplus/campplus_cn_common.bin

This is the quality-reference inference path. The streaming CLI in
``wulfenite.inference.streaming`` should produce numerically
identical output thanks to the stateful equivalence guarantee
verified by tests/test_speakerbeam_ss.py.
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

from .utils import build_model_from_checkpoint


# ---------------------------------------------------------------------------
# Cross-platform checkpoint compatibility shim
# ---------------------------------------------------------------------------
# Checkpoints saved on Linux may contain ``pathlib.PosixPath`` objects
# inside the pickled ``config`` dict. Aliasing the class here lets them
# unpickle on Windows as well. v2 save_checkpoint stringifies Paths so
# this is a safety net for checkpoints from other toolchains.
if platform.system() == "Windows":
    pathlib.PosixPath = pathlib.WindowsPath  # type: ignore[misc, assignment]


SAMPLE_RATE = 16000


def _load_mono(path: Path) -> torch.Tensor:
    data, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if sr != SAMPLE_RATE:
        raise ValueError(
            f"{path} has sample_rate={sr}, expected {SAMPLE_RATE}. "
            "Resample to 16 kHz mono first (e.g., with sox)."
        )
    if data.ndim == 2:
        data = data.mean(axis=1)
    return torch.from_numpy(data)


def run_whole(
    checkpoint: Path,
    mixture: Path,
    enrollment: Path,
    output: Path,
    campplus_checkpoint: Path | None = None,
    device: str = "cpu",
    use_learnable_encoder: bool | None = None,
) -> dict:
    """Run one whole-utterance inference and write the clean wav.

    Returns a dict of metrics (rtf, peak dBFS, rms, ...) for
    programmatic use by callers / tests.
    """
    dev = torch.device(device)

    model, info = build_model_from_checkpoint(
        checkpoint,
        campplus_checkpoint=campplus_checkpoint,
        use_learnable_encoder=use_learnable_encoder,
        device=dev,
    )
    print(
        f"[load] checkpoint {checkpoint} epoch={info.get('epoch')} "
        f"val_loss={info.get('metrics', {}).get('val_loss', '?')}"
    )

    mix_wav = _load_mono(mixture).unsqueeze(0).to(dev)          # [1, T]
    enr_wav = _load_mono(enrollment).unsqueeze(0).to(dev)       # [1, T_enr]

    t0 = time.perf_counter()
    with torch.no_grad():
        outputs = model(mix_wav, enr_wav)
    elapsed = time.perf_counter() - t0

    clean = outputs["clean"][0].cpu().numpy()
    output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output), clean, SAMPLE_RATE)

    duration = mix_wav.shape[-1] / SAMPLE_RATE
    peak = float(abs(clean).max())
    rms = float((clean ** 2).mean() ** 0.5)
    metrics = {
        "audio_seconds": duration,
        "compute_seconds": elapsed,
        "rtf": elapsed / duration if duration > 0 else 0.0,
        "peak": peak,
        "peak_dbfs": 20.0 * math.log10(max(peak, 1e-9)),
        "rms": rms,
        "output": str(output),
    }
    if "presence_logit" in outputs:
        logit = float(outputs["presence_logit"][0].cpu())
        metrics["presence_prob"] = 1.0 / (1.0 + math.exp(-logit))

    print(
        f"[infer] {duration:.2f} s audio in {elapsed * 1000:.0f} ms "
        f"(RTF={metrics['rtf']:.3f}) peak={metrics['peak_dbfs']:.1f} dBFS"
    )
    if "presence_prob" in metrics:
        print(f"[infer] target_present probability = {metrics['presence_prob']:.3f}")
    print(f"[output] {output}")
    return metrics


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Trained Wulfenite training checkpoint (.pt)")
    parser.add_argument("--campplus-checkpoint", type=Path, default=None,
                        help="Frozen CAM++ zh-cn checkpoint (.bin). "
                             "Required for legacy frozen-CAM++ checkpoints.")
    parser.add_argument("--use-learnable-encoder", action="store_true", default=None,
                        help="Force the learnable-encoder inference path. "
                             "Normally auto-detected from the checkpoint config.")
    parser.add_argument("--mixture", type=Path, required=True)
    parser.add_argument("--enrollment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_whole(
        checkpoint=args.checkpoint,
        campplus_checkpoint=args.campplus_checkpoint,
        mixture=args.mixture,
        enrollment=args.enrollment,
        output=args.output,
        device=args.device,
        use_learnable_encoder=args.use_learnable_encoder,
    )


if __name__ == "__main__":
    main()
