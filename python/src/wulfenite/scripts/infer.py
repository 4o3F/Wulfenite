"""Run batch or streaming enhancement from a TOML configuration file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tomllib
from typing import Any

import soundfile as sf
import torch
from tqdm import tqdm

from wulfenite.evaluation import si_sdr
from wulfenite.inference import Enhancer
from wulfenite.models import DfNet, PDfNet2, SpeakerEncoder


ConfigDict = dict[str, Any]


def _load_config(config_path: str, overrides: list[str]) -> ConfigDict:
    with open(config_path, "rb") as handle:
        cfg = tomllib.load(handle)
    for override in overrides:
        key, _, value = override.partition("=")
        parts = key.strip().split(".")
        target = cfg
        for part in parts[:-1]:
            nested = target.setdefault(part, {})
            if not isinstance(nested, dict):
                raise ValueError(f"Override target {'.'.join(parts[:-1])} is not a table.")
            target = nested
        raw_value = value.strip()
        if raw_value.lower() in ("true", "false"):
            target[parts[-1]] = raw_value.lower() == "true"
        else:
            try:
                target[parts[-1]] = int(raw_value)
            except ValueError:
                try:
                    target[parts[-1]] = float(raw_value)
                except ValueError:
                    target[parts[-1]] = raw_value
    return cfg


def _config_table(config: ConfigDict, key: str) -> ConfigDict:
    table = config.get(key)
    if not isinstance(table, dict):
        raise ValueError(f"Missing required [{key}] table.")
    return table


def _optional_path(value: object) -> Path | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"Expected a path-like string, got {type(value)!r}.")
    if not value:
        return None
    return Path(value).expanduser()


def _required_path(table: ConfigDict, key: str) -> Path:
    path = _optional_path(table.get(key))
    if path is None:
        raise ValueError(f"Missing required path: {key}")
    return path


def _load_model(checkpoint_path: str | Path) -> tuple[DfNet | PDfNet2, bool]:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state = ckpt.get("model_state_dict")
    if not isinstance(state, dict):
        raise ValueError(f"Checkpoint at {checkpoint_path} does not contain model_state_dict.")
    try:
        model = PDfNet2()
        model.load_state_dict(state, strict=True)
        return model, True
    except RuntimeError:
        model = DfNet()
        model.load_state_dict(state, strict=True)
        return model, False


def _read_wav(path: Path) -> tuple[torch.Tensor, int]:
    audio, sample_rate = sf.read(str(path), dtype="float32", always_2d=False)
    if getattr(audio, "ndim", 1) != 1:
        raise RuntimeError(f"Expected mono audio at {path}")
    waveform = torch.from_numpy(audio).unsqueeze(0)
    return waveform, sample_rate


def _list_input_files(path: Path) -> tuple[list[Path], Path]:
    if path.is_file():
        return [path], path.parent
    if not path.is_dir():
        raise RuntimeError(f"Input path does not exist: {path}")
    files = sorted(path.rglob("*.wav"))
    if not files:
        raise RuntimeError(f"No .wav files found under {path}")
    return files, path


def _resolve_output_path(
    output_path: Path,
    input_file: Path,
    input_root: Path,
    *,
    single_input: bool,
) -> Path:
    if output_path.suffix.lower() == ".wav":
        if not single_input:
            raise RuntimeError("A file output path can only be used with a single input wav.")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        return output_path
    relative = Path(input_file.name) if single_input else input_file.relative_to(input_root)
    destination = output_path / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    return destination


def _resolve_reference_path(
    reference_path: Path,
    input_file: Path,
    input_root: Path,
    *,
    single_input: bool,
) -> Path:
    if reference_path.is_file():
        return reference_path
    if not reference_path.is_dir():
        raise RuntimeError(f"Reference path does not exist: {reference_path}")
    relative = Path(input_file.name) if single_input else input_file.relative_to(input_root)
    candidate = reference_path / relative
    if not candidate.exists():
        raise RuntimeError(f"Reference wav not found for {input_file}: {candidate}")
    return candidate


def _run_streaming(enhancer: Enhancer, waveform: torch.Tensor, chunk_samples: int) -> torch.Tensor:
    chunks: list[torch.Tensor] = []
    total = waveform.size(-1)
    for start in tqdm(range(0, total, chunk_samples), desc="streaming", leave=False, dynamic_ncols=True):
        end = min(total, start + chunk_samples)
        chunks.append(
            enhancer.enhance_streaming(
                waveform[:, start:end],
                finalize=end == total,
            )
        )
    if not chunks:
        return waveform.new_zeros(waveform.size(0), 0)
    return torch.cat(chunks, dim=-1)


def _print_config(cfg: ConfigDict) -> None:
    print("[config] resolved configuration:")
    print(json.dumps(cfg, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to an inference TOML file.")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a TOML value, for example streaming.enabled=true",
    )
    args = parser.parse_args()

    cfg = _load_config(args.config, args.override)
    _print_config(cfg)

    model_cfg = _config_table(cfg, "model")
    input_cfg = _config_table(cfg, "input")
    output_cfg = _config_table(cfg, "output")
    streaming_cfg = _config_table(cfg, "streaming")
    eval_cfg = _config_table(cfg, "eval")
    runtime_cfg = _config_table(cfg, "runtime")

    checkpoint_path = _required_path(model_cfg, "checkpoint")
    model, needs_speaker = _load_model(checkpoint_path)

    device = runtime_cfg.get("device", "auto")
    if not isinstance(device, str):
        raise ValueError("[runtime].device must be a string.")
    resolved_device = None if device == "auto" else device

    enrollment_path = _optional_path(input_cfg.get("enrollment"))
    speaker_encoder: SpeakerEncoder | None = None
    if needs_speaker:
        if enrollment_path is None:
            raise SystemExit("This checkpoint requires [input].enrollment for speaker conditioning.")
        speaker_encoder = SpeakerEncoder(
            checkpoint_path=_optional_path(model_cfg.get("wespeaker_checkpoint"))
        )

    enhancer = Enhancer(model, enrollment_encoder=speaker_encoder, device=resolved_device)
    if speaker_encoder is not None and enrollment_path is not None:
        enrollment_waveform, enrollment_sr = _read_wav(enrollment_path)
        if enrollment_sr != enhancer._core_model.sample_rate:
            raise RuntimeError(
                f"Enrollment sample rate {enrollment_sr} does not match model sample rate "
                f"{enhancer._core_model.sample_rate}."
            )
        enhancer.enroll(enrollment_waveform)

    input_path = _required_path(input_cfg, "path")
    output_path = _required_path(output_cfg, "path")
    input_files, input_root = _list_input_files(input_path)
    single_input = len(input_files) == 1 and input_path.is_file()

    streaming_enabled = bool(streaming_cfg.get("enabled", False))
    chunk_ms = float(streaming_cfg.get("chunk_ms", 20))
    chunk_samples = max(1, int(round(enhancer._core_model.sample_rate * chunk_ms / 1000.0)))

    evaluate = bool(eval_cfg.get("enabled", False))
    reference_root = _optional_path(eval_cfg.get("reference"))
    if evaluate and reference_root is None:
        raise SystemExit("[eval].enabled=true requires [eval].reference.")

    scores: list[float] = []
    for input_file in tqdm(input_files, desc="enhancing", dynamic_ncols=True):
        waveform, sample_rate = _read_wav(input_file)
        if sample_rate != enhancer._core_model.sample_rate:
            raise RuntimeError(
                f"Input sample rate {sample_rate} does not match model sample rate "
                f"{enhancer._core_model.sample_rate}: {input_file}"
            )
        if streaming_enabled:
            enhanced = _run_streaming(enhancer, waveform, chunk_samples)
        else:
            enhanced = enhancer.enhance(waveform)

        output_file = _resolve_output_path(
            output_path,
            input_file,
            input_root,
            single_input=single_input,
        )
        sf.write(str(output_file), enhanced.squeeze(0).detach().cpu().numpy(), sample_rate)
        print(f"[write] {output_file}")

        if evaluate and reference_root is not None:
            reference_file = _resolve_reference_path(
                reference_root,
                input_file,
                input_root,
                single_input=single_input,
            )
            reference_waveform, reference_sr = _read_wav(reference_file)
            if reference_sr != sample_rate:
                raise RuntimeError(
                    f"Reference sample rate {reference_sr} does not match input sample rate "
                    f"{sample_rate}: {reference_file}"
                )
            min_length = min(reference_waveform.size(-1), enhanced.size(-1))
            score = float(
                si_sdr(
                    enhanced[:, :min_length].detach().cpu(),
                    reference_waveform[:, :min_length],
                ).mean().item()
            )
            scores.append(score)
            print(f"[eval] {input_file.name} si_sdr={score:.3f} dB")

    if scores:
        mean_score = sum(scores) / len(scores)
        print(f"[eval] mean si_sdr={mean_score:.3f} dB over {len(scores)} file(s)")


if __name__ == "__main__":
    main()
