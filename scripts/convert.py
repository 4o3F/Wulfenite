#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import struct
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import torch

try:
    from safetensors.torch import save_file as save_safetensors_file
except ModuleNotFoundError:
    save_safetensors_file = None


LSTM_GATE_NAMES = ("input_gate", "forget_gate", "cell_gate", "output_gate")

LINEAR_WEIGHT_PATTERNS = (
    re.compile(r"^spk_model\.layer[234]\.se\.linear[12]\.weight$"),
    re.compile(r"^spk_model\.linear\.weight$"),
    re.compile(r"^separator\.speaker_fuse\.fc\.weight$"),
    re.compile(r"^separator\.blocks\.\d+\.(band_rnn|band_comm)\.proj\.weight$"),
    re.compile(
        r"^separator\.blocks\.\d+\.(band_rnn|band_comm)\.rnn\."
        r"(forward|reverse)\.(input_gate|forget_gate|cell_gate|output_gate)\."
        r"(input_transform|hidden_transform)\.weight$"
    ),
)

BATCH_OR_GROUP_NORM_WEIGHT_RES = (
    re.compile(r"(\.(bn|norm))\.weight$"),
    re.compile(r"(\.bns\.\d+)\.weight$"),
)
BATCH_OR_GROUP_NORM_BIAS_RES = (
    re.compile(r"(\.(bn|norm))\.bias$"),
    re.compile(r"(\.bns\.\d+)\.bias$"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert the released WeSep training checkpoint into Burn-targeted safetensors."
    )
    parser.add_argument("input", type=Path, help="Path to the source .pt checkpoint.")
    parser.add_argument(
        "--output",
        type=Path,
        help="Output safetensors path. Defaults to converted/<stem>.burn.safetensors in this repo.",
    )
    return parser.parse_args()


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def default_output_path(input_path: Path) -> Path:
    return repo_root() / "converted" / f"{input_path.stem}.burn.safetensors"


def extract_state_dict(checkpoint: object) -> Mapping[str, torch.Tensor]:
    if isinstance(checkpoint, Mapping) and "models" in checkpoint:
        models = checkpoint["models"]
        if isinstance(models, Sequence) and not isinstance(models, (str, bytes)):
            if not models:
                raise ValueError("checkpoint['models'] is empty")
            state_dict = models[0]
        else:
            raise ValueError("checkpoint['models'] is not a non-empty sequence")
    elif isinstance(checkpoint, Mapping) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    if not isinstance(state_dict, Mapping):
        raise ValueError("unable to find a mapping-like state_dict in the checkpoint")

    tensor_items = OrderedDict()
    for key, value in state_dict.items():
        if isinstance(value, torch.Tensor):
            tensor_items[str(key)] = value.detach().cpu()

    if not tensor_items:
        raise ValueError("state_dict did not contain any tensor entries")

    return tensor_items


def convert_state_dict(state_dict: Mapping[str, torch.Tensor]) -> OrderedDict[str, torch.Tensor]:
    converted: OrderedDict[str, torch.Tensor] = OrderedDict()

    for key, value in state_dict.items():
        if key.endswith("num_batches_tracked"):
            continue

        if try_convert_lstm_tensor(converted, key, value):
            continue

        renamed = rename_key(key)
        converted[renamed] = transform_tensor(renamed, value)

    return converted


def try_convert_lstm_tensor(
    converted: OrderedDict[str, torch.Tensor],
    key: str,
    value: torch.Tensor,
) -> bool:
    match = re.fullmatch(
        r"separator\.separation\.(\d+)\.(band_rnn|band_comm)\.rnn\."
        r"(weight_ih|weight_hh|bias_ih|bias_hh)_l0(_reverse)?",
        key,
    )
    if match is None:
        return False

    separation_index = int(match.group(1))
    if separation_index == 0:
        raise ValueError(f"unexpected LSTM tensor under speaker fuse layer: {key}")

    block_index = separation_index - 1
    rnn_name = match.group(2)
    tensor_kind = match.group(3)
    direction = "reverse" if match.group(4) else "forward"
    transform_name = {
        "weight_ih": "input_transform.weight",
        "weight_hh": "hidden_transform.weight",
        "bias_ih": "input_transform.bias",
        "bias_hh": "hidden_transform.bias",
    }[tensor_kind]

    chunks = value.chunk(4, dim=0)
    if len(chunks) != 4:
        raise ValueError(f"expected 4 LSTM gate chunks for {key}, got {len(chunks)}")

    for gate_name, chunk in zip(LSTM_GATE_NAMES, chunks, strict=True):
        final_key = (
            f"separator.blocks.{block_index}.{rnn_name}.rnn.{direction}."
            f"{gate_name}.{transform_name}"
        )
        converted[final_key] = transform_tensor(final_key, chunk.contiguous())

    return True


def rename_key(key: str) -> str:
    key = rename_spk_model_key(key)
    key = rename_bn_key(key)
    key = rename_separator_key(key)
    key = rename_mask_key(key)
    key = rename_norm_param(key)
    return key


def rename_spk_model_key(key: str) -> str:
    return re.sub(
        r"^spk_model\.layer([234])\.se_res2block\.(0|1|2|3)\.",
        lambda match: (
            f"spk_model.layer{match.group(1)}."
            f"{('pre', 'res2', 'post', 'se')[int(match.group(2))]}."
        ),
        key,
    )


def rename_bn_key(key: str) -> str:
    match = re.fullmatch(r"BN\.(\d+)\.(0|1)\.(weight|bias)", key)
    if match is None:
        return key

    module_name = "norm" if match.group(2) == "0" else "proj"
    return f"bn.{match.group(1)}.{module_name}.{match.group(3)}"


def rename_separator_key(key: str) -> str:
    match = re.fullmatch(r"separator\.separation\.0\.fc\.linear\.(weight|bias)", key)
    if match is not None:
        return f"separator.speaker_fuse.fc.{match.group(1)}"

    match = re.fullmatch(
        r"separator\.separation\.(\d+)\.(band_rnn|band_comm)\.(norm|proj)\.(weight|bias)",
        key,
    )
    if match is None:
        return key

    separation_index = int(match.group(1))
    if separation_index == 0:
        raise ValueError(f"unexpected separator tensor layout: {key}")

    return (
        f"separator.blocks.{separation_index - 1}.{match.group(2)}."
        f"{match.group(3)}.{match.group(4)}"
    )


def rename_mask_key(key: str) -> str:
    match = re.fullmatch(r"mask\.(\d+)\.(0|1|3|5)\.(weight|bias)", key)
    if match is None:
        return key

    module_name = {
        "0": "norm",
        "1": "fc1",
        "3": "fc2",
        "5": "fc3",
    }[match.group(2)]
    return f"mask.{match.group(1)}.{module_name}.{match.group(3)}"


def rename_norm_param(key: str) -> str:
    for pattern in BATCH_OR_GROUP_NORM_WEIGHT_RES:
        key = pattern.sub(r"\1.gamma", key)
    for pattern in BATCH_OR_GROUP_NORM_BIAS_RES:
        key = pattern.sub(r"\1.beta", key)
    return key


def transform_tensor(key: str, value: torch.Tensor) -> torch.Tensor:
    if should_transpose_linear_weight(key):
        return value.transpose(0, 1).contiguous()
    return value.contiguous()


def should_transpose_linear_weight(key: str) -> bool:
    return any(pattern.fullmatch(key) for pattern in LINEAR_WEIGHT_PATTERNS)


def main() -> None:
    args = parse_args()
    output = args.output or default_output_path(args.input)
    checkpoint = torch.load(str(args.input), map_location="cpu")
    state_dict = extract_state_dict(checkpoint)
    converted = convert_state_dict(state_dict)

    output.parent.mkdir(parents=True, exist_ok=True)
    save_file(dict(converted), output)
    print(f"saved {len(converted)} tensors to {output}")


def save_file(tensors: Mapping[str, torch.Tensor], output: Path) -> None:
    if save_safetensors_file is not None:
        save_safetensors_file(dict(tensors), str(output))
        return

    save_file_fallback(tensors, output)


def save_file_fallback(tensors: Mapping[str, torch.Tensor], output: Path) -> None:
    header: dict[str, dict[str, object]] = {}
    chunks: list[bytes] = []
    offset = 0

    for name, tensor in tensors.items():
        tensor = tensor.detach().cpu().contiguous()
        chunk = tensor_to_bytes(tensor)
        header[name] = {
            "dtype": safetensors_dtype(tensor.dtype),
            "shape": list(tensor.shape),
            "data_offsets": [offset, offset + len(chunk)],
        }
        chunks.append(chunk)
        offset += len(chunk)

    header_bytes = json.dumps(header, separators=(",", ":"), sort_keys=True).encode("utf-8")
    with output.open("wb") as file:
        file.write(struct.pack("<Q", len(header_bytes)))
        file.write(header_bytes)
        for chunk in chunks:
            file.write(chunk)


def tensor_to_bytes(tensor: torch.Tensor) -> bytes:
    if tensor.dtype == torch.bfloat16:
        return tensor.view(torch.uint16).numpy().tobytes(order="C")
    return tensor.numpy().tobytes(order="C")


def safetensors_dtype(dtype: torch.dtype) -> str:
    mapping = {
        torch.float64: "F64",
        torch.float32: "F32",
        torch.float16: "F16",
        torch.bfloat16: "BF16",
        torch.int64: "I64",
        torch.int32: "I32",
        torch.int16: "I16",
        torch.int8: "I8",
        torch.uint8: "U8",
        torch.bool: "BOOL",
    }
    try:
        return mapping[dtype]
    except KeyError as error:
        raise ValueError(f"unsupported dtype for safetensors export: {dtype}") from error


if __name__ == "__main__":
    main()
