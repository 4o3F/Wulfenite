"""Cross-platform safe checkpoint save/load.

The v1 branch hit a painful gotcha on cross-OS loading: ``Path``
objects inside the pickled ``args`` dict became ``PosixPath`` on
Linux, which Windows' ``pathlib`` cannot instantiate at unpickle
time. The v2 checkpoint format sidesteps this entirely by
stringifying any ``Path`` value in the config dict before saving.

Checkpoints store:

- ``epoch``: last completed epoch (1-indexed)
- ``step``: global step counter
- ``model_state_dict``: the WulfeniteTSE state_dict
- ``optimizer_state_dict``: optimizer state
- ``scheduler_state_dict``: LR scheduler state (may be None)
- ``config``: serialized TrainingConfig, Paths → str
- ``metrics``: arbitrary metrics dict (train/val losses, etc.)
- ``wulfenite_version``: string version tag for future compat checks
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import torch

from .. import __version__


def _serialize_config(config: Any) -> dict[str, Any]:
    """Convert a config (dataclass or dict) to a plain JSON-ish dict.

    Any ``Path`` value is stringified so the pickle can be read on
    any OS regardless of which OS saved it.
    """
    if dataclasses.is_dataclass(config):
        items = dataclasses.asdict(config).items()
    elif isinstance(config, dict):
        items = config.items()
    else:
        raise TypeError(
            f"config must be a dataclass or dict, got {type(config).__name__}"
        )
    out: dict[str, Any] = {}
    for k, v in items:
        if isinstance(v, Path):
            out[k] = str(v)
        else:
            out[k] = v
    return out


def save_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    epoch: int = 0,
    step: int = 0,
    config: Any = None,
    metrics: dict | None = None,
) -> Path:
    """Atomically save a Wulfenite training checkpoint.

    Args:
        path: target file path. Parent directories are created.
        model: the module whose ``state_dict`` should be persisted.
        optimizer: optional optimizer to snapshot.
        scheduler: optional LR scheduler to snapshot.
        epoch / step: counters for resume.
        config: training config (dataclass or dict). ``Path`` values
            are stringified for cross-OS portability.
        metrics: arbitrary extra fields, typically train/val losses.

    Returns:
        The resolved checkpoint path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "wulfenite_version": __version__,
        "epoch": epoch,
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "config": _serialize_config(config) if config is not None else None,
        "metrics": dict(metrics or {}),
    }

    # Write to a temp file first then rename, so an interrupted save
    # does not leave a half-written checkpoint.
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(path)
    return path


def peek_checkpoint_config(path: str | Path) -> dict[str, Any]:
    """Read only the serialized config from a checkpoint.

    This is used by inference to determine which model family should be
    constructed before attempting a full ``load_state_dict``.
    """
    path = Path(path)
    payload = torch.load(str(path), map_location="cpu", weights_only=False)
    config = payload.get("config")
    return config if isinstance(config, dict) else {}


def load_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
) -> dict:
    """Load a Wulfenite checkpoint into the given model (and optionally
    optimizer / scheduler).

    Args:
        path: checkpoint file to read.
        model: target module. Its state_dict is loaded in place.
        optimizer: optional — if provided, its state is restored too.
        scheduler: optional — if provided, its state is restored too.
        map_location: passed through to ``torch.load``.
        strict: passed through to ``model.load_state_dict``.

    Returns:
        Dict with the extra fields the checkpoint carried:
        ``epoch``, ``step``, ``config``, ``metrics``, ``wulfenite_version``.
    """
    path = Path(path)
    payload = torch.load(str(path), map_location=map_location, weights_only=False)

    model.load_state_dict(payload["model_state_dict"], strict=strict)
    if optimizer is not None and payload.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    if scheduler is not None and payload.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(payload["scheduler_state_dict"])

    return {
        "epoch": payload.get("epoch", 0),
        "step": payload.get("step", 0),
        "config": payload.get("config"),
        "metrics": payload.get("metrics", {}),
        "wulfenite_version": payload.get("wulfenite_version"),
    }
