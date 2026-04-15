# Wulfenite

Wulfenite is being reset from a causal TSE codebase into a new
real-time PSE research branch. The full TSE implementation has been
archived on the local `v2` branch; `main` now keeps only reusable
dataset and signal-processing utilities while the next model family is
designed.

```
Wulfenite/
├── rust/       # engineering placeholder for the future runtime
├── python/     # research utilities, datasets, losses, and new model work
└── docs/       # current design notes for the reset
```

## Current State

- The old SpeakerBeam-SS + CAM++ TSE stack has been removed from `main`.
- Dataset scanners for AISHELL-1, AISHELL-3, MAGICDATA, CN-Celeb, and
  noise corpora remain available.
- Generic audio helpers and generic enhancement losses remain available.
- No training loop, inference entry point, or ONNX contract is defined
  on `main` yet.

The reset design note lives in [`docs/architecture.md`](docs/architecture.md).

## Python Setup

```bash
cd python
uv sync
```

## Tests

```bash
uv run --directory python python -m pytest tests/ -v
```

The remaining test suite covers the retained dataset scanners,
augmentation utilities, audio features, and generic enhancement losses.

## Structure

### `rust/`

Minimal engineering placeholder while the next real-time architecture
is selected.

### `python/`

Research utilities retained across the reset:

- dataset scanning and resampling helpers
- generic feature extraction
- generic SDR and MR-STFT losses

### `docs/`

- [`architecture.md`](docs/architecture.md) — reset status and new model constraints
- [`datasets.md`](docs/datasets.md) — retained corpus assumptions and prep notes

## License

GPL-3.0-only.
