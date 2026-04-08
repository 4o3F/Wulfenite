# Wulfenite

Real-time target speaker extraction for Chinese audio. Two sibling
crates at the repository root:

```
Wulfenite/
├── rust/       # engineering — real-time ONNX inference, audio I/O, CLI
├── python/     # research   — model design, training, ONNX export
└── docs/       # design docs shared by both
```

The **only** artifact crossing the Python → Rust boundary is an ONNX
file. See [`docs/onnx_contract.md`](docs/onnx_contract.md) for the
interface specification.

## Current status

Main is in **pre-implementation design phase** — the previous BSRNN
line of work is archived on the [`v1`](../../tree/v1) branch. The new
architecture is [SpeakerBeam-SS](https://arxiv.org/abs/2407.01857)
(causal Conv-TasNet + S4D + frozen Chinese CAM++). See
[`docs/architecture.md`](docs/architecture.md) for the full design.

## Structure

### `rust/` — engineering crate

A minimal real-time CLI. Currently a skeleton while the Python side
builds the model; consumes `wulfenite_tse.onnx` at deploy time.

```bash
cd rust
cargo run --release -- version
```

### `python/` — research crate

Model definition, training, ONNX export. Managed with `uv`.

```bash
cd python
uv sync
```

Editable install of the `wulfenite` package; scripts live in
`python/scripts/` (added incrementally).

### `docs/`

- `architecture.md` — model architecture + design decisions
- `onnx_contract.md` — Python ↔ Rust ONNX interface specification

## License

GPL-3.0-only.
