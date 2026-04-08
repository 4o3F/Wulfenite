# wulfenite — Python research crate

Model design, training, evaluation, and ONNX export for the Wulfenite
real-time target speaker extraction project. This is one of the two
sibling crates at the repository root; the other is `../rust/`, which
consumes the ONNX files exported from here.

## Layout

```
python/
├── pyproject.toml           # uv-managed, editable install of wulfenite
├── src/wulfenite/           # package source
│   └── __init__.py
├── scripts/                 # thin entry-point scripts (added later)
└── tests/                   # unit tests (added later)
```

## Setup

```bash
cd python
uv sync
```

This creates a `.venv/` and installs `wulfenite` in editable mode. You
can then `uv run python -m wulfenite.training.train ...` or import from
any script in the venv.

## Design references

- `../docs/architecture.md` — overall model design, data flow, design decisions
- `../docs/onnx_contract.md` — ONNX interface between Python and Rust
