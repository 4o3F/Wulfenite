# Wulfenite

Real-time target speaker extraction for Chinese audio streams. Extracts
a single target speaker's voice from a multi-speaker mixture, guided by
a short enrollment clip of the target's voice.

```
Wulfenite/
├── rust/       # engineering — real-time ONNX inference, audio I/O, CLI
├── python/     # research   — model design, training, ONNX export
└── docs/       # design docs shared by both
```

The **only** artifact crossing the Python → Rust boundary is an ONNX
file. See [`docs/onnx_contract.md`](docs/onnx_contract.md) for the
interface specification.

## Architecture

**SpeakerBeam-SS** (causal Conv-TasNet + S4D, [arXiv 2407.01857](https://arxiv.org/abs/2407.01857))
with a learnable d-vector speaker encoder trained end-to-end.

- Separator: ~7.6M params, paper-faithful defaults (N=4096, B=256, D=32)
- Speaker encoder: ~1M params learnable x-vector with auxiliary classification
- FiLM speaker conditioning: `feat = feat * gamma(e) + beta(e)`
- Loss: direct SDR + multi-resolution STFT + silence penalty + presence head
- Causal streaming with stateful S4D recurrence

See [`docs/architecture.md`](docs/architecture.md) for the full design.

## Quick start

### Setup

```bash
cd python
uv sync
```

### Training

Three clean Chinese speech datasets (AISHELL-1/3 + MAGICDATA) with
on-the-fly mixing. See [`docs/TRAIN.md`](docs/TRAIN.md) for dataset
download and preparation.

Training uses Adam with ReduceLROnPlateau, early stopping (patience 20),
and selects the best checkpoint by max `val_sdri_db` on speaker-disjoint
validation. If the N=4096 separator OOMs on your GPU, try `--batch-size 12`
or `--batch-size 8`.

**Learnable d-vector** (default, trains encoder from scratch):

```bash
uv run --directory python python -m wulfenite.training.train \
    --aishell1-root ../assets/aishell1 \
    --aishell3-root ../assets/aishell3 \
    --magicdata-root ../assets/magicdata \
    --noise-root ../assets/musan/noise \
    --out-dir ../assets/checkpoints/phase3 \
    --batch-size 16 \
    --epochs 200 \
    --lr 5e-4
```

**Frozen CAM++** (pretrained 200k-speaker encoder, only projection +
separator train):

```bash
uv run --directory python python -m wulfenite.training.train \
    --encoder-type campplus-frozen \
    --campplus-checkpoint ../assets/campplus_cn_common.bin \
    --aishell1-root ../assets/aishell1 \
    --aishell3-root ../assets/aishell3 \
    --magicdata-root ../assets/magicdata \
    --noise-root ../assets/musan/noise \
    --out-dir ../assets/checkpoints/phase3-campplus-frozen \
    --batch-size 16 \
    --epochs 200 \
    --lr 5e-4
```

**Fine-tune CAM++** (pretrained encoder jointly trained at lower LR):

```bash
uv run --directory python python -m wulfenite.training.train \
    --encoder-type campplus-finetune \
    --campplus-checkpoint ../assets/campplus_cn_common.bin \
    --aishell1-root ../assets/aishell1 \
    --aishell3-root ../assets/aishell3 \
    --magicdata-root ../assets/magicdata \
    --noise-root ../assets/musan/noise \
    --out-dir ../assets/checkpoints/phase3-campplus-finetune \
    --batch-size 16 \
    --epochs 200 \
    --lr 5e-4
```

### Inference

**Whole-utterance** (highest quality, offline evaluation):

```bash
uv run --directory python python -m wulfenite.inference.whole \
    --checkpoint ../assets/checkpoints/phase3/best.pt \
    --mixture ./samples/mixture.wav \
    --enrollment ./samples/enrollment.wav \
    --output ./output.wav
```

**Streaming** (frame-by-frame, simulates real-time Rust runtime):

```bash
uv run --directory python python -m wulfenite.inference.streaming \
    --checkpoint ../assets/checkpoints/phase3/best.pt \
    --mixture ./samples/mixture.wav \
    --enrollment ./samples/enrollment.wav \
    --output ./output_stream.wav \
    --chunk-ms 20
```

### Tests

```bash
uv run --directory python pytest tests/ -v
```

Expected: **73 passed**. Tests cover model shapes, streaming/forward
equivalence, FiLM initialization, SDR metric helpers, data scanners,
and training smoke tests. No external datasets or GPU required.

## Structure

### `rust/` — engineering crate

Real-time CLI consuming `wulfenite_tse.onnx` at deploy time.

```bash
cd rust
cargo run --release -- version
```

### `python/` — research crate

Model definition, training, inference, ONNX export. Managed with `uv`.

### `docs/`

- [`architecture.md`](docs/architecture.md) — model architecture + design decisions
- [`onnx_contract.md`](docs/onnx_contract.md) — Python ↔ Rust ONNX interface
- [`TRAIN.md`](docs/TRAIN.md) — dataset preparation + training guide

## License

GPL-3.0-only.
