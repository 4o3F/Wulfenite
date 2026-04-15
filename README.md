# Wulfenite

Wulfenite is a real-time personalized speech enhancement (PSE) research
codebase built around **pDFNet2** — a DeepFilterNet2 backbone with
optional speaker-conditioned refinement.

```
Wulfenite/
├── rust/       # engineering placeholder for the future runtime
├── python/     # models, training, inference, evaluation
├── configs/    # TOML experiment configs (version-controlled)
└── docs/       # design notes and dataset documentation
```

## Quick Start

### 1. Install

```bash
cd python
uv sync
```

### 2. Prepare Data

Place datasets under an `assets/` directory (or adjust paths in config):

```
assets/
├── aishell1/data_aishell/wav/train/S0002/*.wav
├── aishell3/train/wav/SSB0005/*.wav      # resample first (see below)
├── magicdata/train/01_1000/*.wav         # optional
└── musan/noise/*.wav
```

AISHELL-3 ships at 44.1 kHz and must be resampled to 16 kHz:

```bash
uv run python -m wulfenite.scripts.resample_aishell3 --root ../assets/aishell3
```

### 3. Train

Training is driven by TOML config files. Copy and edit the default:

```bash
cp configs/pdfnet2_aishell.toml configs/my_experiment.toml
# Edit paths and hyperparameters
vim configs/my_experiment.toml
```

Start training:

```bash
cd python
uv run python -m wulfenite.scripts.train_pdfnet2 --config ../configs/my_experiment.toml
```

Override parameters without editing the file:

```bash
uv run python -m wulfenite.scripts.train_pdfnet2 --config ../configs/my_experiment.toml \
    --override training.batch_size=16 \
    --override training.max_epochs=50
```

Quick sanity check (tiny run):

```bash
uv run python -m wulfenite.scripts.train_pdfnet2 --config ../configs/pdfnet2_aishell.toml \
    --override training.batch_size=2 \
    --override training.max_epochs=2 \
    --override data.epoch_size=10 \
    --override data.val_size=5
```

#### Model types

Set `[model].type` in the config:

| Type | Description | Speaker encoder needed |
|------|-------------|----------------------|
| `dfnet` | Plain DfNet2 speech enhancement (no speaker conditioning) | No |
| `pdfnet2` | Personalized DfNet2 with 192-D speaker embedding | Yes (WeSpeaker-compatible `.pt` checkpoint) |

#### Config reference

See [`configs/pdfnet2_aishell.toml`](configs/pdfnet2_aishell.toml) for all available options.

### 4. Inference

Inference also uses TOML config:

```bash
# Edit inference config
vim configs/infer.toml

# Batch enhancement
cd python
uv run python -m wulfenite.scripts.infer --config ../configs/infer.toml

# Streaming enhancement
uv run python -m wulfenite.scripts.infer --config ../configs/infer.toml \
    --override streaming.enabled=true

# With SI-SDR evaluation against clean reference
uv run python -m wulfenite.scripts.infer --config ../configs/infer.toml \
    --override eval.enabled=true \
    --override eval.reference=clean_wavs/
```

See [`configs/infer.toml`](configs/infer.toml) for all inference options.

### 5. Resume Training

Uncomment the `resume` line in your training config:

```toml
[training]
resume = "checkpoints/pdfnet2_aishell/last.pt"
```

## Tests

```bash
cd python
uv run python -m pytest tests/ -v
```

86 tests covering models, data pipeline, training utilities, losses, and evaluation.

## Architecture

- **DfNet2**: Dual ERB + DF branch encoder, GroupedGRU fusion, iterative deep filtering
- **PDfNet2**: DfNet2 + frozen native ECAPA-TDNN speaker embedding conditioning
- **PDfNet2+**: PDfNet2 + TinyECAPA on-the-fly similarity refinement (future phase)

Details in [`docs/architecture.md`](docs/architecture.md).

## Structure

### `python/`

- `wulfenite.models` — DfNet2, PDfNet2, PDfNet2Plus, TinyECAPA, SpeakerEncoder
- `wulfenite.data` — AISHELL/MAGICDATA scanners, PSEMixer, augmentation
- `wulfenite.training` — training loops, LR scheduler, KD dataset
- `wulfenite.inference` — batch and streaming Enhancer
- `wulfenite.evaluation` — SI-SDR, optional PESQ/STOI
- `wulfenite.scripts` — CLI entry points

### `configs/`

TOML experiment configs, designed for version control.

### `docs/`

- [`architecture.md`](docs/architecture.md) — model design and constraints
- [`datasets.md`](docs/datasets.md) — corpus assumptions and prep notes

## License

GPL-3.0-only.
