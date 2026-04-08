# Wulfenite — Training Guide

This document walks through every step needed to train the Wulfenite
SpeakerBeam-SS target speaker extractor from a clean clone, from
dataset downloads through to a trained ONNX file ready for the Rust
runtime.

The training pipeline is implemented inside `python/`. The
engineering runtime in `rust/` only consumes the final ONNX file
(see `onnx_contract.md`) and is not involved in training.

## Table of contents

1. [Hardware and environment](#1-hardware-and-environment)
2. [Datasets to download](#2-datasets-to-download)
3. [Expected directory layout](#3-expected-directory-layout)
4. [CAM++ checkpoint](#4-cam-checkpoint)
5. [Python environment setup](#5-python-environment-setup)
6. [Running the unit tests](#6-running-the-unit-tests)
7. [Training](#7-training)
8. [ONNX export](#8-onnx-export)
9. [Inference on real audio](#9-inference-on-real-audio)
10. [Troubleshooting](#10-troubleshooting)

---

## 1. Hardware and environment

**Minimum**:
- 1× NVIDIA GPU with ≥ 12 GB VRAM (tested on RTX 3090 / 4090, 24 GB)
- ≥ 64 GB RAM
- ≥ 300 GB free disk (see dataset sizes below)
- Linux (Ubuntu 22.04+ recommended) or WSL2
- Python 3.11 or 3.12
- [uv](https://github.com/astral-sh/uv) ≥ 0.5

**Recommended**:
- 24 GB VRAM so you can comfortably fit batch size 16 with 4 s segments
- Fast NVMe for the training data (spinning rust will bottleneck the
  `WulfeniteMixer` workers)

---

## 2. Datasets to download

Three separate datasets, all free:

### AISHELL-1 (primary clean Chinese speech)

- **Source**: http://openslr.org/33/
- **Size**: ~15 GB extracted
- **Content**: 178 hours of 16 kHz mono Mandarin speech from 400
  speakers, studio recordings, read utterances
- **Why**: the largest clean Mandarin corpus with disjoint speaker
  splits — gives the model a solid baseline of speaker diversity

Download link: `data_aishell.tgz` from openslr.org/33/resources.

Extract to `~/datasets/aishell1/` so you end up with
`~/datasets/aishell1/data_aishell/wav/train/S0002/BAC009S0002W0122.wav`
etc. AISHELL-1 ships with nested per-speaker tarballs that must be
unpacked once more:

```bash
cd ~/datasets/aishell1/data_aishell/wav/train
for f in *.tar.gz; do tar xzf "$f" && rm "$f"; done
cd ../dev    && for f in *.tar.gz; do tar xzf "$f" && rm "$f"; done
cd ../test   && for f in *.tar.gz; do tar xzf "$f" && rm "$f"; done
```

### AISHELL-3 (additional speaker diversity)

- **Source**: http://openslr.org/93/
- **Size**: ~19 GB extracted
- **Content**: ~85 hours across 218 new speakers (disjoint from
  AISHELL-1), 16 kHz mono
- **Why**: doubles the speaker count, improving cross-speaker
  generalization

Extract to `~/datasets/aishell3/` so you end up with
`~/datasets/aishell3/train/wav/SSB0005/SSB00050001.wav`.

### DNS Challenge 4 noise (~80 GB)

- **Source**: https://github.com/microsoft/DNS-Challenge (ICASSP 2023
  DNS5 or DNS4 equivalent)
- **Content**: thousands of short noise recordings (room tone, HVAC,
  traffic, keyboard, crowd, etc.)
- **Why**: teaches the mixture-aware silence branch to distinguish
  "speech from another speaker" from "non-speech noise"

DNS ships as a Git LFS repo — cloning it is slow but straightforward:

```bash
git clone https://github.com/microsoft/DNS-Challenge.git
cd DNS-Challenge/datasets/noise
# flat .wav files, 16 kHz mono
```

Point the mixer at `DNS-Challenge/datasets/noise/` (or a subset you
have space for).

---

## 3. Expected directory layout

After all downloads are in place, the local layout should look like:

```
~/datasets/
├── aishell1/
│   └── data_aishell/
│       └── wav/
│           ├── train/S0002/BAC009S0002W0122.wav
│           └── dev/...
├── aishell3/
│   └── train/
│       └── wav/SSB0005/SSB00050001.wav
└── dns_noise/
    ├── noise_000001.wav
    └── ...
```

The Wulfenite scanners (`scan_aishell1`, `scan_aishell3`,
`scan_dns_noise`) accept either the top-level roots or the nested
`data_aishell/` / `train/wav/` subdirs — both work.

---

## 4. CAM++ checkpoint

The speaker encoder is the frozen 200k-speaker Chinese CAM++ release
from 3D-Speaker on ModelScope:
[`iic/speech_campplus_sv_zh-cn_16k-common`](https://modelscope.cn/models/iic/speech_campplus_sv_zh-cn_16k-common).

Download the single file `campplus_cn_common.bin` (~28 MB) and
place it anywhere, e.g. `~/datasets/campplus/campplus_cn_common.bin`.

A download script matching v1's will be added to
`python/scripts/download_campplus.py` (TODO); for now, pull manually
from the ModelScope web UI or via `modelscope snapshot_download`.

---

## 5. Python environment setup

From a fresh clone:

```bash
git clone git@github.com:4o3F/Wulfenite.git
cd Wulfenite/python
uv sync
```

This creates `python/.venv/` and installs `wulfenite` in editable
mode using the default PyTorch CUDA 12.6 wheels from Aliyun's mirror
(fast inside China, works anywhere). If you are outside China and
would prefer the upstream PyTorch index, edit
`python/pyproject.toml` and delete the `[tool.uv.sources]` and
`[[tool.uv.index]]` sections.

Verify the install:

```bash
uv run python -c "import wulfenite; print(wulfenite.__version__)"
```

---

## 6. Running the unit tests

Unit tests are fast (< 15 seconds total) and do not require any
datasets — they use synthetic wav fixtures. Run them before any
training to catch install or environment issues early:

```bash
uv run --directory python pytest tests/ -v
```

Expected output: **39 passed** across models / losses / data. Any
failure here is a red flag and should be reported before continuing.

---

## 7. Training

> **Status**: training loop is not yet implemented. This section
> documents the planned workflow so the interface is clear before
> the training script lands.

Once `python/src/wulfenite/training/train.py` is written, training
will look like:

```bash
uv run --directory python python -m wulfenite.training.train \
    --aishell1-root ~/datasets/aishell1 \
    --aishell3-root ~/datasets/aishell3 \
    --dns-noise-root ~/datasets/dns_noise \
    --campplus-checkpoint ~/datasets/campplus/campplus_cn_common.bin \
    --out-dir ./checkpoints/phase1 \
    --batch-size 16 \
    --epochs 50 \
    --samples-per-epoch 20000 \
    --lr 1e-3
```

Key design points of the training loop (from `docs/architecture.md`):

- **Whole-utterance forward** — trains the S4D parallel mode,
  exploits FFT convolution for speed, does NOT thread per-frame state
- **`WulfeniteLoss` combined loss** — direct SDR + MR-STFT + target-
  absent energy penalty + presence BCE
- **Mixture-aware silence** — `MixerConfig.target_present_prob` set
  to 0.85, so ~15 % of batches teach the "target is not talking →
  output silence" behavior
- **CAM++ is frozen** — `tse.speaker_encoder.eval()`, no gradients
  flow through the encoder
- **Mirror / offline-safe** — the default `pyproject.toml` uses the
  Aliyun PyTorch mirror so `uv sync` is fast inside China

Training should converge in 30-60 epochs at ~20 k samples per epoch
on a 24 GB GPU, wall-clock ~24-48 hours.

---

## 8. ONNX export

> **Status**: export script is not yet implemented. Planned
> interface:

```bash
uv run --directory python python -m wulfenite.inference.export_onnx \
    --checkpoint ./checkpoints/phase1/best.pt \
    --out-dir ./exported/
```

Produces two files matching `docs/onnx_contract.md`:

- `exported/cam_plus_chinese.onnx` (~30 MB, exported from the frozen
  CAM++ weights)
- `exported/wulfenite_tse.onnx` (~8 MB, the trained separator in its
  stateful form with opaque state tensors)

These are the only artifacts the Rust runtime consumes.

---

## 9. Inference on real audio

> **Status**: inference script is not yet implemented. Planned:

Whole-utterance (offline evaluation, full quality):

```bash
uv run --directory python python -m wulfenite.inference.whole \
    --checkpoint ./checkpoints/phase1/best.pt \
    --mixture ./samples/real_mixture.wav \
    --enrollment ./samples/real_enrollment.wav \
    --output ./output.wav
```

Streaming (simulates the Rust real-time path, validates the
stateful mode):

```bash
uv run --directory python python -m wulfenite.inference.streaming \
    --checkpoint ./checkpoints/phase1/best.pt \
    --mixture ./samples/real_mixture.wav \
    --enrollment ./samples/real_enrollment.wav \
    --output ./output_stream.wav \
    --chunk-ms 20 \
    --lookahead-ms 20
```

The two outputs should sound near-identical, since SpeakerBeam-SS's
S4D + causal conv form is exactly equivalent between the parallel
training path and the stateful streaming path (verified by
`tests/test_s4d.py::test_s4d_parallel_equals_step`).

---

## 10. Troubleshooting

**`uv sync` is slow or fails downloading torch**
: The default config uses Aliyun's PyTorch mirror. If you are outside
  China, torch downloads may timeout. Temporarily switch the
  `[tool.uv.sources]` section in `python/pyproject.toml` to the
  upstream PyTorch index, or run `UV_INDEX=... uv sync`.

**`AISHELL-1 layout not found under ...`**
: The scanner expects either `<root>/data_aishell/wav/{train,dev,test}/`
  or `<root>/wav/{train,dev,test}/`. If you extracted the archive to
  an unusual place, point `--aishell1-root` at the directory that
  directly contains `data_aishell/`, not at a parent two levels above.

**Per-speaker AISHELL-1 tarballs are missing**
: Older AISHELL-1 distributions ship `.tar.gz` files inside
  `wav/{train,dev,test}/` that need to be unpacked once. See
  section 2 for the one-liner.

**GPU is idle / `nvidia-smi` shows < 30 % utilization**
: The mixer is CPU-bound. Increase `--num-workers` (aim for ≥ 8 on
  a modern CPU), put datasets on NVMe, and bump `--prefetch-factor`
  if your DataLoader config supports it. Verify that
  `augmentation.apply_rir` is FFT-based (it is in this branch); if
  it were direct conv it would dominate per-sample cost.

**Loss diverges / NaNs**
: Lower the learning rate (`--lr 5e-4`). If the NaN comes from the
  S4D layer specifically, lower `dt_max` in the S4D constructor (see
  the comment in `python/src/wulfenite/models/s4d.py`).

**Cannot load trained checkpoint on Windows**
: v1 hit a `PosixPath` unpickling issue on cross-OS loads. v2
  training scripts should stringify all `Path` values in the `args`
  dict before saving. If you see this on a future checkpoint, open
  an issue with the full traceback.
