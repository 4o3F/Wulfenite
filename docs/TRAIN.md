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
- ≥ 50 GB free disk under the repo's `assets/` directory
  (~15 GB AISHELL-1 + ~19 GB AISHELL-3 + ~3.6 GB MUSAN noise subset
  + ~30 MB CAM++ checkpoint + training checkpoints)
- Linux (Ubuntu 22.04+ recommended) or WSL2
- Python 3.11 or 3.12
- [uv](https://github.com/astral-sh/uv) ≥ 0.5

**Where the data lives**: every local dataset, downloaded model, and
training output lives under the repo-root `assets/` directory. That
directory is gitignored, so nothing under it is ever tracked. The
training / inference CLIs expect paths relative to the `python/`
working directory (where `uv run --directory python` puts you), so
the canonical form is ``../assets/<dataset>/``.

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

From the repo root:

```bash
mkdir -p assets/aishell1
# Option A — keep the tarball's data_aishell/ wrapper directory:
tar xzf data_aishell.tgz -C assets/aishell1/
# Option B — strip the wrapper for a flatter layout:
# tar xzf data_aishell.tgz --strip-components=1 -C assets/aishell1/
```

The scanner auto-detects both layouts by looking for either
`<root>/data_aishell/` or `<root>/wav/`. Pass
``--aishell1-root ../assets/aishell1`` either way.

AISHELL-1 also ships with nested per-speaker tarballs inside each
split that must be unpacked once more:

```bash
# Adjust the first cd depending on which extraction layout you chose
cd assets/aishell1/data_aishell/wav/train
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

Extract to `assets/aishell3/` so you end up with
`assets/aishell3/train/wav/SSB0005/SSB00050001.wav`:

```bash
mkdir -p assets/aishell3
tar xzf data_aishell3.tgz --strip-components=1 -C assets/aishell3/
```

**Critical: resample to 16 kHz before training.** The official
AISHELL-3 distribution is **44.1 kHz**, not 16 kHz. Wulfenite needs
16 kHz mono, so you must convert the tree once before training.
Two options below — the ffmpeg path is 3-5× faster and recommended;
the Python script is a pure-Python fallback for environments
without ffmpeg.

**Option A — ffmpeg (recommended, ~3-5 minutes on 16 cores)**

Install ffmpeg if you don't already have it:

```bash
sudo apt install ffmpeg
```

Then from the repo root:

```bash
find assets/aishell3 -type f -name "*.wav" -print0 | \
  xargs -0 -P "$(nproc)" -n 1 -I {} bash -c '
    f="{}"
    sr=$(ffprobe -v error -select_streams a:0 -show_entries stream=sample_rate -of csv=p=0 "$f" 2>/dev/null)
    if [ "$sr" != "16000" ]; then
      tmp="${f%.wav}.16k.wav"
      ffmpeg -y -loglevel error -i "$f" -ar 16000 -ac 1 -c:a pcm_s16le "$tmp" && mv "$tmp" "$f"
    fi
  '
```

Runs one ffmpeg job per CPU core. Idempotent: ``ffprobe`` skips
files already at 16 kHz. Writes each output to a ``.16k.wav``
sibling, then atomically renames over the original — so an
interrupted run never leaves a half-written file.

**Option A with progress bar** (needs GNU parallel):

```bash
sudo apt install parallel
find assets/aishell3 -type f -name "*.wav" | \
  parallel --bar -j "$(nproc)" '
    sr=$(ffprobe -v error -select_streams a:0 -show_entries stream=sample_rate -of csv=p=0 {})
    if [ "$sr" != "16000" ]; then
      ffmpeg -y -loglevel error -i {} -ar 16000 -ac 1 -c:a pcm_s16le {.}.16k.wav && mv {.}.16k.wav {}
    fi
  '
```

``--bar`` gives you a live progress bar with ETA.

**Optional: SoX-quality resampler.** If your ffmpeg was built with
``libsoxr`` (Ubuntu's default package is), you can add
``-af aresample=resampler=soxr:precision=28`` before ``-ar 16000``
for a marginally higher-quality resample at a small speed cost.
For speech at 44.1 → 16 kHz the default swresample filter is
already fine.

**Option B — bundled Python script (slower, no extra deps)**

If you cannot install ffmpeg, the repo ships a pure-Python
equivalent using torchaudio's Kaiser-window resampler. Same
behavior (idempotent, in-place, 16-bit PCM output) but 3-5× slower
than the ffmpeg version. Runs in ~15-25 minutes on 16 cores with
a live tqdm progress bar:

```bash
uv run --directory python python -m wulfenite.scripts.resample_aishell3 \
    --root ../assets/aishell3
```

Use ``--dry-run`` to preview which files would be touched, or
``--num-workers N`` to control parallelism (default: cores − 1).

### CN-Celeb (optional speaker-count expansion)

- **Source**: http://openslr.org/82/
- **Format**: FLAC files (the archive extracts to `CN-Celeb_flac/`)
- **Content**: ~1000 Chinese speakers across interview, reading,
  singing, vlog, and other acoustic conditions
- **Why**: materially increases train-speaker diversity for the
  learnable d-vector path once AISHELL-only training stops improving

Extract so you end up with `assets/CN-Celeb_flac/data/id00001/*.flac`:

```bash
tar xzf cn-celeb_v2.tar.gz -C assets/
```

**Critical: convert FLAC to 16 kHz mono WAV before training.**
CN-Celeb ships as FLAC at mixed sample rates. Wulfenite rejects
non-16 kHz files at scan time, so unconverted files will be skipped.

Bundled Python converter (FLAC → 16 kHz WAV, written alongside
the original FLAC files):

```bash
uv run --directory python python -m wulfenite.scripts.resample_cnceleb \
    --root ../assets/CN-Celeb_flac
```

Or with GNU ``parallel`` + ffmpeg (converts FLAC to 16 kHz WAV and
removes the original FLAC files):

```bash
find assets/CN-Celeb_flac/data -type f -name '*.flac' | \
  parallel ffmpeg -y -loglevel error -i {} -ar 16000 -ac 1 -c:a pcm_s16le {.}.wav '&&' rm {}
```

### MUSAN noise (~3.6 GB — recommended default)

- **Source**: http://openslr.org/17/
- **Size**: the full MUSAN archive is ~11 GB but we only need the
  `noise/` subdirectory, ~3.6 GB (~930 files)
- **Content**: free-sound and sound-bible noise recordings covering
  room tone, HVAC, traffic, keyboard, crowd, etc., 16 kHz mono
- **Why**: teaches the mixture-aware silence branch to distinguish
  "speech from another speaker" from "non-speech noise". Much
  smaller than DNS Challenge 4 (~80 GB) with comparable coverage
  for our use case.

Download and extract just the noise subset from the repo root:

```bash
wget http://www.openslr.org/resources/17/musan.tar.gz
tar xzf musan.tar.gz -C assets/
# Optionally delete speech/ and music/ subsets — we only use noise/
rm -rf assets/musan/speech assets/musan/music
```

You end up with:

```
assets/musan/
└── noise/
    ├── free-sound/
    │   └── noise-free-sound-*.wav
    └── sound-bible/
        └── noise-sound-bible-*.wav
```

Point the mixer at `../assets/musan/noise/` (from `python/`'s working
directory). The scanner is generic (`scan_noise_dir`) and works
unchanged with any other 16 kHz noise corpus: DEMAND (~2 GB), FSD50K,
ESC-50 resampled to 16 kHz, DNS Challenge, or a custom recording set.
Just pass the root directory containing the `.wav` files.

---

## 3. Expected directory layout

After all downloads are in place, the repo-root `assets/` directory
should look like this:

```
Wulfenite/
├── assets/                        # gitignored, local-only
│   ├── aishell1/
│   │   └── data_aishell/          # or skip via --strip-components=1
│   │       └── wav/
│   │           ├── train/S0002/BAC009S0002W0122.wav
│   │           └── dev/...
│   ├── aishell3/
│   │   └── train/
│   │       └── wav/SSB0005/SSB00050001.wav
│   ├── CN-Celeb_flac/
│   │   └── data/
│   │       └── id00001/example.wav
│   ├── musan/
│   │   └── noise/
│   │       ├── free-sound/
│   │       └── sound-bible/
│   ├── campplus/
│   │   └── campplus_cn_common.bin
│   └── checkpoints/               # training outputs land here
├── python/
└── rust/
```

Everything under `assets/` is gitignored. The Wulfenite scanners
(`scan_aishell1`, `scan_aishell3`, `scan_noise_dir`) accept either
the top-level roots or the nested `data_aishell/` / `train/wav/`
subdirs — both work.

---

## 4. CAM++ checkpoint

The speaker encoder is the frozen 200k-speaker Chinese CAM++ release
from 3D-Speaker on ModelScope:
[`iic/speech_campplus_sv_zh-cn_16k-common`](https://modelscope.cn/models/iic/speech_campplus_sv_zh-cn_16k-common).

Download the single file `campplus_cn_common.bin` (~28 MB) into
`assets/campplus/`:

```bash
mkdir -p assets/campplus
# Easiest: grab it from the ModelScope web UI and move it here.
# Or use the modelscope CLI if you have it installed:
# modelscope snapshot_download iic/speech_campplus_sv_zh-cn_16k-common \
#     --local_dir assets/campplus --allow_patterns 'campplus_cn_common.bin'
```

A dedicated download script will likely land at
`python/src/wulfenite/scripts/download_campplus.py` in a future
commit; until then, manual download is the path.

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

Unit tests are fast (~40 seconds total) and do not require any
datasets — they use synthetic wav fixtures and randomly-initialized
CAM++ modules. Run them before any training to catch install or
environment issues early:

```bash
uv run --directory python pytest tests/ -v
```

Expected output: **50 passed** across models / losses / data /
training / inference. Any failure here is a red flag and should be
reported before continuing.

---

## 7. Training

The training loop lives at `python/src/wulfenite/training/train.py`
and is invoked as a module. A minimal frozen-CAM++ run with the
canonical `assets/` paths:

```bash
uv run --directory python python -m wulfenite.training.train \
    --aishell1-root ../assets/aishell1 \
    --aishell3-root ../assets/aishell3 \
    --noise-root ../assets/musan/noise \
    --campplus-checkpoint ../assets/campplus/campplus_cn_common.bin \
    --out-dir ../assets/checkpoints/phase1 \
    --batch-size 16 \
    --epochs 50 \
    --samples-per-epoch 20000 \
    --lr 1e-3
```

For the Plan C5/C5b learnable-encoder path, add `--use-learnable-encoder`
and optionally merge CN-Celeb:

```bash
uv run --directory python python -m wulfenite.training.train \
    --aishell1-root ../assets/aishell1 \
    --aishell3-root ../assets/aishell3 \
    --cnceleb-root ../assets/CN-Celeb_flac \
    --noise-root ../assets/musan/noise \
    --use-learnable-encoder \
    --out-dir ../assets/checkpoints/phase5b_cnceleb \
    --batch-size 16 \
    --epochs 50 \
    --loss-speaker-cls 0.2
```

All paths are relative to the `python/` directory because
`uv run --directory python` puts the process there. The
`../assets/...` form reaches back into the repo-root assets tree.

Before committing to a full run, do a 100-sample smoke run first to
validate the pipeline end-to-end:

```bash
uv run --directory python python -m wulfenite.training.train \
    --aishell1-root ../assets/aishell1 \
    --aishell3-root ../assets/aishell3 \
    --noise-root ../assets/musan/noise \
    --campplus-checkpoint ../assets/campplus/campplus_cn_common.bin \
    --out-dir ../assets/checkpoints/smoke_test \
    --batch-size 4 \
    --epochs 1 \
    --samples-per-epoch 100 \
    --val-samples 20 \
    --num-workers 2
```

Key design points of the training loop (from `docs/architecture.md`):

- **Whole-utterance forward** — trains the S4D parallel mode,
  exploits FFT convolution for speed. The
  ``streaming_step`` deployment path is numerically equivalent to
  the same forward pass, verified by
  ``tests/test_speakerbeam_ss.py::test_speakerbeam_streaming_matches_forward``.
- **`WulfeniteLoss` combined loss** — direct SDR + MR-STFT + target-
  absent energy penalty + presence BCE.
- **Mixture-aware silence** — `target_present_prob` defaults to
  0.85, so ~15 % of batches teach the "target is not talking →
  output silence" behavior.
- **CAM++ is frozen** — `tse.speaker_encoder` is put in eval mode
  and excluded from the optimizer, so no gradients flow through the
  encoder.
- **Mirror / offline-safe** — the default `pyproject.toml` uses the
  Aliyun PyTorch mirror so `uv sync` is fast inside China.

Training should converge in 30-60 epochs at ~20 k samples per epoch
on a 24 GB GPU, wall-clock ~24-48 hours.

Checkpoints are written to `--out-dir`:

```
assets/checkpoints/phase1/
├── train.log          # stdout mirror
├── epoch001.pt
├── epoch002.pt
├── ...
└── best.pt            # copy of whichever epoch had the lowest val_loss
```

---

## 8. ONNX export

> **Status**: export script is not yet implemented. Planned
> interface:

```bash
uv run --directory python python -m wulfenite.inference.export_onnx \
    --checkpoint ../assets/checkpoints/phase1/best.pt \
    --campplus-checkpoint ../assets/campplus/campplus_cn_common.bin \
    --out-dir ../assets/exported/
```

Produces two files matching `docs/onnx_contract.md`:

- `assets/exported/cam_plus_chinese.onnx` (~30 MB, exported from the
  frozen CAM++ weights)
- `assets/exported/wulfenite_tse.onnx` (~8 MB, the trained separator
  in its stateful form with opaque state tensors)

These are the only artifacts the Rust runtime consumes.

---

## 9. Inference on real audio

Two CLIs ship in `python/src/wulfenite/inference/`:

**Whole-utterance** (reference path, best-quality offline evaluation):

```bash
uv run --directory python python -m wulfenite.inference.whole \
    --checkpoint ../assets/checkpoints/phase5b_cnceleb/best.pt \
    --mixture ../assets/samples/real_mixture.wav \
    --enrollment ../assets/samples/real_enrollment.wav \
    --output ../assets/samples/output_whole.wav
```

Learnable-encoder checkpoints auto-detect the encoder type from the
saved checkpoint config and do **not** require `--campplus-checkpoint`.
For legacy frozen-CAM++ checkpoints, keep passing the CAM++ `.bin`:

```bash
uv run --directory python python -m wulfenite.inference.whole \
    --checkpoint ../assets/checkpoints/phase1/best.pt \
    --campplus-checkpoint ../assets/campplus/campplus_cn_common.bin \
    --mixture ../assets/samples/real_mixture.wav \
    --enrollment ../assets/samples/real_enrollment.wav \
    --output ../assets/samples/output_whole.wav
```

Prints audio duration, compute time, RTF, peak dBFS, and the
presence-head probability if the model has one.

**Streaming** (stateful frame-by-frame, simulates the Rust runtime,
measures real-time latency):

```bash
uv run --directory python python -m wulfenite.inference.streaming \
    --checkpoint ../assets/checkpoints/phase5b_cnceleb/best.pt \
    --mixture ../assets/samples/real_mixture.wav \
    --enrollment ../assets/samples/real_enrollment.wav \
    --output ../assets/samples/output_stream.wav \
    --chunk-ms 20
```

Legacy frozen-CAM++ streaming inference still requires:

```bash
--campplus-checkpoint ../assets/campplus/campplus_cn_common.bin
```

`--chunk-ms` must be a positive multiple of 10 ms (the encoder
stride). Common values: 10, 20, 40 ms. The script prints per-chunk
mean / p50 / p95 / max latency, aggregate RTF, and a real-time
feasibility verdict (warns when p95 compute exceeds the hop window).

**The two outputs should be bit-for-bit identical** — SpeakerBeam-SS's
`forward` path is aligned with `streaming_step` via deterministic
zero-padding and output cropping, verified by
`tests/test_speakerbeam_ss.py::test_speakerbeam_streaming_matches_forward`
at three chunk sizes and by `tests/test_inference.py::test_whole_vs_streaming_end_to_end`
for the full TSE wrapper.

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
  `save_checkpoint` stringifies all `Path` values in the config
  dict before saving (see `training/checkpoint.py` and the
  `test_checkpoint_roundtrip` test), and the inference scripts
  include a defensive Windows shim that aliases `PosixPath` to
  `WindowsPath` before `torch.load` runs. If you still see this
  error, open an issue with the full traceback.

**`assets/` paths not found when running training**
: The training CLI is launched via ``uv run --directory python``,
  which changes the working directory to `python/`. From there the
  repo-root `assets/` is reachable as ``../assets/``. If you prefer
  absolute paths, pass them instead — either form works.
