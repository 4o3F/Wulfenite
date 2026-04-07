# Phase 0a: Fuse-Layer-Only Fine-Tune (BSRNN + CAM++ zh-cn)

Goal: improve Chinese target-speaker-extraction quality without retraining
the whole BSRNN separator. We swap the English ECAPA-TDNN speaker encoder
for the Chinese CAM++ 192-dim encoder (iic/speech_campplus_sv_zh-cn_16k-common,
trained on ~200k Chinese speakers) and train only the single
`SpeakerFuseLayer` linear projection (`Linear(192, 128)`, ~24 k parameters).

If this works: the whole pipeline moves to Chinese in a few hours of training.
If it does not work: we escalate to fine-tuning the separator itself on a
larger Chinese mixture corpus (Phase 0b).

## Deployment

After cloning the repo:

```bash
cd scripts
uv sync
```

This pulls CUDA 12.4 PyTorch wheels (~2 GB). The env works with or without a
GPU — training requires CUDA, inference does not.

## One-time: fetch the CAM++ checkpoint

```bash
cd scripts
uv run python download_campplus.py
```

Saves `campplus_cn_common.bin` (~28 MB) to `../assets/campplus/`. Public
ModelScope endpoint, no account or token required.

## Download AISHELL-1

Get the full archive (~15 GB) from OpenSLR:

- http://openslr.org/33/ → `data_aishell.tgz`

Extract so that you end up with:

```
<AISHELL_ROOT>/data_aishell/wav/train/S0002/BAC009S0002W0122.wav
<AISHELL_ROOT>/data_aishell/wav/dev/...
<AISHELL_ROOT>/data_aishell/wav/test/...
```

You may need to unpack the per-speaker `.tar.gz` files inside `wav/train/`
(AISHELL ships them nested). A one-liner:

```bash
cd <AISHELL_ROOT>/data_aishell/wav/train
for f in *.tar.gz; do tar xzf "$f" && rm "$f"; done
# repeat for dev/ and test/
```

## Run Phase 0a training

```bash
cd scripts
uv run python train_fuse.py \
    --aishell-root /path/to/aishell \
    --batch-size 16 \
    --epochs 30 \
    --samples-per-epoch 10000
```

On a 24 GB GPU, `batch-size=16` with 4 s segments is comfortable. Each epoch
processes `samples_per_epoch` fresh on-the-fly mixtures; 30 epochs at 10 000
samples/epoch is ~300 k training mixtures total and should run in a few
hours. Checkpoints land in `../assets/campplus/train_phase0a/epoch*.pt` and
`best.pt`.

Important flags:

| flag | default | notes |
|---|---|---|
| `--batch-size` | 8 | 16 fits in 24 GB with 4 s chunks |
| `--samples-per-epoch` | 10000 | virtual epoch size for the on-the-fly mixer |
| `--segment-seconds` | 4.0 | mixture + target chunk length |
| `--enrollment-seconds` | 4.0 | enrollment chunk length |
| `--snr-range-db` | -5 5 | random target-vs-interferer SNR |
| `--lr` | 1e-3 | large LR is fine: only 24 k params move |
| `--epochs` | 30 | fuse layer usually converges well before this |

## Validate on real audio

```bash
uv run python infer_fuse.py \
    --bsrnn-ckpt ../assets/avg_model.pt \
    --campplus-ckpt ../assets/campplus/campplus_cn_common.bin \
    --fuse-ckpt ../assets/campplus/train_phase0a/best.pt \
    --mixture ../assets/mixture2.wav \
    --enrollment ../assets/enrollment2.wav \
    --output /tmp/bsrnn_campplus_out.wav
```

A/B listen against the English baseline:

```bash
uv run python ../assets/wesep_bsrnn_ecapa_pytorch_only.py \
    --checkpoint ../assets/avg_model.pt \
    --mixture ../assets/mixture2.wav \
    --enrollment ../assets/enrollment2.wav \
    --output /tmp/english_baseline_out.wav
```

## What "success" looks like for Phase 0a

- **Training converges**: SI-SDR loss drops below ~-6 dB on the validation
  split within 5–10 epochs.
- **A/B listening**: on Chinese mixtures with interfering speakers, the new
  output audibly suppresses the interferer more than the English baseline.
- **RTF unchanged**: CAM++ is smaller than ECAPA; CPU inference RTF should
  remain comparable to the existing 0.37.

If all three hold: we promote this checkpoint and continue with the ORT
refactor. If training converges but A/B is not better, the separator itself
carries English priors and we escalate to Phase 0b (unfreeze the separator,
train on CN-Celeb).

## Phase 0b: unfreeze the separator + acoustic augmentation

If Phase 0a hits its ceiling (works on AISHELL dev, fails on real
small-room audio with low output level), run Phase 0b. This:

- Unfreezes the whole BSRNN separator (CAM++ stays frozen).
- Initializes from `../assets/campplus/train_phase0a/best.pt` to keep
  the learned fuse-layer mapping.
- Adds synthetic room impulse responses (RT60 0.08-0.35 s) applied
  per-speaker so the training mixtures look like real small-room audio.
- Adds light broadband noise at 15-30 dB SNR.
- Uses SI-SDR + log-magnitude penalty so the training objective is
  **not** scale-invariant (fixes Phase 0a's "output is always quiet"
  artifact).
- Runs at lr=1e-4 (vs 1e-3 for Phase 0a) since ~6M params move.

```bash
uv run python train_phase0b.py \
    --aishell-root /path/to/aishell \
    --batch-size 12 \
    --epochs 50
```

On a 24 GB GPU, batch_size=12 with 4 s segments fits comfortably with
full separator grads. Expect ~3-5 hours for 50 epochs.

Validate the same way as Phase 0a but point `--fuse-ckpt` at the
Phase 0b checkpoint:

```bash
uv run python infer_fuse_debug.py \
    --fuse-ckpt ../assets/campplus/train_phase0b/best.pt \
    --mixture <your_real_mixture.wav> \
    --enrollment <your_real_enrollment.wav> \
    --output /tmp/phase0b_out.wav
```

The debug script also supports `--rescale-output-to-input` for a quick
sanity check, though with the log-magnitude penalty in Phase 0b's loss
the output level should already be close to the mixture level.

## Files in this directory

- `campplus.py` — standalone CAM++ model class (merged from 3D-Speaker)
- `bsrnn_campplus.py` — wrapper: WeSep BSRNN with CAM++ swapped in
- `aishell_mixer.py` — on-the-fly AISHELL-1 2-speaker mixer dataset (augmentable)
- `augmentation.py` — synthetic RIRs + noise helpers for Phase 0b
- `train_fuse.py` — Phase 0a training loop (fuse-layer-only)
- `train_phase0b.py` — Phase 0b training loop (separator unfrozen + augmentation)
- `infer_fuse.py` — plain inference
- `infer_fuse_debug.py` — inference with internal stat dumps + output rescale
- `diagnose_phase0a.py` — sanity check a Phase 0a checkpoint on AISHELL dev
- `download_campplus.py` — fetches the CAM++ checkpoint from ModelScope
