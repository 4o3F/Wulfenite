# Wulfenite — Main Reset

This branch has been reset after the TSE phase. The complete TSE stack
is archived on `v2`; `main` is now the working branch for a new
real-time PSE design.

## Goal

Design a real-time personalized speech enhancement system for Mandarin
speech that:

- suppresses competing background talkers as aggressively as possible
- keeps the target speaker natural and intelligible
- stays robust when the target speaker's emotion or speaking style
  shifts during a session
- remains deployable in low-latency interactive settings

## What Was Removed

The following TSE-specific components no longer exist on `main`:

- SpeakerBeam-SS separator implementation
- CAM++ enrollment encoder path
- enrollment-conditioned data mixer and scene composer
- TSE training loop, checkpoint format, and inference entry points
- ONNX contract tied to the previous two-network runtime

Those artifacts remain available on `v2` if they need to be consulted.

## What Remains

The reset keeps only code that is still useful for the next phase:

- AISHELL-1, AISHELL-3, MAGICDATA, CN-Celeb, and noise dataset scanners
- resampling and corpus-prep utilities
- generic audio feature extraction
- generic SDR and MR-STFT losses
- minimal Rust and Python package scaffolding

## New Design Constraints

The next model family should satisfy all of the following:

1. Streaming-capable inference with low algorithmic latency.
2. A target-conditioning path that can tolerate intra-speaker drift,
   not just a single frozen enrollment embedding.
3. Training and evaluation on Mandarin corpora, with explicit coverage
   for emotion or style mismatch between reference and target speech.
4. A deployment story that does not assume a heavyweight research-only
   runtime.

## Candidate Directions

The most relevant public baselines for the next phase are:

- pDeepFilterNet2 / pDFNet2+
- E3Net
- GTCRN with a new speaker-conditioning path
- Personalized PercepNet

The branch does not commit to one of them yet. The next code changes on
`main` should start by defining a new model interface and a new
training contract around PSE rather than reusing the old TSE wrappers.
