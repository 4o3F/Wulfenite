# Wulfenite — Architecture

This document records the frozen design decisions for the main branch
(post v1 reset). The v1 branch preserves the earlier BSRNN pipeline;
main starts from this architecture.

## 1. Goal

Real-time target speaker extraction (TSE) for Chinese audio, with two
deployment scenarios:

- **Live-streaming denoising** — 500 ms – 2 s acceptable latency,
  one-way broadcast.
- **Voice calls / gaming chat** — sub-50 ms one-way algorithmic
  latency, bidirectional interactive.

The same model must serve both. The second constraint is the binding
one: any architecture that hits < 50 ms will also be fine for
streaming.

## 2. Why the v1 BSRNN was abandoned

BSRNN's separator uses **bidirectional LSTM** throughout. BiLSTM's
backward pass requires seeing the END of the sequence before emitting
any output, which imposes an algorithmic latency floor equal to the
chunk length. The smallest chunk that still gives acceptable quality
is ~500 ms, which is fine for broadcast but unusable for voice calls.
BiLSTM hidden state also cannot be carried across chunks, so there is
no trivial "save state, process new samples" streaming trick.

Additionally, v1 hit two training-pipeline pitfalls:

1. **Scale-invariant SI-SDR loss → degenerate solutions.** Phase 0a
   converged to tiny-output (shape-correct but amplitude ≈ 0). Phase
   0b with SI-SDR + log-magnitude penalty converged to pass-through
   (`output ≈ mixture`). Both look good by SI-SDR numbers on the val
   set but fail on real audio.
2. **No target-absent training samples.** Every training mixture
   contained a fully-speaking target, so the model never learned
   "when target is silent, output silence." Real audio has natural
   target pauses; the model falsely extracts the interferer during
   those pauses.

The new design fixes both by changing loss and training data
construction — see sections 5 and 6.

## 3. Model architecture

**Separator**: [SpeakerBeam-SS](https://arxiv.org/abs/2407.01857)
(Sato, Moriya, Mimura, Horiguchi, Ochiai, Ashihara, Ando, Shinayama,
Delcroix — NTT, Interspeech 2024).

- **Backbone**: Conv-TasNet-style time-domain encoder → separator
  stack → decoder.
- **Paper-faithful defaults**: encoder channels `N = 4096`,
  bottleneck `B = 256`, hidden `H = 512`, S4D state `D = 32`.
- **Separator stack**: `2 × (TCN(d=1), TCN(d=2), TCN(d=4), S4D, TCN(d=8))`.
  The S4D block uses channel-wise LayerNorm, the S4D core, then an
  `Linear(B, H) -> activation -> Linear(H, B)` sublayer before the
  residual add.
- **Frontend**: learned 1-D conv encoder, kernel size 320, hop 160
  (20 ms at 16 kHz). Over-parameterized relative to the 2 ms kernel
  of the original TD-SpeakerBeam to compensate for the larger stride.
- **Parameter count**: ~7.57 M for the current separator implementation.
  The paper reports ~7.93 M for its reference configuration; the
  remaining delta is now isolated to the exact block-level accounting,
  not the gross topology or channel widths.
- **Causality**: **causal by default.** Global LayerNorm replaced by
  channel-wise LayerNorm; convolutions are strictly causal. S4D has a
  recurrent form that supports stateful inference across frames.

**Speaker encoder**:

- **Single path**: a fine-tuned CAM++ encoder whose L2-normalized
  embedding is native 192-d. The separator's FiLM layers project from
  192-d to 256-d internally.

The speaker encoder still runs exactly ONCE per session, not per frame.
TSE's core assumption is that the target speaker is fixed for the
duration of a session (one streamer per broadcast, one caller per phone
call, one gamer per match). The enrollment is processed once at session
startup, the resulting speaker embedding is cached, and every
subsequent frame of the real-time separator is conditioned on that same
cached embedding. This keeps the encoder in the setup budget rather than
the steady-state per-frame budget — see
`docs/onnx_contract.md` for the concrete two-file split.

### Speaker embedding fusion — design decision

The separator uses **FiLM speaker adaptation** (feature-wise linear
modulation) on one of its bottleneck feature tensors. The L2-normalized
speaker embedding `e` feeds two learnable `Linear(192, 256, bias=False)`
branches:

```
feat ← feat * (1 + gamma(e)) + beta(e)
```

Both `gamma` and `beta` weight matrices are initialized to zeros, so
at step 0 the FiLM conditioning is an exact identity/no-op. The
residual `1 + gamma(e)` formulation ensures the separator starts from
unmodulated features and gradually learns speaker-dependent modulation.
See `speakerbeam_ss.py::_apply_speaker_conditioning` for the
implementation shared between `forward()` and `streaming_step()`.

The separator uses `B = 256` internally, while the CAM++ speaker
encoder emits 192-d embeddings. The FiLM layers handle the dimension
adaptation:

```python
raw, emb = campplus_encoder(enrollment) # [B, 192]
emb = F.normalize(emb, p=2, dim=-1)    # [B, 192]
# FiLM internally: Linear(192, 256) → modulate 256-d features
```

## 4. Latency target: 40 ms

SpeakerBeam-SS ships in three latency variants:

| Variant | Frame | Lookahead | Algo latency | Relative SDR |
|---|---|---|---|---|
| Pure causal | 20 ms | 0 ms | 20 ms | baseline |
| **Low-latency** ⭐ | 20 ms | 20 ms | **40 ms** | +0.5–1 dB |
| Mid-latency | 20 ms | 100 ms | 120 ms | +0.8–1.3 dB |

Main branch targets the **low-latency (40 ms) variant**. Rationale:

- 40 ms is comfortably below the sub-50 ms budget for voice calls.
- The +0.5–1 dB quality uplift vs pure causal is worth the extra 20 ms.
- Going to 120 ms triples latency for only +0.3 dB more.

The 20 ms lookahead is architecturally a one-frame forward buffer: the
model receives a 20 ms chunk each call and emits a 20 ms clean chunk
that corresponds to the audio **one frame earlier**. See
`docs/onnx_contract.md` for the concrete stateful I/O interface.

## 5. Training objective

Loss = **SDR (non-scale-invariant) + multi-resolution STFT loss**,
plus a target-absent energy penalty for mixture-aware silence training.

### Why not SI-SDR

SI-SDR is scale-invariant, which seems convenient but creates two
degenerate minima:

- **Tiny-output** (Phase 0a): estimate = α·target with α→0 minimizes
  the L2 projection error, gives SI-SDR → +∞.
- **Pass-through** (when combined with a naive level penalty): output
  = mixture gives SI-SDR ≈ input SNR and matches mixture RMS, sitting
  in a local flat region that gradient descent does not escape.

Direct SDR (`−10·log10(‖target‖² / ‖estimate − target‖²)`) has neither
escape hatch: pass-through gives SDR ≈ input SNR (weak signal), tiny
output gives SDR ≈ 0 (weak signal), only real separation gives strong
negative loss.

### Why add MR-STFT

Direct SDR in the time domain treats all error uniformly, which can
leave residual frequency-domain artifacts (colored noise, comb
filtering) that are inaudible to the loss but audible to humans.
Multi-resolution STFT loss (magnitude L1 at several window sizes)
adds frequency-domain supervision without being scale-invariant.

Combined loss:
```
L_total = L_sdr + λ_stft · L_mr_stft
```
with λ_stft tuned on a held-out set. Initial value: λ_stft = 1.0.

### Mixture-aware silence training

Instead of v1's random-interval target-silencing (which caused
over-suppression when overdone), new training includes an **explicit
target-absent branch**:

- **~15 % of training mixtures**: target = silence (all zeros),
  mixture = interferer only.
- **Loss on absent samples**: simple energy penalty
  `L_absent = ‖estimate‖² / (‖mixture‖² + ε)` — drive output toward zero.
- **Loss on present samples**: `L_sdr + λ · L_mr_stft` as above.

The indicator is a **binary, per-sample label** (`target_present`),
not a continuous random process. Combined with a target-presence
detection head (a small auxiliary classifier on the bottleneck) the
model learns a clean "extract vs mute" decision rather than a fuzzy
"output quieter" heuristic.

The auxiliary presence head uses binary cross-entropy, weight ~0.1
of the main loss. Its output is also exported through the ONNX
interface so Rust-side code can use it for optional VAD-style gating.

## 6. Training data

- **AISHELL-1** (178 h, 400 spk) — primary clean Chinese speech
- **AISHELL-3** (~85 h, 218 spk) — additional speaker diversity
- **MAGICDATA** (~755 h, 1080 spk) — large clean Mandarin speaker expansion
- **MUSAN noise** (~3.6 GB subset) — additive non-speech noise augmentation

On-the-fly 2-speaker mixer produces training samples:

- 4-second segments of target + interferer from distinct AISHELL speakers
- SNR drawn from [-5, +5] dB
- ~15 % of samples swap target for silence (target-absent branch)
- Additive DNS4 noise at 10-25 dB SNR on the final mixture
- Optional room reverb (synthetic RIRs, RT60 0.08-0.25 s)

CN-Celeb and WenetSpeech remain optional later expansions if AISHELL +
MAGICDATA is still insufficient. Start with the clean three-corpus pool,
measure, then scale only if needed.

## 7. Training → deployment boundary

The **only** artifact crossing from Python (research) to Rust
(engineering) is **ONNX**. No PyTorch, no state dicts, no framework
coupling.

Two separate ONNX files:

1. `wulfenite_speaker_encoder.onnx` — the trained CAM++ speaker
   encoder, exported in eval mode for once-per-session enrollment.
2. `wulfenite_tse.onnx` — the trained SpeakerBeam-SS separator, with
   explicit state tensors for frame-by-frame stateful inference.

Full I/O specification: `docs/onnx_contract.md`.

## 8. Out of scope for this phase

- **Real-time Rust implementation** — `../rust/` stays as a minimal
  skeleton until the Python side produces a working ONNX.
- **Android / iOS / browser deployment** — ONNX Runtime has paths to
  all three, but none are targeted now.
- **Multi-speaker cocktail-party (5+ concurrent speakers)** —
  training is strict 2-speaker. Real inference may degrade on 3+
  active speakers. Document the limitation; do not try to fix.
- **Streaming quality fine-tuning** — model is trained whole-utterance
  style (stateful inference is faithful by architecture). If we later
  observe training-inference distribution mismatch, we revisit.
