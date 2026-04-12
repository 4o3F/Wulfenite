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
- **Paper-faithful defaults**: encoder channels `N = 2048`,
  bottleneck `B = 256`, hidden `H = 512`, S4D state `D = 32`.
- **Separator stack**:
  - Initial projection: channel-wise LayerNorm on the encoder features,
    then `1x1 Conv(N -> B)`.
  - Pre-fusion stage: `R1 = 3` repetitions.
  - Post-fusion stage: `R2 = 1` repetitions.
  - Each repetition contains `X = 2` sequential sub-blocks with
    exponentially increasing dilation `d = 2^(x - 1)`.
  - Each sub-block is `TCN(d) -> S4D`.
- **S4D block**: pre-norm residual structure with two branches:
  - sequence mixing: `cLN -> S4D -> GELU -> 1x1 Conv(B -> 2B) -> GLU -> residual add`
  - channel mixing: `cLN -> 1x1 Conv(B -> 4B) -> GELU -> 1x1 Conv(4B -> B) -> residual add`
- **Frontend**: learned 1-D conv encoder, kernel size 320, hop 160
  (20 ms at 16 kHz), followed by ReLU.
- **Mask generator**: `1x1 Conv(B -> N)` followed by ReLU. The resulting
  non-negative mask is applied to the encoder features before decoding.
- **Causality**: **causal by default.** Global LayerNorm replaced by
  channel-wise LayerNorm; convolutions are strictly causal. S4D has a
  recurrent form that supports stateful inference across frames.

**Speaker encoder**:

- **Single path**: a fine-tuned CAM++ encoder whose L2-normalized
  embedding is native 192-d. The separator projects the 192-d speaker
  embedding into its 256-d bottleneck space.

The speaker encoder still runs exactly ONCE per session, not per frame.
TSE's core assumption is that the target speaker is fixed for the
duration of a session (one streamer per broadcast, one caller per phone
call, one gamer per match). The enrollment is processed once at session
startup, the resulting speaker embedding is cached, and every
subsequent frame of the real-time separator is conditioned on that same
cached embedding. This keeps the encoder in the setup budget rather than
the steady-state per-frame budget — see
`docs/onnx_contract.md` for the concrete two-file split.

### Speaker embedding fusion

SpeakerBeam-SS modulates the separator **between** the pre-fusion and
post-fusion stages.

1. Run the encoder and bottleneck projection.
2. Run the `R1` pre-fusion stage.
3. Project the target-speaker embedding from 192-d into the bottleneck
   dimension `B = 256`.
4. Broadcast that projected vector across time and apply element-wise
   multiplication to the intermediate separator features.
5. Run the `R2` post-fusion stage and generate the mask.

In the current repo this uses the CAM++ enrollment embedding:

```python
raw, emb = campplus_encoder(enrollment)   # [B, 192]
emb = F.normalize(emb, p=2, dim=-1)      # [B, 192]
speaker_scale = Linear(192, 256)(emb)    # [B, 256]
feat = feat * speaker_scale.unsqueeze(-1)
```

The paper's core architectural point is the **position** of this
speaker modulation: it is applied after the first separator stage, not
immediately after the encoder bottleneck.

## 4. Latency target

The paper reports pure-causal and lookahead variants. The current repo
implementation aligns the **separator topology** to Figure 1 but keeps
the runtime path **strictly causal**:

- frame size: 20 ms
- encoder hop: 10 ms
- separator lookahead: 0 ms
- algorithmic latency from the separator itself: 20 ms plus any caller
  side buffering

The paper's 40 ms and 120 ms lookahead variants require non-causal
changes in the early convolution blocks. Those lookahead modifications
are not implemented in the current code path.

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
not a continuous random process.

An auxiliary target-presence head remains available as an **optional
repo extension** for experiments, but it is disabled by default in the
paper-aligned architecture path.

## 6. Training data

- **AISHELL-1** (178 h, 400 spk) — primary clean Chinese speech
- **AISHELL-3** (~85 h, 218 spk) — additional speaker diversity
- **MAGICDATA** (~755 h, 1080 spk) — large clean Mandarin speaker expansion
- **MUSAN noise** (~3.6 GB subset) — additive non-speech noise augmentation

On-the-fly clip composer produces training samples:

- 4-second clips built from 4-8 frame-aligned conversational events
- Three clip families: multi-turn target-present, overlap-heavy, and
  hard-negative target-absent
- Event types: target-only, nontarget-only, overlap, and background-only
- Clip-level acoustic consistency: one room / RIR choice per mixture,
  one background-noise bed, and slow gain drift instead of abrupt
  per-source level jumps
- Framewise labels at the encoder stride (10 ms / 160 samples):
  `target_active_frames`, `nontarget_active_frames`, and `overlap_frames`
- Additive noise at 10-25 dB SNR on the final mixture
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
