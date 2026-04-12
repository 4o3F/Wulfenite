# Wulfenite — ONNX Interface Contract

This document is the **only** interface specification between the
Python research crate and the Rust engineering crate. Anything not
documented here must not be assumed by either side.

**Versioning**: the contract is versioned with a single integer stored
as an ONNX metadata field `wulfenite_contract_version`. The initial
version is **1**. Any breaking change increments this integer;
backward-compatible additions keep the version unchanged.

---

## Files

The model is split into **two ONNX files** with distinct lifecycles:

| File | Purpose | Lifecycle |
|---|---|---|
| `wulfenite_speaker_encoder.onnx` | Speaker encoder — runs once per session | Exported in eval mode from the trained checkpoint |
| `wulfenite_tse.onnx` | Target speaker extractor — runs every audio frame | Exported after each training run |

Both files target **ONNX opset 17** (the minimum opset that supports
`torch.stft` and a broad set of state-space ops). The Rust side uses
[`ort`](https://crates.io/crates/ort) with the CPU execution provider.

Expected on disk (under `rust/assets/` or similar at deploy time):

```
<deployment>/
├── wulfenite_speaker_encoder.onnx
└── wulfenite_tse.onnx        # ≈ 8 MB
```

---

## File 1: `wulfenite_speaker_encoder.onnx`

The trained CAM++ speaker encoder. This file contains both the
FBank extraction and the speaker encoder network; the Rust side sees
raw audio in and a speaker embedding out.

### Inputs

| Name | dtype | Shape | Description |
|---|---|---|---|
| `enrollment_audio` | float32 | `[1, T_enr]` | 16 kHz mono enrollment audio. `T_enr` is variable (dynamic axis). Recommended length: 3–10 s. Expected amplitude range `[-1, 1]`. No mean subtraction required by the caller. |

### Outputs

| Name | dtype | Shape | Description |
|---|---|---|---|
| `speaker_embedding` | float32 | `[1, 192]` | Already L2-normalized CAM++ speaker embedding. Feed it directly into `wulfenite_tse.onnx`. |

### Semantics

- Runs once per session at startup (or whenever the user swaps enrollment).
- Output is deterministic for a given enrollment waveform; no state.
- The embedded FBank extraction uses 80 mel bins, 25 ms window, 10 ms hop,
  dither = 0, utterance-level mean normalization — matches the training path.

### Internal structure (not part of the contract, but documented for debugging)

```
enrollment_audio
  ↓ Kaldi FBank (n_mels=80, dither=0, utt mean norm)
  ↓ CAMPPlus.forward
  ↓ raw_embedding [1, 192]
  ↓ L2 normalize (torch.nn.functional.normalize(..., p=2, dim=-1))
  ↓ speaker_embedding [1, 192]
```

---

## File 2: `wulfenite_tse.onnx`

The trained SpeakerBeam-SS separator. This file is **stateful** —
callers hold opaque state tensors across frames and feed them back on
each call. The Rust side is responsible for state lifetime; the model
is pure.

### Constants baked into the model

| Constant | Value | Meaning |
|---|---|---|
| Sample rate | 16000 Hz | Inputs/outputs must be 16 kHz |
| Frame size | **320 samples** (20 ms) | One call processes exactly this many new samples |
| Algorithmic latency | **40 ms** | Output at call N corresponds to input from call N-1 |
| Number of state tensors | **(to be finalized at export time, documented in ONNX metadata `wulfenite_num_states`)** | Opaque; Rust side iterates by name |

### Inputs

| Name | dtype | Shape | Description |
|---|---|---|---|
| `mixture_chunk` | float32 | `[1, 320]` | The **latest** 20 ms of microphone audio, 16 kHz mono, amplitude `[-1, 1]`. Never shorter, never longer. If the caller has no more audio, pad with zeros to 320 samples and run a final call to flush. |
| `speaker_embedding` | float32 | `[1, 256]` | Separator-space speaker embedding from `wulfenite_speaker_encoder.onnx`. |
| `state_in_0`, `state_in_1`, … | float32 | Various (fixed at export time) | Opaque state tensors. On the first call of a session, pass zero tensors of the correct shape. Shapes are documented in ONNX metadata `wulfenite_state_shape_N` for each N. |

### Outputs

| Name | dtype | Shape | Description |
|---|---|---|---|
| `clean_chunk` | float32 | `[1, 320]` | 20 ms of clean target speaker audio, corresponding to the audio **that was received on the previous call**. For the first call, the output is silence (zeros); for normal operation, feed straight to the speaker. |
| `state_out_0`, `state_out_1`, … | float32 | Match `state_in_N` | Updated state tensors. Store each and pass as `state_in_N` on the next call. |
| `target_present_logit` | float32 | `[1]` | Pre-sigmoid logit for "target speaker is active in this 20 ms". Apply `sigmoid()` to get a probability in `[0, 1]`. Threshold at 0.5 for a binary decision. Optional — the caller may ignore it. |

### Call semantics — the 40 ms latency pattern

Because the model looks one frame ahead, the output of call N
corresponds to the input of call N-1. Concretely:

```
Wall time:   0ms      20ms      40ms      60ms      80ms
             │         │         │         │         │
Call:      call1    call2    call3    call4    call5
Input:     chunk1   chunk2   chunk3   chunk4   chunk5   (audio captured in previous 20 ms)
Output:    silence  clean1   clean2   clean3   clean4   (clean version of chunk1, chunk2, ...)
```

- At wall time 0: the caller has 20 ms of audio, calls the model. The
  output `clean1` is silence (zeros) because the model has not yet
  accumulated enough future context.
- At wall time 20 ms: the caller has the next 20 ms (`chunk2`), calls
  the model. The output is now `clean1` — the cleaned version of the
  audio from 0–20 ms.
- Steady state: every 20 ms in, 20 ms out, with a constant 20 ms
  additional delay on top of the frame size. Total 40 ms latency
  from audio capture to clean emission.

### State lifetime and shape discovery

The set of state tensors is determined at export time and encoded in
the ONNX graph. The Rust side **must not hardcode** the number or
shapes of states — both should be discovered via ONNX metadata:

```
wulfenite_num_states      : i64       → e.g., 8
wulfenite_state_shape_0   : i64[rank] → e.g., [1, 256, 16]
wulfenite_state_shape_1   : i64[rank] → ...
...
```

On session start, the Rust code allocates `num_states` zero tensors of
the specified shapes and uses them as the initial `state_in_N`
inputs. On each call, the returned `state_out_N` replaces the stored
`state_in_N`.

### Error handling

| Caller mistake | ONNX behavior | Rust caller responsibility |
|---|---|---|
| `mixture_chunk` not `[1, 320]` | ORT shape error | Assemble exactly one frame before calling |
| `speaker_embedding` not `[1, 256]` | ORT shape error | Feed the normalized 256-d encoder output |
| Wrong sample rate | Silent garbage output | Resample to 16 kHz before the ring buffer |
| Forgot to pass state | ORT missing-input error | Initialize once, feed back every call |
| Stale state from previous session | Garbage for first ~100 ms | Reset state to zero on session change |

### What is **not** exported

- The mixer used in training (target/interferer mixing, reverb,
  silence injection) — training-only.
- The loss function — training-only.
- Intermediate activations, debug hooks — not in the released model.
- Configurable dropout / BN training-mode — exported in eval mode only.

---

## Metadata schema (ONNX graph attributes)

Both files include the following string/int metadata attributes on
the graph, readable from `ort::Session::metadata()` in Rust:

| Key | Type | Required in | Value example |
|---|---|---|---|
| `wulfenite_contract_version` | i64 | both | 1 |
| `wulfenite_sample_rate` | i64 | both | 16000 |
| `wulfenite_frame_size` | i64 | tse only | 320 |
| `wulfenite_algorithmic_latency_ms` | f32 | tse only | 40.0 |
| `wulfenite_num_states` | i64 | tse only | e.g., 8 |
| `wulfenite_state_shape_N` | i64[] | tse only, one per state | e.g., [1, 256, 16] |
| `wulfenite_trained_on` | str | tse only | "AISHELL1+AISHELL3+DNS4" |
| `wulfenite_checkpoint_sha256` | str | tse only | 64-char hex |
| `wulfenite_export_date` | str | both | ISO 8601 date |

The Rust loader should **fail-fast** if `wulfenite_contract_version`
is missing or does not match the version it was built against.

---

## Rust pseudocode (not part of contract, illustrative)

```rust
// one-time setup
let speaker_encoder =
    ort::Session::builder()?.commit_from_file("wulfenite_speaker_encoder.onnx")?;
let tse = ort::Session::builder()?.commit_from_file("wulfenite_tse.onnx")?;

let contract_version = tse.metadata()?.custom("wulfenite_contract_version")?;
assert_eq!(contract_version, "1");

// enrollment (once per session)
let enrollment_wav: Vec<f32> = load_wav_16k_mono("enrollment.wav")?;
let emb_outputs = speaker_encoder.run(ort::inputs![
    "enrollment_audio" => TensorRef::from_array_view(&enrollment_wav)?
])?;
let speaker_embedding: ArrayD<f32> = emb_outputs["speaker_embedding"].try_extract_tensor()?;

// initialize states to zeros (shapes from metadata)
let mut states: Vec<Array<f32>> = allocate_zero_states(&tse)?;

// streaming loop
loop {
    let chunk = ring_buffer.pop_exactly_320_samples();
    let inputs = assemble_tse_inputs(&chunk, &speaker_embedding, &states);
    let outputs = tse.run(inputs)?;

    let clean: ArrayD<f32> = outputs["clean_chunk"].try_extract_tensor()?;
    update_states_in_place(&mut states, &outputs);
    speaker_ring_buffer.push(&clean);
}
```

---

## Open questions (to resolve before exporting v1 of the file)

1. **Should state tensors be combined into a single `state_in` /
   `state_out` tensor of shape `[N]`?** Pro: simpler Rust loop. Con:
   loses per-tensor shape info, forces packing. Current plan:
   multiple named state tensors. Revisit if Rust allocation gets
   ugly.
2. **Should `target_present_logit` be a float or skipped entirely?**
   Current plan: include it as an optional output; caller may ignore.
3. **How to handle enrollment drift** (speaker changes tone/volume
   over time)? Out of scope for v1; the ONNX contract pins
   enrollment to session start.
