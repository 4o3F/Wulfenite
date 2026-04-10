"""Wulfenite research crate.

Real-time target speaker extraction for Chinese audio.
Architecture: SpeakerBeam-SS (Sato et al., Interspeech 2024) — causal
Conv-TasNet + S4D state-space blocks + a learnable d-vector speaker encoder.

This package is the research side of the Wulfenite project. It handles
model definition, training, evaluation, and ONNX export. The engineering
side (real-time Rust deployment) lives under `../rust/` and consumes the
ONNX files exported here.

The only artifact crossing the Python -> Rust boundary is ONNX; see
`docs/onnx_contract.md` at the repo root for the interface specification.
"""

__version__ = "0.1.0"
