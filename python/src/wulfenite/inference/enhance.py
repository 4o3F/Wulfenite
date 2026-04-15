"""Batch and streaming inference helpers for pDFNet2."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

import torch
import torch.nn.functional as F

from wulfenite.models import DfNet, DfNetStreamState, PDfNet2, PDfNet2Plus, SpeakerEncoder


@dataclass
class _StreamingState:
    input_buffer: torch.Tensor
    output_buffer: torch.Tensor
    norm_buffer: torch.Tensor
    recent_audio: torch.Tensor
    model_state: DfNetStreamState
    total_input_samples: int = 0
    total_output_samples: int = 0
    conditioning_frame_idx: int = 0
    last_similarity: torch.Tensor | None = None
    last_conditioning: torch.Tensor | None = None


class Enhancer:
    """Runtime wrapper around the pDFNet2 family."""

    def __init__(
        self,
        model: DfNet | PDfNet2 | PDfNet2Plus,
        *,
        enrollment_encoder: SpeakerEncoder | None = None,
        device: str | None = None,
    ) -> None:
        self.model = model
        self.device = torch.device(device) if device is not None else (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.model.to(self.device)
        self.model.eval()
        self.enrollment_encoder = enrollment_encoder
        if self.enrollment_encoder is not None:
            self.enrollment_encoder.to(self.device)
            self.enrollment_encoder.eval()
        self._cached_speaker_emb: torch.Tensor | None = None
        self._stream_state: _StreamingState | None = None
        self._stream_window: torch.Tensor
        self._stream_window_sq: torch.Tensor

    @property
    def _core_model(self) -> DfNet:
        if isinstance(self.model, PDfNet2Plus):
            return self.model.pdfnet2
        return self.model

    def _compute_speaker_embedding(
        self,
        enrollment_waveform: torch.Tensor,
    ) -> torch.Tensor:
        if self.enrollment_encoder is None:
            raise RuntimeError(
                "Enhancer.enroll() requires an enrollment_encoder for personalized models."
            )
        with torch.no_grad():
            emb = self.enrollment_encoder(enrollment_waveform.to(self.device))
        return emb

    def enroll(self, enrollment_waveform: torch.Tensor) -> torch.Tensor:
        """Extract and cache the enrollment embedding."""
        if enrollment_waveform.dim() == 1:
            enrollment_waveform = enrollment_waveform.unsqueeze(0)
        self._cached_speaker_emb = self._compute_speaker_embedding(enrollment_waveform)
        return self._cached_speaker_emb

    def _resolve_embedding(
        self,
        speaker_emb: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if isinstance(self.model, DfNet) and not isinstance(self.model, PDfNet2):
            return None
        if speaker_emb is not None:
            return speaker_emb.to(self.device)
        if self._cached_speaker_emb is not None:
            return self._cached_speaker_emb
        raise RuntimeError("No speaker embedding provided or enrolled.")

    def enhance(
        self,
        waveform: torch.Tensor,
        *,
        speaker_emb: torch.Tensor | None = None,
        conditioning_mode: Literal["causal", "offline"] | None = None,
    ) -> torch.Tensor:
        """Enhance a full waveform in batch mode."""
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        waveform = waveform.to(self.device)
        resolved_emb = self._resolve_embedding(speaker_emb)
        with torch.no_grad():
            if isinstance(self.model, PDfNet2Plus):
                assert resolved_emb is not None
                enhanced_spec, _gains, _lsnr, _alpha = self.model(
                    waveform,
                    resolved_emb,
                    conditioning_mode=conditioning_mode,
                )
                enhanced = self.model.pdfnet2.spec_to_waveform(
                    enhanced_spec,
                    length=waveform.size(-1),
                )
            elif isinstance(self.model, PDfNet2):
                spec, _ = self.model.waveform_to_spec(waveform)
                enhanced_spec, _gains, _lsnr, _alpha = self.model(spec, resolved_emb)
                enhanced = self.model.spec_to_waveform(enhanced_spec, length=waveform.size(-1))
            else:
                spec, _ = self.model.waveform_to_spec(waveform)
                enhanced_spec, _gains, _lsnr, _alpha = self.model(spec)
                enhanced = self.model.spec_to_waveform(enhanced_spec, length=waveform.size(-1))
        return enhanced

    def reset_stream(self) -> None:
        self._stream_state = None

    def _init_stream(self, batch_size: int, dtype: torch.dtype) -> None:
        core = self._core_model
        window = core.window.to(self.device, dtype=dtype)
        self._stream_state = _StreamingState(
            input_buffer=torch.zeros(batch_size, 0, device=self.device, dtype=dtype),
            output_buffer=torch.zeros(batch_size, core.win_size, device=self.device, dtype=dtype),
            norm_buffer=torch.zeros(batch_size, core.win_size, device=self.device, dtype=dtype),
            recent_audio=torch.zeros(batch_size, 0, device=self.device, dtype=dtype),
            model_state=core.init_stream_state(batch_size, device=self.device, dtype=dtype),
        )
        self._stream_window = window
        self._stream_window_sq = window.pow(2)

    def _conditioning_window_samples(self) -> int:
        if isinstance(self.model, PDfNet2Plus):
            samples = int(round(self.model.conditioning_window_seconds * self._core_model.sample_rate))
            return max(1, samples)
        return self._core_model.sample_rate

    def _append_recent_audio(self, state: _StreamingState, frame: torch.Tensor) -> None:
        if not isinstance(self.model, PDfNet2Plus):
            return
        core = self._core_model
        new_audio = frame if state.conditioning_frame_idx == 0 else frame[:, -core.hop_size :]
        state.recent_audio = torch.cat((state.recent_audio, new_audio), dim=-1)
        max_recent = self._conditioning_window_samples()
        if state.recent_audio.size(-1) > max_recent:
            state.recent_audio = state.recent_audio[:, -max_recent:]

    def _frame_conditioning(
        self,
        state: _StreamingState,
        recent_audio: torch.Tensor,
        speaker_emb: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if isinstance(self.model, PDfNet2Plus):
            if speaker_emb is None:
                raise RuntimeError("pDFNet2+ streaming requires a speaker embedding.")
            update_interval = self.model.conditioning_update_interval_frames
            if (
                state.last_conditioning is not None
                and state.conditioning_frame_idx % update_interval != 0
            ):
                state.conditioning_frame_idx += 1
                return state.last_conditioning

            max_recent = self._conditioning_window_samples()
            if recent_audio.size(-1) < max_recent:
                padded_recent = F.pad(recent_audio, (max_recent - recent_audio.size(-1), 0))
            else:
                padded_recent = recent_audio[:, -max_recent:]

            prev_similarity = state.last_similarity
            energy = padded_recent.pow(2).mean(dim=-1)
            below = energy < self.model.conditioning_energy_threshold
            if torch.all(below):
                sim = (
                    prev_similarity
                    if prev_similarity is not None
                    else speaker_emb.new_zeros(speaker_emb.size(0))
                )
            else:
                with torch.no_grad():
                    chunk_emb = self.model.tiny_ecapa(padded_recent)
                current = self.model.similarity_to_gate(
                    torch.nn.functional.cosine_similarity(speaker_emb, chunk_emb, dim=-1)
                )
                if self.model.conditioning_energy_threshold > 0.0:
                    held = (
                        prev_similarity
                        if prev_similarity is not None
                        else current.new_zeros(current.size(0))
                    )
                    current = torch.where(below, held, current)
                sim = current

            if prev_similarity is not None and self.model.similarity_ema_decay > 0.0:
                sim = self.model.similarity_ema_decay * prev_similarity + (
                    1.0 - self.model.similarity_ema_decay
                ) * sim
            state.last_similarity = sim
            state.last_conditioning = torch.cat((speaker_emb, sim.unsqueeze(-1)), dim=-1)
            state.conditioning_frame_idx += 1
            return state.last_conditioning
        return speaker_emb

    def _process_stream_frame(
        self,
        state: _StreamingState,
        frame: torch.Tensor,
        speaker_emb: torch.Tensor | None,
    ) -> torch.Tensor:
        core = self._core_model
        self._append_recent_audio(state, frame)
        spec = torch.fft.rfft(frame * self._stream_window, n=core.fft_size)
        spec_frame = torch.view_as_real(spec).unsqueeze(1).unsqueeze(2)
        conditioning = self._frame_conditioning(state, state.recent_audio, speaker_emb)
        with torch.no_grad():
            enhanced_spec, _gains, _lsnr, _alpha, state.model_state = core.stream_step(
                spec_frame,
                conditioning,
                state.model_state,
            )
        frame_out = torch.fft.irfft(
            torch.view_as_complex(enhanced_spec.squeeze(1).squeeze(1).contiguous()),
            n=core.fft_size,
        )[:, : core.win_size]
        frame_out = frame_out * self._stream_window
        state.output_buffer[:, : core.win_size] += frame_out
        state.norm_buffer[:, : core.win_size] += self._stream_window_sq
        emitted = state.output_buffer[:, : core.hop_size] / state.norm_buffer[:, : core.hop_size].clamp_min(1e-8)
        state.total_output_samples += emitted.size(-1)
        state.output_buffer = torch.cat(
            (
                state.output_buffer[:, core.hop_size :],
                torch.zeros(
                    state.output_buffer.size(0),
                    core.hop_size,
                    device=self.device,
                    dtype=state.output_buffer.dtype,
                ),
            ),
            dim=-1,
        )
        state.norm_buffer = torch.cat(
            (
                state.norm_buffer[:, core.hop_size :],
                torch.zeros(
                    state.norm_buffer.size(0),
                    core.hop_size,
                    device=self.device,
                    dtype=state.norm_buffer.dtype,
                ),
            ),
            dim=-1,
        )
        return emitted

    def enhance_streaming(
        self,
        chunk: torch.Tensor,
        *,
        speaker_emb: torch.Tensor | None = None,
        finalize: bool = False,
    ) -> torch.Tensor:
        """Enhance one streaming chunk with persistent model state."""
        if chunk.dim() == 1:
            chunk = chunk.unsqueeze(0)
        if chunk.dim() != 2:
            raise ValueError(f"chunk must be [T] or [B, T], got {tuple(chunk.shape)}")
        chunk = chunk.to(self.device)
        resolved_emb = self._resolve_embedding(speaker_emb)

        if self._stream_state is None:
            self._init_stream(chunk.size(0), chunk.dtype)
        state = self._stream_state
        if state is None:
            raise RuntimeError("stream state was not initialized")

        core = self._core_model
        state.total_input_samples += chunk.size(-1)
        state.input_buffer = torch.cat((state.input_buffer, chunk), dim=-1)

        outputs: list[torch.Tensor] = []
        while state.input_buffer.size(-1) >= core.win_size:
            frame = state.input_buffer[:, : core.win_size]
            state.input_buffer = state.input_buffer[:, core.hop_size :]
            outputs.append(self._process_stream_frame(state, frame, resolved_emb))

        if finalize:
            if state.total_input_samples <= core.fft_size:
                desired_frames = 1
            else:
                desired_frames = (
                    math.ceil((state.total_input_samples - core.fft_size) / core.hop_size) + 1
                )
            frames_processed = state.total_output_samples // core.hop_size
            while frames_processed < desired_frames:
                if state.input_buffer.size(-1) < core.win_size:
                    state.input_buffer = F.pad(
                        state.input_buffer,
                        (0, core.win_size - state.input_buffer.size(-1)),
                    )
                frame = state.input_buffer[:, : core.win_size]
                state.input_buffer = state.input_buffer[:, core.hop_size :]
                outputs.append(self._process_stream_frame(state, frame, resolved_emb))
                frames_processed += 1
            remaining = state.total_input_samples - state.total_output_samples
            if remaining > 0:
                tail = state.output_buffer[:, :remaining] / state.norm_buffer[:, :remaining].clamp_min(1e-8)
                outputs.append(tail)
            self.reset_stream()

        if not outputs:
            return torch.zeros(chunk.size(0), 0, device=self.device, dtype=chunk.dtype)
        return torch.cat(outputs, dim=-1)


__all__ = ["Enhancer"]
