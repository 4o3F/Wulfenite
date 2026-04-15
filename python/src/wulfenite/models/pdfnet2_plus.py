"""pDFNet2+ with TinyECAPA similarity refinement."""

from __future__ import annotations

from typing import Any, Literal

import torch
import torch.nn.functional as F
from torch import nn

from .pdfnet2 import PDfNet2
from .tiny_ecapa import TinyECAPA


class PDfNet2Plus(nn.Module):
    """Personalized DFNet2 with on-the-fly chunk similarity refinement."""

    def __init__(
        self,
        *,
        alpha_scale: float = 6.0,
        tiny_ecapa: TinyECAPA | None = None,
        conditioning_mode: Literal["causal", "offline"] = "causal",
        conditioning_window_seconds: float = 1.0,
        conditioning_update_interval_frames: int = 5,
        similarity_activation: Literal["sigmoid", "clamp"] = "sigmoid",
        similarity_threshold: float = 1.0 / 6.0,
        similarity_ema_decay: float = 0.8,
        conditioning_energy_threshold: float = 1e-4,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        if tiny_ecapa is None:
            raise ValueError(
                "PDfNet2Plus requires an explicit TinyECAPA instance. "
                "Load pretrained TinyECAPA weights in the caller and pass tiny_ecapa=..."
            )
        if conditioning_mode not in ("causal", "offline"):
            raise ValueError(f"Unsupported conditioning_mode: {conditioning_mode}")
        if conditioning_window_seconds <= 0.0:
            raise ValueError("conditioning_window_seconds must be positive.")
        if conditioning_update_interval_frames <= 0:
            raise ValueError("conditioning_update_interval_frames must be positive.")
        if similarity_activation not in ("sigmoid", "clamp"):
            raise ValueError(f"Unsupported similarity_activation: {similarity_activation}")
        if not 0.0 <= similarity_ema_decay < 1.0:
            raise ValueError("similarity_ema_decay must be in [0, 1).")
        if conditioning_energy_threshold < 0.0:
            raise ValueError("conditioning_energy_threshold must be non-negative.")
        self.alpha_scale = alpha_scale
        self.conditioning_mode = conditioning_mode
        self.conditioning_window_seconds = conditioning_window_seconds
        self.conditioning_update_interval_frames = conditioning_update_interval_frames
        self.similarity_activation = similarity_activation
        self.similarity_threshold = similarity_threshold
        self.similarity_ema_decay = similarity_ema_decay
        self.conditioning_energy_threshold = conditioning_energy_threshold
        self.pdfnet2 = PDfNet2(speaker_emb_dim=193, **kwargs)
        self.tiny_ecapa = tiny_ecapa
        self.tiny_ecapa.eval()
        for param in self.tiny_ecapa.parameters():
            param.requires_grad = False

    def train(self, mode: bool = True) -> "PDfNet2Plus":
        """Override to keep TinyECAPA frozen in eval mode."""
        super().train(mode)
        self.tiny_ecapa.eval()
        return self

    def similarity_to_gate(self, cosine: torch.Tensor) -> torch.Tensor:
        """Convert cosine similarity into a [0, 1] conditioning gate."""
        if self.similarity_activation == "clamp":
            return torch.clamp(self.alpha_scale * cosine, 0.0, 1.0)
        return torch.sigmoid(self.alpha_scale * (cosine - self.similarity_threshold))

    def _postprocess_similarity(
        self,
        gate: torch.Tensor,
        energies: torch.Tensor,
    ) -> torch.Tensor:
        """Apply silence hold and EMA smoothing to update-point gates."""
        outputs: list[torch.Tensor] = []
        prev: torch.Tensor | None = None
        zero = gate.new_zeros(gate.size(0))
        for idx in range(gate.size(1)):
            current = gate[:, idx]
            if self.conditioning_energy_threshold > 0.0:
                below = energies[:, idx] < self.conditioning_energy_threshold
                if prev is None:
                    current = torch.where(below, zero, current)
                else:
                    current = torch.where(below, prev, current)
            if prev is not None and self.similarity_ema_decay > 0.0:
                current = self.similarity_ema_decay * prev + (1.0 - self.similarity_ema_decay) * current
            outputs.append(current)
            prev = current
        return torch.stack(outputs, dim=1)

    def _conditioning_window_samples(self) -> int:
        samples = int(round(self.conditioning_window_seconds * self.pdfnet2.sample_rate))
        return max(1, samples)

    def _conditioning_update_frames(self, frames: int, device: torch.device) -> torch.Tensor:
        update_frames = torch.arange(
            0,
            frames,
            self.conditioning_update_interval_frames,
            device=device,
        )
        last_frame = torch.tensor([frames - 1], device=device)
        if update_frames.numel() == 0:
            return last_frame
        if int(update_frames[-1].item()) == frames - 1:
            return update_frames
        return torch.cat((update_frames, last_frame))

    def _expand_update_gates(
        self,
        update_gate: torch.Tensor,
        update_frames: torch.Tensor,
        frames: int,
    ) -> torch.Tensor:
        full_gate = update_gate.new_empty(update_gate.size(0), frames)
        starts = update_frames.tolist()
        for idx, start in enumerate(starts):
            end = starts[idx + 1] if idx + 1 < len(starts) else frames
            full_gate[:, start:end] = update_gate[:, idx].unsqueeze(1)
        return full_gate

    def _validate_conditioning_inputs(
        self,
        noisy_waveform: torch.Tensor,
        speaker_emb: torch.Tensor,
        frames: int,
    ) -> None:
        if noisy_waveform.dim() != 2:
            raise ValueError(f"noisy_waveform must be [B, T], got {tuple(noisy_waveform.shape)}")
        if speaker_emb.dim() != 2 or speaker_emb.size(-1) != 192:
            raise ValueError(
                f"speaker_emb must be [B, 192], got {tuple(speaker_emb.shape)}"
            )
        if noisy_waveform.size(0) != speaker_emb.size(0):
            raise ValueError(
                "noisy_waveform and speaker_emb must have the same batch size. "
                f"Got B={noisy_waveform.size(0)} and B={speaker_emb.size(0)}."
            )
        if frames <= 0:
            raise ValueError(f"frames must be positive, got {frames}")

    def _offline_conditioning(
        self,
        noisy_waveform: torch.Tensor,
        speaker_emb: torch.Tensor,
        frames: int,
    ) -> torch.Tensor:
        with torch.no_grad():
            chunk_emb = self.tiny_ecapa.forward_chunks(noisy_waveform)
        sim = F.cosine_similarity(
            speaker_emb.unsqueeze(-1),
            chunk_emb,
            dim=1,
        )
        sim = self.similarity_to_gate(sim)
        if sim.size(1) == 1:
            sim = sim.expand(-1, frames)
        else:
            sim = F.interpolate(sim.unsqueeze(1), size=frames, mode="linear", align_corners=True)
            sim = sim.squeeze(1)
        spk = speaker_emb.unsqueeze(1).expand(-1, frames, -1)
        return torch.cat((spk, sim.unsqueeze(-1)), dim=-1)

    def _causal_conditioning(
        self,
        noisy_waveform: torch.Tensor,
        speaker_emb: torch.Tensor,
        frames: int,
    ) -> torch.Tensor:
        core = self.pdfnet2
        window = self._conditioning_window_samples()
        total_samples = core.win_size + max(frames - 1, 0) * core.hop_size
        waveform = noisy_waveform[:, :total_samples]
        if waveform.size(-1) < total_samples:
            waveform = F.pad(waveform, (0, total_samples - waveform.size(-1)))
        pad_left = max(window - core.win_size, 0)
        start_offset = max(core.win_size - window, 0)
        windows = F.pad(waveform, (pad_left, 0))[:, start_offset:].unfold(-1, window, core.hop_size)
        update_frames = self._conditioning_update_frames(frames, noisy_waveform.device)
        update_windows = windows.index_select(1, update_frames)
        energies = update_windows.pow(2).mean(dim=-1)
        flat_windows = update_windows.reshape(-1, update_windows.size(-1))
        with torch.no_grad():
            chunk_emb = self.tiny_ecapa(flat_windows)
        chunk_emb = chunk_emb.view(noisy_waveform.size(0), update_frames.numel(), -1)
        update_gate = F.cosine_similarity(speaker_emb.unsqueeze(1), chunk_emb, dim=-1)
        update_gate = self.similarity_to_gate(update_gate)
        update_gate = self._postprocess_similarity(update_gate, energies)
        gate = self._expand_update_gates(update_gate, update_frames, frames)
        spk = speaker_emb.unsqueeze(1).expand(-1, frames, -1)
        return torch.cat((spk, gate.unsqueeze(-1)), dim=-1)

    def refine_conditioning(
        self,
        noisy_waveform: torch.Tensor,
        speaker_emb: torch.Tensor,
        frames: int,
        *,
        conditioning_mode: Literal["causal", "offline"] | None = None,
    ) -> torch.Tensor:
        self._validate_conditioning_inputs(noisy_waveform, speaker_emb, frames)
        mode = self.conditioning_mode if conditioning_mode is None else conditioning_mode
        if mode == "causal":
            return self._causal_conditioning(noisy_waveform, speaker_emb, frames)
        if mode == "offline":
            return self._offline_conditioning(noisy_waveform, speaker_emb, frames)
        raise ValueError(f"Unsupported conditioning_mode: {mode}")

    def forward(
        self,
        noisy_waveform: torch.Tensor,
        speaker_emb: torch.Tensor,
        spec: torch.Tensor | None = None,
        *,
        conditioning_mode: Literal["causal", "offline"] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if noisy_waveform.dim() == 1:
            noisy_waveform = noisy_waveform.unsqueeze(0)
        if spec is None:
            spec, _ = self.pdfnet2.waveform_to_spec(noisy_waveform)
        frames = spec.size(2)
        conditioning = self.refine_conditioning(
            noisy_waveform,
            speaker_emb,
            frames,
            conditioning_mode=conditioning_mode,
        )
        return self.pdfnet2(spec, conditioning)


__all__ = ["PDfNet2Plus"]
