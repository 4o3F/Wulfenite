"""Target Speaker Extraction wrapper combining speaker encoder + separator.

This module is the single public entry point for training and
inference. It owns:

- a frozen :class:`wulfenite.models.campplus.CAMPPlus` speaker encoder
- a trainable :class:`wulfenite.models.dvector.LearnableDVector`
  speaker encoder
- a trainable :class:`wulfenite.models.speakerbeam_ss.SpeakerBeamSS`
  separator

and the enrollment-once / separate-many pattern required by the
architecture: the speaker encoder is invoked only when the enrollment
waveform changes, and its output is cached so each subsequent
separator call reuses the same L2-normalized embedding.

This matches the split described in ``docs/onnx_contract.md``, where
``cam_plus_chinese.onnx`` is exported separately from
``wulfenite_tse.onnx`` precisely because the two components have
different call frequencies (once per session vs every frame).
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from .campplus import CAMPPlus, encode_enrollment as campplus_encode_enrollment
from .campplus import load_campplus_cn_common
from .dvector import LearnableDVector, compute_fbank_batch
from .speakerbeam_ss import SpeakerBeamSS, SpeakerBeamSSConfig


class WulfeniteTSE(nn.Module):
    """End-to-end TSE model: speaker encoder + SpeakerBeam-SS.

    Intended usage in training:

        tse = WulfeniteTSE.from_campplus_checkpoint(
            "assets/campplus/campplus_cn_common.bin"
        )
        out = tse(mixture_wav, enrollment_wav)
        loss = combined_loss(out["clean"], target_wav, ...)

    Intended usage in inference (streaming, Python-side eval):

        tse.eval()
        emb = tse.encode_enrollment(enrollment_wav)   # once per session
        clean_chunk = tse.separate(mixture_chunk, emb)  # every frame

    The ONNX export lives in a separate module (not written yet) and
    will split the model into two ONNX graphs matching the contract.
    """

    def __init__(
        self,
        speaker_encoder: CAMPPlus | LearnableDVector,
        separator: SpeakerBeamSS | None = None,
        learnable_encoder: bool = False,
    ) -> None:
        super().__init__()
        self.speaker_encoder = speaker_encoder
        self.separator = separator or SpeakerBeamSS()
        self.learnable_encoder = learnable_encoder

        if not self.learnable_encoder:
            # Belt-and-braces: make sure the encoder is frozen even if
            # the caller forgot to freeze it before passing it in.
            self.speaker_encoder.eval()
            for p in self.speaker_encoder.parameters():
                p.requires_grad_(False)

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_campplus_checkpoint(
        cls,
        campplus_path: str | Path,
        separator_config: SpeakerBeamSSConfig | None = None,
        device: str | torch.device = "cpu",
    ) -> "WulfeniteTSE":
        """Build a TSE model with a pretrained CAM++ and a fresh separator.

        Args:
            campplus_path: path to the ``campplus_cn_common.bin``
                checkpoint (downloaded separately from ModelScope).
            separator_config: optional override for the SpeakerBeam-SS
                hyperparameters. Defaults to the 40 ms low-latency
                config.
            device: where to put the encoder while loading its weights;
                the full model can be moved afterwards via ``.to(...)``.
        """
        encoder = load_campplus_cn_common(campplus_path, device=device)
        separator = SpeakerBeamSS(separator_config)
        return cls(
            speaker_encoder=encoder,
            separator=separator,
            learnable_encoder=False,
        )

    @classmethod
    def from_learnable_dvector(
        cls,
        num_speakers: int,
        separator_config: SpeakerBeamSSConfig | None = None,
        dvector_kwargs: dict | None = None,
    ) -> "WulfeniteTSE":
        """Build a TSE model with a learnable d-vector encoder."""
        separator = SpeakerBeamSS(separator_config)
        encoder = LearnableDVector(
            num_speakers=num_speakers,
            emb_dim=separator.config.bottleneck_channels,
            **(dvector_kwargs or {}),
        )
        return cls(
            speaker_encoder=encoder,
            separator=separator,
            learnable_encoder=True,
        )

    # ------------------------------------------------------------------
    # Enrollment (once per session)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def encode_enrollment(self, waveform: torch.Tensor) -> torch.Tensor:
        """Compute the L2-normalized speaker embedding from raw audio.

        This helper is batch-aware: ``waveform`` may be ``[T]`` or
        ``[B, T]`` and the return shape will be ``[1, D]`` or ``[B, D]``.

        Args:
            waveform: ``[T]`` or ``[B, T]`` 16 kHz mono enrollment,
                3-10 seconds recommended.

        Returns:
            Unit-L2 speaker embeddings.
        """
        if self.learnable_encoder:
            return self.speaker_encoder.encode_enrollment(waveform)

        if waveform.dim() == 1:
            return campplus_encode_enrollment(self.speaker_encoder, waveform)
        if waveform.dim() != 2:
            raise ValueError(
                f"waveform must be [T] or [B, T], got {tuple(waveform.shape)}"
            )
        embs = [
            campplus_encode_enrollment(self.speaker_encoder, waveform[i])
            for i in range(waveform.size(0))
        ]
        return torch.cat(embs, dim=0)

    # ------------------------------------------------------------------
    # Separation (per-frame or whole-utterance)
    # ------------------------------------------------------------------

    def separate(
        self,
        mixture: torch.Tensor,
        speaker_embedding: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Run the separator with a pre-computed embedding.

        Args:
            mixture: ``[B, T]`` 16 kHz mixture.
            speaker_embedding: ``[B, 192]`` or ``[1, 192]`` L2-normalized
                CAM++ output; broadcast over the batch if ``B`` > 1 and
                the embedding has batch size 1.

        Returns:
            Same dict as :meth:`SpeakerBeamSS.forward` — keys
            ``"clean"`` and, when enabled, ``"presence_logit"``.
        """
        if speaker_embedding.size(0) == 1 and mixture.size(0) > 1:
            speaker_embedding = speaker_embedding.expand(mixture.size(0), -1)
        return self.separator(mixture, speaker_embedding)

    # ------------------------------------------------------------------
    # End-to-end training forward (embedding computed on the fly)
    # ------------------------------------------------------------------

    def forward(
        self,
        mixture: torch.Tensor,
        enrollment: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Convenience forward for training.

        Args:
            mixture: ``[B, T_mix]``.
            enrollment: ``[B, T_enr]`` or a single ``[T_enr]`` to
                broadcast across the batch.

        Returns:
            Separator output dict plus cached speaker-encoder outputs.
        """
        if enrollment.dim() == 1:
            enrollment = enrollment.unsqueeze(0)

        if self.learnable_encoder:
            fbank = compute_fbank_batch(enrollment)
            raw_emb, norm_emb, logits = self.speaker_encoder(fbank)
            if norm_emb.size(0) == 1 and mixture.size(0) > 1:
                raw_emb = raw_emb.expand(mixture.size(0), -1)
                norm_emb = norm_emb.expand(mixture.size(0), -1)
                if logits is not None:
                    logits = logits.expand(mixture.size(0), -1)
            out = self.separate(mixture, norm_emb)
            out["embedding"] = norm_emb
            out["raw_embedding"] = raw_emb
            out["speaker_logits"] = logits
            return out

        emb = self.encode_enrollment(enrollment)
        out = self.separate(mixture, emb)
        out["embedding"] = emb
        return out
