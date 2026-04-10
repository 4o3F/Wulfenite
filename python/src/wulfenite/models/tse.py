"""Target Speaker Extraction wrapper combining speaker encoder + separator.

This module is the single public entry point for training and
inference. It owns:

- a trainable :class:`wulfenite.models.dvector.LearnableDVector`
  speaker encoder
- a trainable :class:`wulfenite.models.speakerbeam_ss.SpeakerBeamSS`
  separator

and the enrollment-once / separate-many pattern required by the
architecture: the speaker encoder is invoked only when the enrollment
waveform changes, and its output is cached so each subsequent
separator call reuses the same L2-normalized embedding.
"""

from __future__ import annotations

import torch
from torch import nn

from .dvector import LearnableDVector, compute_fbank_batch
from .speakerbeam_ss import SpeakerBeamSS, SpeakerBeamSSConfig


class WulfeniteTSE(nn.Module):
    """End-to-end TSE model: speaker encoder + SpeakerBeam-SS.

    Intended usage in training:

        tse = WulfeniteTSE.from_learnable_dvector(num_speakers=512)
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
        speaker_encoder: LearnableDVector,
        separator: SpeakerBeamSS | None = None,
    ) -> None:
        super().__init__()
        self.speaker_encoder = speaker_encoder
        self.separator = separator or SpeakerBeamSS()

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_learnable_dvector(
        cls,
        num_speakers: int | None,
        separator_config: SpeakerBeamSSConfig | None = None,
        dvector_kwargs: dict | None = None,
    ) -> "WulfeniteTSE":
        """Build a TSE model with a learnable d-vector encoder.

        When ``num_speakers`` is ``None`` the encoder is built without
        the auxiliary classifier head, which is useful for inference.
        """
        separator = SpeakerBeamSS(separator_config)
        encoder = LearnableDVector(
            num_speakers=num_speakers,
            emb_dim=separator.config.bottleneck_channels,
            **(dvector_kwargs or {}),
        )
        return cls(
            speaker_encoder=encoder,
            separator=separator,
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
        return self.speaker_encoder.encode_enrollment(waveform)

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
            speaker_embedding: pre-computed separator-space embeddings
                ``[B, bottleneck_channels]`` / ``[1, bottleneck_channels]``.

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
