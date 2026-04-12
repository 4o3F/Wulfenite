"""SpeakerBeam-SS separator.

Based on Sato, Moriya, Mimura, Horiguchi, Ochiai, Ashihara, Ando,
Shinayama, Delcroix, "SpeakerBeam-SS: Real-time Target Speaker
Extraction with Lightweight Conv-TasNet and State Space Modeling",
Interspeech 2024 (`arXiv:2407.01857`).

Architecture at a glance:

    waveform [B, T]
        ↓ learned encoder (Conv1d kernel=320, hop=160, causal)
    encoded features [B, N_enc, L]
        ↓ bottleneck conv (Conv1d 1×1)
    bottleneck [B, B_dim, L]
        ↓ FiLM speaker conditioning  (affine modulation from L2-normed embedding)
    conditioned [B, B_dim, L]
        ↓ separator stack
          repeat x2:
            TCN(d=1), TCN(d=2), TCN(d=4), S4D+FC, TCN(d=8)
    refined [B, B_dim, L]
        ↓ mask head (Conv1d 1×1 → sigmoid)  +  presence head (Linear → logit)
    mask [B, N_enc, L]
        ↓ masked encoded (element-wise multiply)
    masked [B, N_enc, L]
        ↓ learned decoder (ConvTranspose1d kernel=320, hop=160)
    clean waveform [B, T]

Default config matches the paper-faithful Phase 3 recipe:

- encoder channels ``N = 4096``
- bottleneck channels ``B = 256``
- hidden channels ``H = 512``
- S4D state dim ``D = 32``
- block topology ``2 × (3 TCN + 1 S4D + 1 TCN)``

The model exposes both ``forward`` (whole-sequence training) and
``streaming_step`` (stateful inference). In eval mode the two must agree
numerically; the per-module streaming state carries S4D modes and
causal-conv left-context buffers.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .layers import CausalConv1d, ChannelwiseLayerNorm
from .s4d import S4D


# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------


@dataclass
class SpeakerBeamSSConfig:
    """Hyperparameter bundle for SpeakerBeam-SS.

    The defaults match the paper-faithful low-latency (40 ms) variant.
    """

    sample_rate: int = 16000

    # Encoder / decoder (learned 1-D conv, matches SpeakerBeam-SS paper).
    enc_kernel_size: int = 320   # 20 ms @ 16 kHz
    enc_stride: int = 160        # 10 ms hop
    enc_channels: int = 4096

    # Separator bottleneck.
    bottleneck_channels: int = 256
    speaker_embed_dim: int = 192

    # TCN+S4D block stack.
    num_repeats: int = 2         # "SS-2" in the paper
    r1_blocks: int = 3           # TCN blocks before S4D
    r2_blocks: int = 1           # TCN blocks after S4D
    conv_kernel_size: int = 3
    hidden_channels: int = 512   # "H" in the paper

    # S4D block.
    s4d_state_dim: int = 32

    # Auxiliary heads.
    target_presence_head: bool = True


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class TCNBlock(nn.Module):
    """One temporal block: causal separable conv + cLN + PReLU + 1×1.

    The structure follows Conv-TasNet's residual block with causal
    padding. The S4D block is a sibling (see ``TCNS4DBlock``).
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        kernel_size: int,
        dilation: int,
    ) -> None:
        super().__init__()
        self.pointwise_in = nn.Conv1d(in_channels, hidden_channels, 1)
        self.prelu_in = nn.PReLU(hidden_channels)
        self.norm_in = ChannelwiseLayerNorm(hidden_channels)

        self.depthwise = CausalConv1d(
            hidden_channels, hidden_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            groups=hidden_channels,
        )
        self.prelu_mid = nn.PReLU(hidden_channels)
        self.norm_mid = ChannelwiseLayerNorm(hidden_channels)

        self.pointwise_out = nn.Conv1d(hidden_channels, in_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Residual forward.

        Args:
            x: ``[B, C_in, T]``.

        Returns:
            ``[B, C_in, T]`` with the TCN block's residual update added.
        """
        y = self.pointwise_in(x)
        y = self.norm_in(self.prelu_in(y))
        y = self.depthwise(y)
        y = self.norm_mid(self.prelu_mid(y))
        y = self.pointwise_out(y)
        return x + y

    def zero_state(self, batch_size: int, device: torch.device,
                   dtype: torch.dtype = torch.float32):
        """Streaming state = depthwise conv left-context buffer."""
        return self.depthwise.zero_state(batch_size, device, dtype)

    def forward_step(
        self,
        x_new: torch.Tensor,
        state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Streaming forward for one or more new timesteps."""
        y = self.pointwise_in(x_new)
        y = self.norm_in(self.prelu_in(y))
        y, new_state = self.depthwise.forward_step(y, state)
        y = self.norm_mid(self.prelu_mid(y))
        y = self.pointwise_out(y)
        return x_new + y, new_state


class S4DBlock(nn.Module):
    """Paper-faithful S4D block: cLN -> S4D -> FC MLP -> dropout -> residual.

    The enclosing shape is ``[B, C, T]`` (channels-first) to match the
    TCN block; internally we transpose into the S4D's ``[B, T, C]``
    convention.
    """

    def __init__(
        self,
        channels: int,
        d_state: int,
        hidden_channels: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm = ChannelwiseLayerNorm(channels)
        self.s4d = S4D(d_model=channels, d_state=d_state, transposed=False)
        self.fc_in = nn.Linear(channels, hidden_channels)
        self.activation = nn.PReLU(hidden_channels)
        self.fc_out = nn.Linear(hidden_channels, channels)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm(x)
        # S4D (transposed=False) expects [B, T, C]
        y = y.transpose(1, 2)
        y = self.s4d(y)
        y = self.fc_in(y)
        y = self.activation(y.transpose(1, 2)).transpose(1, 2)
        y = self.fc_out(y)
        y = self.dropout(y)
        y = y.transpose(1, 2)
        return x + y

    def zero_state(self, batch_size: int, device: torch.device,
                   dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Streaming state = S4D complex modes in real form."""
        return self.s4d.initial_state(batch_size, device=device, dtype=dtype)

    def forward_step(
        self,
        x_new: torch.Tensor,
        state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Process one or more new timesteps incrementally.

        Args:
            x_new: ``[B, C, T_new]``.
            state: ``[B, C, N/2, 2]`` from the previous call.

        Returns:
            Tuple of ``[B, C, T_new]`` output and the updated state.
            Each timestep of the new slice is fed through
            ``S4D.forward_step`` sequentially, which is the canonical
            streaming protocol (one state update per timestep).
        """
        y = self.norm(x_new)
        # [B, C, T_new] → iterate over T_new
        outputs = []
        cur_state = state
        for t in range(y.size(-1)):
            y_t, cur_state = self.s4d.forward_step(y[..., t], cur_state)
            outputs.append(y_t)
        out = torch.stack(outputs, dim=-1)  # [B, C, T_new]
        out = out.transpose(1, 2)
        out = self.fc_in(out)
        out = self.activation(out.transpose(1, 2)).transpose(1, 2)
        out = self.fc_out(out)
        out = self.dropout(out)
        out = out.transpose(1, 2)
        return x_new + out, cur_state


# ---------------------------------------------------------------------------
# Full separator
# ---------------------------------------------------------------------------


class SpeakerBeamSS(nn.Module):
    """SpeakerBeam-SS target speaker extractor.

    Takes a mixture waveform and an L2-normalized speaker
    embedding, returns the estimated target waveform and an optional
    target-presence logit.

    The embedding is expected to have been produced by the speaker
    encoder on the user's enrollment audio and cached for the entire
    session.
    """

    def __init__(self, config: SpeakerBeamSSConfig | None = None) -> None:
        super().__init__()
        self.config = config or SpeakerBeamSSConfig()
        cfg = self.config

        # --- Learned encoder + decoder ---
        self.encoder = nn.Conv1d(
            in_channels=1,
            out_channels=cfg.enc_channels,
            kernel_size=cfg.enc_kernel_size,
            stride=cfg.enc_stride,
            padding=0,
            bias=False,
        )
        self.decoder = nn.ConvTranspose1d(
            in_channels=cfg.enc_channels,
            out_channels=1,
            kernel_size=cfg.enc_kernel_size,
            stride=cfg.enc_stride,
            padding=0,
            bias=False,
        )

        # --- Bottleneck projection ---
        self.pre_norm = ChannelwiseLayerNorm(cfg.enc_channels)
        self.bottleneck = nn.Conv1d(
            cfg.enc_channels, cfg.bottleneck_channels, 1
        )

        # --- Separator stack: per repeat = 3 TCN, 1 S4D, 1 TCN ---
        self.blocks = nn.ModuleList()
        for _ in range(cfg.num_repeats):
            for i in range(cfg.r1_blocks):
                dilation = 2 ** i
                self.blocks.append(TCNBlock(
                    in_channels=cfg.bottleneck_channels,
                    hidden_channels=cfg.hidden_channels,
                    kernel_size=cfg.conv_kernel_size,
                    dilation=dilation,
                ))
            self.blocks.append(S4DBlock(
                channels=cfg.bottleneck_channels,
                d_state=cfg.s4d_state_dim,
                hidden_channels=cfg.hidden_channels,
            ))
            for i in range(cfg.r2_blocks):
                dilation = 2 ** (cfg.r1_blocks + i)
                self.blocks.append(TCNBlock(
                    in_channels=cfg.bottleneck_channels,
                    hidden_channels=cfg.hidden_channels,
                    kernel_size=cfg.conv_kernel_size,
                    dilation=dilation,
                ))

        # --- FiLM speaker conditioning ---
        d = cfg.bottleneck_channels
        e = cfg.speaker_embed_dim
        self.speaker_gamma = nn.Linear(e, d, bias=False)
        self.speaker_beta = nn.Linear(e, d, bias=False)
        with torch.no_grad():
            # Residual FiLM: gamma = 1 + W_g(e), beta = W_b(e), so zero
            # initialization starts from an exact identity/no-op.
            self.speaker_gamma.weight.zero_()
            self.speaker_beta.weight.zero_()

        # --- Mask head (back to encoder dim, sigmoid activation) ---
        self.mask_head = nn.Sequential(
            nn.PReLU(cfg.bottleneck_channels),
            nn.Conv1d(cfg.bottleneck_channels, cfg.enc_channels, 1),
            nn.Sigmoid(),
        )

        # --- Optional target-presence head: global mean pool → linear ---
        self.presence_head: nn.Module | None
        if cfg.target_presence_head:
            self.presence_head = nn.Linear(cfg.bottleneck_channels, 1)
        else:
            self.presence_head = None

    # ------------------------------------------------------------------
    # Whole-sequence forward (training)
    # ------------------------------------------------------------------

    def _apply_speaker_conditioning(
        self,
        feat: torch.Tensor,
        speaker_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """FiLM speaker fusion: ``feat * gamma(e) + beta(e)``.

        This helper is called from both ``forward`` and ``streaming_step``
        so the two paths cannot drift. FiLM is strictly pointwise in time,
        so streaming vs whole-sequence equivalence is preserved without
        any extra state handling.

        Args:
            feat: ``[B, bottleneck_channels, T]`` post-bottleneck features.
            speaker_embedding: ``[B, speaker_embed_dim]`` L2-normalized
                speaker embedding (from ``encode_enrollment``).

        Returns:
            ``[B, bottleneck_channels, T]`` conditioned features.
        """
        if (
            speaker_embedding.dim() != 2
            or speaker_embedding.size(-1) != self.config.speaker_embed_dim
        ):
            raise ValueError(
                "speaker_embedding must be [B, "
                f"{self.config.speaker_embed_dim}], "
                f"got {tuple(speaker_embedding.shape)}"
            )
        gamma = 1.0 + self.speaker_gamma(speaker_embedding)  # [B, d]
        beta = self.speaker_beta(speaker_embedding)          # [B, d]
        return feat * gamma.unsqueeze(-1) + beta.unsqueeze(-1)  # [B, d, T]

    def forward(
        self,
        mixture: torch.Tensor,
        speaker_embedding: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Whole-sequence forward, numerically aligned with streaming.

        The input is left-padded with ``enc_stride`` zeros and the
        output is cropped back to ``T`` at the end, so that calling
        ``forward`` on a full utterance produces bit-identical results
        to calling ``streaming_step`` on the same utterance in any
        valid chunking. This alignment is important: it means the
        model we train with ``forward`` is the same model we deploy
        via ``streaming_step`` / ONNX.

        Args:
            mixture: ``[B, T]`` waveform in ``[-1, 1]`` at 16 kHz.
                ``T`` should be a multiple of ``enc_stride`` (160
                default). Values that are not will still run but
                will have a few samples of boundary artifact at the
                decoder end.
            speaker_embedding: ``[B, speaker_embed_dim]`` L2-normalized speaker
                embedding.

        Returns:
            Dict with:
            - ``"clean"``: ``[B, T]`` estimated target waveform, same
              length as the input mixture (cropped from the padded
              decoder output).
            - ``"presence_logit"``: ``[B]`` pre-sigmoid target-presence
              logit (present only when ``target_presence_head`` is
              enabled).
        """
        if mixture.dim() != 2:
            raise ValueError(
                f"expected mixture shape [B, T], got {tuple(mixture.shape)}"
            )
        if speaker_embedding.dim() != 2 or speaker_embedding.size(-1) != self.config.speaker_embed_dim:
            raise ValueError(
                "speaker_embedding must be [B, "
                f"{self.config.speaker_embed_dim}], "
                f"got {tuple(speaker_embedding.shape)}"
            )

        T = mixture.shape[-1]
        x = mixture.unsqueeze(1)                    # [B, 1, T]

        # Left-pad with ``enc_kernel_size - enc_stride`` zeros to match
        # the streaming encoder's zero-initialized buffer. This makes
        # forward() and streaming_step() produce the same output.
        pad_left = self.config.enc_kernel_size - self.config.enc_stride
        x = torch.nn.functional.pad(x, (pad_left, 0))

        enc = self.encoder(x)                       # [B, N_enc, L]
        enc_abs = torch.relu(enc)

        # Bottleneck + normalize.
        feat = self.pre_norm(enc_abs)
        feat = self.bottleneck(feat)                # [B, B, L]

        # FiLM speaker conditioning (shared with streaming_step through
        # _apply_speaker_conditioning).
        feat = self._apply_speaker_conditioning(feat, speaker_embedding)

        # Separator stack.
        for block in self.blocks:
            feat = block(feat)                      # [B, B, L]

        # Mask head → apply to encoded features.
        mask = self.mask_head(feat)                 # [B, N_enc, L]
        masked = enc * mask

        clean = self.decoder(masked).squeeze(1)     # [B, T + pad_left]

        # Crop the decoder's trailing overhang so the output matches
        # the input length. The dropped tail is the ``decoder_overlap``
        # that would otherwise be carried into the next streaming call.
        clean = clean[..., :T]

        outputs: dict[str, torch.Tensor] = {"clean": clean}
        if self.presence_head is not None:
            pooled = feat.mean(dim=-1)              # [B, B]
            outputs["presence_logit"] = self.presence_head(pooled).squeeze(-1)
        return outputs

    # ------------------------------------------------------------------
    # Block-level streaming state (kept for tests that inspect it)
    # ------------------------------------------------------------------

    def initial_state(
        self,
        batch_size: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> list:
        """Return zero per-block streaming states for the separator stack.

        This is a low-level helper that returns only the separator
        blocks' states. For a full session streaming state including
        encoder buffer and decoder overlap, use
        :meth:`initial_streaming_state` instead.
        """
        if device is None:
            device = next(self.parameters()).device
        return [
            block.zero_state(batch_size, device, dtype)
            for block in self.blocks
        ]

    # ------------------------------------------------------------------
    # Full session streaming
    # ------------------------------------------------------------------

    def initial_streaming_state(
        self,
        batch_size: int = 1,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> dict:
        """Return the full streaming state for a fresh session.

        The state dict threads three kinds of buffers across
        ``streaming_step`` calls:

        - ``"encoder_buffer"``: the last ``enc_kernel_size - enc_stride``
          input samples, needed for the next encoder frame.
        - ``"block_states"``: per-separator-block states, same thing
          ``initial_state`` returns.
        - ``"decoder_overlap"``: the trailing samples of the previous
          decoder output that overlap with the next chunk's leading
          samples under the ConvTranspose1d overlap-add arithmetic.

        The initial state is all zeros, matching what ``forward``'s
        left-pad represents. This ensures the first streaming call on
        a fresh session produces the same samples as the first
        ``enc_stride`` × N samples of a ``forward`` pass over the
        concatenated chunk sequence.
        """
        if device is None:
            device = next(self.parameters()).device
        enc_overlap = self.config.enc_kernel_size - self.config.enc_stride
        return {
            "encoder_buffer": torch.zeros(
                batch_size, 1, enc_overlap, device=device, dtype=dtype,
            ),
            "block_states": [
                block.zero_state(batch_size, device, dtype) for block in self.blocks
            ],
            "decoder_overlap": torch.zeros(
                batch_size, 1, enc_overlap, device=device, dtype=dtype,
            ),
        }

    def streaming_step(
        self,
        mixture_chunk: torch.Tensor,
        speaker_embedding: torch.Tensor,
        state: dict,
        s4d_state_decay: float = 0.995,
    ) -> tuple[torch.Tensor, dict]:
        """Stateful frame-by-frame forward.

        Args:
            mixture_chunk: ``[B, T_chunk]`` new audio samples. ``T_chunk``
                must be a positive multiple of ``enc_stride`` (160 by
                default). Typical values: 160 (10 ms) or 320 (20 ms).
            speaker_embedding: ``[B, speaker_embed_dim]`` L2-normalized
                speaker embedding, normally computed once per session.
            state: dict from :meth:`initial_streaming_state` or the
                previous call's second return value.
            s4d_state_decay: Per-step multiplicative decay for S4D
                recurrent state. ``1.0`` disables decay.

        Returns:
            Tuple of:
            - ``clean_chunk``: ``[B, T_chunk]`` clean output for
              exactly the samples that were fed in. The overlap-add
              with the next chunk is handled via the updated state.
            - ``new_state``: dict to pass into the next call.

        This method does NOT compute the presence head. The presence
        head uses global mean pool over the full utterance and is
        meaningful only in the whole-sequence ``forward`` path.
        """
        if mixture_chunk.dim() != 2:
            raise ValueError(
                f"expected mixture_chunk shape [B, T], got {tuple(mixture_chunk.shape)}"
            )
        cfg = self.config
        stride = cfg.enc_stride
        if mixture_chunk.shape[-1] == 0 or mixture_chunk.shape[-1] % stride != 0:
            raise ValueError(
                f"streaming chunk length {mixture_chunk.shape[-1]} must be a "
                f"positive multiple of enc_stride={stride}"
            )
        if speaker_embedding.dim() != 2 or speaker_embedding.size(-1) != cfg.speaker_embed_dim:
            raise ValueError(
                "speaker_embedding must be [B, "
                f"{cfg.speaker_embed_dim}], got {tuple(speaker_embedding.shape)}"
            )

        # ---- Encoder with left-context buffer ----
        x = mixture_chunk.unsqueeze(1)  # [B, 1, T_chunk]
        x_full = torch.cat([state["encoder_buffer"], x], dim=-1)
        enc = self.encoder(x_full)  # [B, N_enc, L_frames]
        enc_abs = torch.relu(enc)
        # New encoder buffer = last enc_kernel_size - enc_stride samples of x_full.
        enc_overlap = cfg.enc_kernel_size - cfg.enc_stride
        new_encoder_buffer = x_full[..., -enc_overlap:]

        # ---- Bottleneck + speaker fusion ----
        feat = self.pre_norm(enc_abs)
        feat = self.bottleneck(feat)                     # [B, B, L_frames]
        feat = self._apply_speaker_conditioning(feat, speaker_embedding)

        # ---- Separator stack (with per-block state) ----
        new_block_states = []
        for block, block_state in zip(self.blocks, state["block_states"]):
            feat, new_bs = block.forward_step(feat, block_state)
            if s4d_state_decay < 1.0 and isinstance(block, S4DBlock):
                new_bs = new_bs * s4d_state_decay
            new_block_states.append(new_bs)

        # ---- Mask head ----
        mask = self.mask_head(feat)
        masked = enc * mask

        # ---- Decoder with overlap-add ----
        decoded = self.decoder(masked)  # [B, 1, T_chunk + enc_overlap]
        # Overlap-add with the previous chunk's tail.
        decoded = decoded.clone()
        decoded[..., :enc_overlap] = decoded[..., :enc_overlap] + state["decoder_overlap"]

        # Commit exactly T_chunk samples, save the trailing enc_overlap
        # samples as the next call's decoder_overlap.
        committed = decoded[..., : mixture_chunk.shape[-1]]
        new_decoder_overlap = decoded[..., mixture_chunk.shape[-1]:]

        new_state = {
            "encoder_buffer": new_encoder_buffer,
            "block_states": new_block_states,
            "decoder_overlap": new_decoder_overlap,
            # Lightweight mask diagnostics — always computed (one
            # float each, negligible cost) so callers can inspect
            # mask behavior without changing the API.
            "mask_mean": float(mask.mean().item()),
            "mask_max": float(mask.max().item()),
            "mask_min": float(mask.min().item()),
        }
        return committed.squeeze(1), new_state
