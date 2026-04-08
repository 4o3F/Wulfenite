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
    bottleneck [B, B, L]                          ← B = 192, matches CAM++
        ↓ speaker fusion  (element-wise multiply with L2-normed embedding)
    conditioned [B, B, L]
        ↓ separator stack  (num_blocks * TCN+S4D block, all causal)
    refined [B, B, L]
        ↓ mask head (Conv1d 1×1 → sigmoid)  +  presence head (Linear → logit)
    mask [B, N_enc, L]
        ↓ masked encoded (element-wise multiply)
    masked [B, N_enc, L]
        ↓ learned decoder (ConvTranspose1d kernel=320, hop=160)
    clean waveform [B, T]

The critical design choices from `docs/architecture.md`:

- Bottleneck dimension **B = 192** to match CAM++'s 192-dim output
  directly, eliminating any projection layer between the encoder and
  the separator. Speaker fusion is pure multiplicative adaptation on
  an already-L2-normalized embedding.
- Fully causal: every Conv1d in the post-encoder path is a
  ``CausalConv1d`` from :mod:`wulfenite.models.layers`, and every
  normalization is cLN.
- S4D blocks provide long-range temporal context; the 2-block stack
  follows the paper's "SS-2" configuration (2 repetitions, not 8 like
  baseline TD-SpeakerBeam, because S4D is more sample-efficient).
- The model exposes BOTH ``forward`` (whole-sequence training) and
  ``forward_step`` (stateful streaming). The two must agree
  numerically; the per-module streaming state carries S4D modes and
  causal-conv left-context buffers.

Parameter count at default config: ~7.9 M.
"""

from __future__ import annotations

from dataclasses import dataclass, field

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

    The defaults match the paper's low-latency (40 ms) variant with
    the Wulfenite-specific change of bottleneck ``B = 192`` to accept
    CAM++ embeddings without a projection.
    """

    sample_rate: int = 16000

    # Encoder / decoder (learned 1-D conv, matches SpeakerBeam-SS paper).
    enc_kernel_size: int = 320   # 20 ms @ 16 kHz
    enc_stride: int = 160        # 10 ms hop
    enc_channels: int = 512

    # Separator bottleneck — must equal the speaker embedding dim.
    bottleneck_channels: int = 192

    # TCN+S4D block stack.
    num_repeats: int = 2         # "SS-2" in the paper
    num_blocks_per_repeat: int = 8
    conv_kernel_size: int = 3
    hidden_channels: int = 512   # "H" in the paper

    # S4D block.
    s4d_state_dim: int = 64

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
    """Wrap an S4D layer in a cLN + PReLU + residual shell.

    The enclosing shape is ``[B, C, T]`` (channels-first) to match the
    TCN block; internally we transpose into the S4D's ``[B, T, C]``
    convention.
    """

    def __init__(self, channels: int, d_state: int) -> None:
        super().__init__()
        self.norm = ChannelwiseLayerNorm(channels)
        self.prelu = nn.PReLU(channels)
        self.s4d = S4D(d_model=channels, d_state=d_state, transposed=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm(x)
        y = self.prelu(y)
        # S4D (transposed=False) expects [B, T, C]
        y = y.transpose(1, 2)
        y = self.s4d(y)
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
        y = self.prelu(y)
        # [B, C, T_new] → iterate over T_new
        outputs = []
        cur_state = state
        for t in range(y.size(-1)):
            y_t, cur_state = self.s4d.forward_step(y[..., t], cur_state)
            outputs.append(y_t)
        out = torch.stack(outputs, dim=-1)  # [B, C, T_new]
        return x_new + out, cur_state


# ---------------------------------------------------------------------------
# Full separator
# ---------------------------------------------------------------------------


class SpeakerBeamSS(nn.Module):
    """SpeakerBeam-SS target speaker extractor.

    Takes a mixture waveform and an L2-normalized 192-d speaker
    embedding, returns the estimated target waveform and an optional
    target-presence logit.

    The embedding is expected to have been produced by
    :func:`wulfenite.models.campplus.encode_enrollment` on the user's
    enrollment audio and cached for the entire session.
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

        # --- Separator stack: alternating TCN + S4D blocks ---
        self.blocks = nn.ModuleList()
        for _ in range(cfg.num_repeats):
            for i in range(cfg.num_blocks_per_repeat):
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
            ))

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

    def forward(
        self,
        mixture: torch.Tensor,
        speaker_embedding: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Whole-sequence training forward.

        Args:
            mixture: ``[B, T]`` waveform in ``[-1, 1]`` at 16 kHz.
                ``T`` should be a multiple of ``enc_stride`` (160 by
                default) to avoid encoder/decoder length mismatch.
            speaker_embedding: ``[B, 192]`` L2-normalized CAM++ output.

        Returns:
            Dict with:
            - ``"clean"``: ``[B, T_out]`` estimated target waveform.
            - ``"presence_logit"``: ``[B]`` pre-sigmoid target presence
              logit (present only when ``target_presence_head`` is
              enabled).
        """
        if mixture.dim() != 2:
            raise ValueError(
                f"expected mixture shape [B, T], got {tuple(mixture.shape)}"
            )
        if speaker_embedding.dim() != 2 or speaker_embedding.size(-1) != self.config.bottleneck_channels:
            raise ValueError(
                "speaker_embedding must be [B, "
                f"{self.config.bottleneck_channels}], "
                f"got {tuple(speaker_embedding.shape)}"
            )

        x = mixture.unsqueeze(1)                    # [B, 1, T]
        enc = self.encoder(x)                       # [B, N_enc, L]
        enc_abs = torch.relu(enc)                   # Conv-TasNet-style half-wave

        # Bottleneck + normalize.
        feat = self.pre_norm(enc_abs)
        feat = self.bottleneck(feat)                # [B, B, L]

        # Multiplicative speaker fusion — broadcast embedding across T.
        feat = feat * speaker_embedding.unsqueeze(-1)

        # Separator stack.
        for block in self.blocks:
            feat = block(feat)                      # [B, B, L]

        # Mask head → apply to encoded features.
        mask = self.mask_head(feat)                 # [B, N_enc, L]
        masked = enc * mask                         # real-masking in encoder domain

        clean = self.decoder(masked).squeeze(1)     # [B, T]

        outputs: dict[str, torch.Tensor] = {"clean": clean}
        if self.presence_head is not None:
            pooled = feat.mean(dim=-1)              # [B, B]
            outputs["presence_logit"] = self.presence_head(pooled).squeeze(-1)
        return outputs

    # ------------------------------------------------------------------
    # Streaming helpers
    # ------------------------------------------------------------------

    def initial_state(
        self,
        batch_size: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> list:
        """Allocate zero streaming state for one session.

        Returns a list of per-block states, each entry being whatever
        the block's ``zero_state`` returns. The caller does not need
        to introspect the list — just pass it back into
        ``forward_step``.

        Note: this is the state for the *separator stack* only. The
        encoder Conv1d also has left-context that the streaming code
        in Rust will manage via its own ring buffer (outside this
        module), because the encoder's stride > 1 needs careful
        frame-aligned handling that is cleaner to do in the runtime.
        """
        if device is None:
            device = next(self.parameters()).device
        return [
            block.zero_state(batch_size, device, dtype)
            for block in self.blocks
        ]
