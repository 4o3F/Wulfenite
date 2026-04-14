"""SpeakerBeam-SS separator aligned to the paper's Figure 1 topology.

Architecture at a glance:

    waveform [B, T]
        ↓ learned encoder (Conv1d kernel=320, hop=160) + ReLU
    encoded features [B, N_enc, L]
        ├─ skip branch kept for final masking
        └─ LayerNorm + 1×1 Conv → bottleneck [B, B_dim, L]
             ↓ R1 repetitions of X × (TCN -> S4D)
             ↓ speaker modulation (broadcast multiply)
             ↓ R2 repetitions of X × (TCN -> S4D)
             ↓ 1×1 Conv + ReLU
    mask [B, N_enc, L]
        ↓ element-wise multiply with encoded features
    masked features [B, N_enc, L]
        ↓ learned decoder (ConvTranspose1d kernel=320, hop=160)
    clean waveform [B, T]

The implementation keeps the repo's streaming equivalence contract:
whole-sequence ``forward`` and chunked ``streaming_step`` are numerically
aligned when run with the same parameters.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .layers import CausalConv1d, ChannelwiseLayerNorm
from .s4d import S4D


_VALID_LOOKAHEAD_POLICIES = frozenset({"post_fusion_frontloaded"})


def _build_lookahead_plan(
    separator_lookahead_frames: int,
    policy: str,
    n_pre_blocks: int,
    n_post_blocks: int,
) -> tuple[int, ...]:
    """Return per-block TCN right-context allocation.

    The returned tuple contains one entry per ``ConvS4DSubBlock`` in the
    order ``pre_fusion_blocks`` then ``post_fusion_blocks``. Each entry
    is currently either 0 or 1, indicating whether that block consumes
    one future encoder frame of look-ahead.
    """
    total_blocks = n_pre_blocks + n_post_blocks
    if policy not in _VALID_LOOKAHEAD_POLICIES:
        raise ValueError(
            f"unsupported lookahead_policy {policy!r}; "
            f"expected one of {sorted(_VALID_LOOKAHEAD_POLICIES)}"
        )
    if not 0 <= separator_lookahead_frames <= total_blocks:
        raise ValueError(
            "separator_lookahead_frames must be between 0 and "
            f"{total_blocks}; got {separator_lookahead_frames}"
        )

    if policy == "post_fusion_frontloaded":
        priority = list(range(n_pre_blocks, total_blocks))
        priority += list(range(n_pre_blocks - 1, -1, -1))
    else:  # pragma: no cover - guarded by validation above.
        raise AssertionError(f"unhandled lookahead policy {policy!r}")

    contexts = [0] * total_blocks
    for i in range(separator_lookahead_frames):
        contexts[priority[i]] = 1
    return tuple(contexts)


@dataclass
class SpeakerBeamSSConfig:
    """Hyperparameter bundle for SpeakerBeam-SS."""

    sample_rate: int = 16000

    # Encoder / decoder.
    enc_kernel_size: int = 320
    enc_stride: int = 160
    enc_channels: int = 2048

    # Separator bottleneck.
    bottleneck_channels: int = 256
    speaker_embed_dim: int = 192

    # Conv-TasNet + S4D separator stages.
    r1_repeats: int = 3
    r2_repeats: int = 1
    conv_blocks_per_repeat: int = 2
    conv_kernel_size: int = 3
    hidden_channels: int = 512
    s4d_state_dim: int = 32
    s4d_ffn_multiplier: int = 4
    separator_lookahead_frames: int = 0
    lookahead_policy: str = "post_fusion_frontloaded"

    # Mask activation: "relu" (Conv-TasNet default) or "scaled_sigmoid"
    # (bounded [0, 2], no dead zone). Default "relu" for legacy checkpoint
    # compatibility; new training should use "scaled_sigmoid".
    mask_activation: str = "relu"

    # Repo extension kept optional for experimentation; disabled by default
    # because it is not part of the paper architecture.
    target_presence_head: bool = False


class TCNBlock(nn.Module):
    """Standard Conv-TasNet residual 1-D convolution block."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        kernel_size: int,
        dilation: int,
        *,
        depthwise_right_context: int = 0,
    ) -> None:
        super().__init__()
        self.pointwise_in = nn.Conv1d(in_channels, hidden_channels, 1)
        self.prelu_in = nn.PReLU(hidden_channels)
        self.norm_in = ChannelwiseLayerNorm(hidden_channels)

        self.depthwise = CausalConv1d(
            hidden_channels,
            hidden_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            groups=hidden_channels,
            right_context=depthwise_right_context,
        )
        self.depthwise_right_context = depthwise_right_context
        self.prelu_mid = nn.PReLU(hidden_channels)
        self.norm_mid = ChannelwiseLayerNorm(hidden_channels)

        self.pointwise_out = nn.Conv1d(hidden_channels, in_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.pointwise_in(x)
        y = self.norm_in(self.prelu_in(y))
        y = self.depthwise(y)
        y = self.norm_mid(self.prelu_mid(y))
        y = self.pointwise_out(y)
        return x + y

    def zero_state(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        return self.depthwise.zero_state(batch_size, device, dtype)

    def forward_step(
        self,
        x_new: torch.Tensor,
        state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        y, new_state = self.forward_step_branch(x_new, state)
        return x_new + y, new_state

    def forward_step_branch(
        self,
        x_new: torch.Tensor,
        state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        y = self.pointwise_in(x_new)
        y = self.norm_in(self.prelu_in(y))
        y, new_state = self.depthwise.forward_step(y, state)
        y = self.norm_mid(self.prelu_mid(y))
        y = self.pointwise_out(y)
        return y, new_state


class S4DBlock(nn.Module):
    """Figure 1(b) S4D block with GLU branch and FFN branch."""

    def __init__(
        self,
        channels: int,
        d_state: int,
        *,
        ffn_multiplier: int = 4,
    ) -> None:
        super().__init__()
        self.norm_1 = ChannelwiseLayerNorm(channels)
        self.s4d = S4D(d_model=channels, d_state=d_state, transposed=False)
        self.act_1 = nn.GELU()
        self.glu_in = nn.Conv1d(channels, 2 * channels, 1)

        self.norm_2 = ChannelwiseLayerNorm(channels)
        self.ffn_in = nn.Conv1d(channels, ffn_multiplier * channels, 1)
        self.act_2 = nn.GELU()
        self.ffn_out = nn.Conv1d(ffn_multiplier * channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual_1 = x
        y = self.norm_1(x)
        y = self.s4d(y.transpose(1, 2)).transpose(1, 2)
        y = self.act_1(y)
        y = self.glu_in(y)
        y = F.glu(y, dim=1)
        mid = residual_1 + y

        residual_2 = mid
        y = self.norm_2(mid)
        y = self.ffn_in(y)
        y = self.act_2(y)
        y = self.ffn_out(y)
        return residual_2 + y

    def zero_state(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        return self.s4d.initial_state(batch_size, device=device, dtype=dtype)

    def forward_step(
        self,
        x_new: torch.Tensor,
        state: torch.Tensor,
        *,
        s4d_state_decay: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        residual_1 = x_new
        y = self.norm_1(x_new)
        outputs = []
        cur_state = state
        for t in range(y.size(-1)):
            if s4d_state_decay < 1.0:
                cur_state = cur_state * s4d_state_decay
            y_t, cur_state = self.s4d.forward_step(y[..., t], cur_state)
            outputs.append(y_t)
        y = torch.stack(outputs, dim=-1)
        y = self.act_1(y)
        y = self.glu_in(y)
        y = F.glu(y, dim=1)
        mid = residual_1 + y

        residual_2 = mid
        y = self.norm_2(mid)
        y = self.ffn_in(y)
        y = self.act_2(y)
        y = self.ffn_out(y)
        return residual_2 + y, cur_state


class ConvS4DSubBlock(nn.Module):
    """One separator sub-block: Conv-TasNet block followed by S4D block."""

    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        kernel_size: int,
        dilation: int,
        d_state: int,
        *,
        ffn_multiplier: int,
        tcn_right_context: int = 0,
    ) -> None:
        super().__init__()
        self.conv = TCNBlock(
            in_channels=channels,
            hidden_channels=hidden_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            depthwise_right_context=tcn_right_context,
        )
        self.tcn_right_context = tcn_right_context
        self.s4d = S4DBlock(
            channels=channels,
            d_state=d_state,
            ffn_multiplier=ffn_multiplier,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.s4d(self.conv(x))

    def zero_state(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            self.conv.zero_state(batch_size, device, dtype),
            self.s4d.zero_state(batch_size, device, dtype),
        )

    def forward_step(
        self,
        x_new: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor],
        *,
        s4d_state_decay: float = 1.0,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        conv_state, s4d_state = state
        y, new_conv_state = self.conv.forward_step(x_new, conv_state)
        y, new_s4d_state = self.s4d.forward_step(
            y,
            s4d_state,
            s4d_state_decay=s4d_state_decay,
        )
        return y, (new_conv_state, new_s4d_state)

    def reset_s4d_state(
        self,
        state: tuple[torch.Tensor, torch.Tensor],
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        conv_state, _ = state
        return (
            conv_state,
            self.s4d.zero_state(batch_size, device, dtype),
        )


class SpeakerBeamSS(nn.Module):
    """SpeakerBeam-SS target speaker extractor."""

    def __init__(self, config: SpeakerBeamSSConfig | None = None) -> None:
        super().__init__()
        self.config = config or SpeakerBeamSSConfig()
        cfg = self.config
        n_pre_blocks = cfg.r1_repeats * cfg.conv_blocks_per_repeat
        n_post_blocks = cfg.r2_repeats * cfg.conv_blocks_per_repeat
        self._lookahead_plan = _build_lookahead_plan(
            cfg.separator_lookahead_frames,
            cfg.lookahead_policy,
            n_pre_blocks,
            n_post_blocks,
        )
        pre_contexts = self._lookahead_plan[:n_pre_blocks]
        post_contexts = self._lookahead_plan[n_pre_blocks:]

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

        self.pre_norm = ChannelwiseLayerNorm(cfg.enc_channels)
        self.bottleneck = nn.Conv1d(cfg.enc_channels, cfg.bottleneck_channels, 1)

        self.pre_fusion_blocks = nn.ModuleList(
            self._build_stage_blocks(
                cfg,
                repeats=cfg.r1_repeats,
                right_contexts=pre_contexts,
            )
        )
        self.post_fusion_blocks = nn.ModuleList(
            self._build_stage_blocks(
                cfg,
                repeats=cfg.r2_repeats,
                right_contexts=post_contexts,
            )
        )

        self.speaker_projection = nn.Linear(
            cfg.speaker_embed_dim,
            cfg.bottleneck_channels,
        )
        with torch.no_grad():
            self.speaker_projection.weight.zero_()
            self.speaker_projection.bias.fill_(1.0)

        self.mask_head = nn.Conv1d(cfg.bottleneck_channels, cfg.enc_channels, 1)

        self.presence_head: nn.Module | None
        if cfg.target_presence_head:
            self.presence_head = nn.Linear(cfg.bottleneck_channels, 1)
        else:
            self.presence_head = None

    @staticmethod
    def _build_stage_blocks(
        cfg: SpeakerBeamSSConfig,
        *,
        repeats: int,
        right_contexts: tuple[int, ...],
    ) -> list[ConvS4DSubBlock]:
        expected = repeats * cfg.conv_blocks_per_repeat
        if len(right_contexts) != expected:
            raise ValueError(
                "right_contexts length must match stage block count; "
                f"expected {expected}, got {len(right_contexts)}"
            )
        blocks: list[ConvS4DSubBlock] = []
        context_index = 0
        for _ in range(repeats):
            for block_index in range(cfg.conv_blocks_per_repeat):
                blocks.append(
                    ConvS4DSubBlock(
                        channels=cfg.bottleneck_channels,
                        hidden_channels=cfg.hidden_channels,
                        kernel_size=cfg.conv_kernel_size,
                        dilation=2 ** block_index,
                        d_state=cfg.s4d_state_dim,
                        ffn_multiplier=cfg.s4d_ffn_multiplier,
                        tcn_right_context=right_contexts[context_index],
                    )
                )
                context_index += 1
        return blocks

    def _all_blocks(self) -> tuple[ConvS4DSubBlock, ...]:
        return tuple(self.pre_fusion_blocks) + tuple(self.post_fusion_blocks)

    _VALID_MASK_ACTIVATIONS = frozenset({"relu", "scaled_sigmoid"})

    def _apply_mask_activation(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply the configured mask activation to raw mask logits."""
        act = self.config.mask_activation
        if act == "scaled_sigmoid":
            return 2.0 * torch.sigmoid(logits)
        if act == "relu":
            return torch.relu(logits)
        raise ValueError(
            f"unsupported mask_activation {act!r}; "
            f"expected one of {sorted(self._VALID_MASK_ACTIVATIONS)}"
        )

    def _apply_speaker_modulation(
        self,
        feat: torch.Tensor,
        speaker_embedding: torch.Tensor,
    ) -> torch.Tensor:
        if (
            speaker_embedding.dim() != 2
            or speaker_embedding.size(-1) != self.config.speaker_embed_dim
        ):
            raise ValueError(
                "speaker_embedding must be [B, "
                f"{self.config.speaker_embed_dim}], "
                f"got {tuple(speaker_embedding.shape)}"
            )
        speaker_scale = self.speaker_projection(speaker_embedding)
        return feat * speaker_scale.unsqueeze(-1)

    def forward(
        self,
        mixture: torch.Tensor,
        speaker_embedding: torch.Tensor,
        *,
        return_training_aux: bool = False,
    ) -> dict[str, torch.Tensor]:
        if mixture.dim() != 2:
            raise ValueError(
                f"expected mixture shape [B, T], got {tuple(mixture.shape)}"
            )

        cfg = self.config
        total_samples = mixture.shape[-1]
        x = mixture.unsqueeze(1)
        pad_left = cfg.enc_kernel_size - cfg.enc_stride
        pad_right = cfg.separator_lookahead_frames * cfg.enc_stride
        x = F.pad(x, (pad_left, pad_right))

        enc_full = torch.relu(self.encoder(x))
        real_frames = (total_samples + cfg.enc_stride - 1) // cfg.enc_stride
        enc_real = enc_full[..., :real_frames]

        feat = self.pre_norm(enc_full)
        feat = self.bottleneck(feat)

        for block in self.pre_fusion_blocks:
            feat = block(feat)
        feat = self._apply_speaker_modulation(feat, speaker_embedding)
        for block in self.post_fusion_blocks:
            feat = block(feat)

        feat_real = feat[..., :real_frames]
        mask = self._apply_mask_activation(self.mask_head(feat_real))
        masked = enc_real * mask
        clean = self.decoder(masked).squeeze(1)
        clean = clean[..., :total_samples]

        outputs: dict[str, torch.Tensor] = {
            "clean": clean,
            "mask": mask,
        }
        if return_training_aux:
            outputs["ae_reconstruction"] = (
                self.decoder(enc_real).squeeze(1)[..., :total_samples]
            )
        if self.presence_head is not None:
            pooled = feat_real.mean(dim=-1)
            outputs["presence_logit"] = self.presence_head(pooled).squeeze(-1)
        return outputs

    def initial_state(
        self,
        batch_size: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        if device is None:
            device = next(self.parameters()).device
        return [
            block.zero_state(batch_size, device, dtype)
            for block in self._all_blocks()
        ]

    def initial_streaming_state(
        self,
        batch_size: int = 1,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> dict:
        if device is None:
            device = next(self.parameters()).device
        cfg = self.config
        enc_overlap = cfg.enc_kernel_size - cfg.enc_stride
        lookahead = cfg.separator_lookahead_frames
        return {
            "encoder_buffer": torch.zeros(
                batch_size,
                1,
                enc_overlap,
                device=device,
                dtype=dtype,
            ),
            "block_states": self.initial_state(batch_size, device, dtype),
            "decoder_overlap": torch.zeros(
                batch_size,
                1,
                enc_overlap,
                device=device,
                dtype=dtype,
            ),
            "skip_buffer": torch.zeros(
                batch_size,
                cfg.enc_channels,
                lookahead,
                device=device,
                dtype=dtype,
            ),
            "block_startup_remaining": [
                block.tcn_right_context
                for block in self._all_blocks()
            ],
            "block_input_buffers": [
                torch.zeros(
                    batch_size,
                    cfg.bottleneck_channels,
                    block.tcn_right_context,
                    device=device,
                    dtype=dtype,
                )
                for block in self._all_blocks()
            ],
            "output_buffer": torch.zeros(
                batch_size,
                lookahead * cfg.enc_stride,
                device=device,
                dtype=dtype,
            ),
        }

    def reset_s4d_states_only(
        self,
        state: dict,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> dict:
        new_state = dict(state)
        new_state["block_states"] = [
            block.reset_s4d_state(
                block_state,
                batch_size=batch_size,
                device=device,
                dtype=dtype,
            )
            for block, block_state in zip(self._all_blocks(), state["block_states"])
        ]
        return new_state

    def streaming_hop(
        self,
        mixture_hop: torch.Tensor,
        speaker_embedding: torch.Tensor,
        state: dict,
        s4d_state_decay: float = 1.0,
    ) -> tuple[torch.Tensor, dict]:
        if mixture_hop.dim() != 2:
            raise ValueError(
                f"expected mixture_hop shape [B, T], got {tuple(mixture_hop.shape)}"
            )
        cfg = self.config
        stride = cfg.enc_stride
        if mixture_hop.shape[-1] != stride:
            raise ValueError(
                f"streaming_hop length {mixture_hop.shape[-1]} must equal "
                f"enc_stride={stride}"
            )

        x = mixture_hop.unsqueeze(1)
        enc_overlap = cfg.enc_kernel_size - cfg.enc_stride
        x_full = torch.cat([state["encoder_buffer"], x], dim=-1)
        enc = torch.relu(self.encoder(x_full))
        new_encoder_buffer = x_full[..., -enc_overlap:]

        feat = self.pre_norm(enc)
        feat = self.bottleneck(feat)

        old_block_states = state["block_states"]
        new_block_states: list[tuple[torch.Tensor, torch.Tensor]] = []
        new_startup = list(state["block_startup_remaining"])
        old_input_buffers = state["block_input_buffers"]
        new_input_buffers: list[torch.Tensor] = []
        pre_count = len(self.pre_fusion_blocks)

        for idx, block in enumerate(self.pre_fusion_blocks):
            if feat.shape[-1] == 0:
                new_block_states.append(old_block_states[idx])
                new_input_buffers.append(old_input_buffers[idx])
                continue
            if block.tcn_right_context > 0:
                conv_state, s4d_state = old_block_states[idx]
                residual_source = torch.cat([old_input_buffers[idx], feat], dim=-1)
                feat, new_conv_state = block.conv.forward_step_branch(feat, conv_state)
                drop = min(new_startup[idx], feat.shape[-1])
                if drop > 0:
                    feat = feat[..., drop:]
                    new_startup[idx] -= drop
                new_input_buffers.append(
                    residual_source[..., -block.tcn_right_context:]
                )
                if feat.shape[-1] == 0:
                    new_block_states.append((new_conv_state, s4d_state))
                    continue
                feat = residual_source[..., :feat.shape[-1]] + feat
                feat, new_s4d_state = block.s4d.forward_step(
                    feat,
                    s4d_state,
                    s4d_state_decay=s4d_state_decay,
                )
                new_block_states.append((new_conv_state, new_s4d_state))
                continue
            feat, new_state = block.forward_step(
                feat,
                old_block_states[idx],
                s4d_state_decay=s4d_state_decay,
            )
            new_block_states.append(new_state)
            new_input_buffers.append(old_input_buffers[idx])

        if feat.shape[-1] > 0:
            feat = self._apply_speaker_modulation(feat, speaker_embedding)

        for idx, block in enumerate(self.post_fusion_blocks):
            state_index = pre_count + idx
            if feat.shape[-1] == 0:
                new_block_states.append(old_block_states[state_index])
                new_input_buffers.append(old_input_buffers[state_index])
                continue
            if block.tcn_right_context > 0:
                conv_state, s4d_state = old_block_states[state_index]
                residual_source = torch.cat(
                    [old_input_buffers[state_index], feat],
                    dim=-1,
                )
                feat, new_conv_state = block.conv.forward_step_branch(
                    feat,
                    conv_state,
                )
                drop = min(new_startup[state_index], feat.shape[-1])
                if drop > 0:
                    feat = feat[..., drop:]
                    new_startup[state_index] -= drop
                new_input_buffers.append(
                    residual_source[..., -block.tcn_right_context:]
                )
                if feat.shape[-1] == 0:
                    new_block_states.append((new_conv_state, s4d_state))
                    continue
                feat = residual_source[..., :feat.shape[-1]] + feat
                feat, new_s4d_state = block.s4d.forward_step(
                    feat,
                    s4d_state,
                    s4d_state_decay=s4d_state_decay,
                )
                new_block_states.append((new_conv_state, new_s4d_state))
                continue
            feat, new_state = block.forward_step(
                feat,
                old_block_states[state_index],
                s4d_state_decay=s4d_state_decay,
            )
            new_block_states.append(new_state)
            new_input_buffers.append(old_input_buffers[state_index])

        enc_source = torch.cat([state["skip_buffer"], enc], dim=-1)
        if cfg.separator_lookahead_frames > 0:
            new_skip_buffer = enc_source[..., -cfg.separator_lookahead_frames:]
        else:
            new_skip_buffer = state["skip_buffer"]

        if feat.shape[-1] == 0:
            committed = mixture_hop.new_zeros(mixture_hop.shape[0], 0)
            new_decoder_overlap = state["decoder_overlap"]
            mask_mean = 0.0
            mask_max = 0.0
            mask_min = 0.0
        else:
            valid_len = feat.shape[-1]
            enc_valid = enc_source[..., :valid_len]
            mask = self._apply_mask_activation(self.mask_head(feat))
            masked = enc_valid * mask

            decoded = self.decoder(masked)
            decoded = decoded.clone()
            decoded[..., :enc_overlap] = (
                decoded[..., :enc_overlap] + state["decoder_overlap"]
            )
            committed = decoded[..., : valid_len * stride].squeeze(1)
            new_decoder_overlap = decoded[..., valid_len * stride:]
            mask_mean = float(mask.mean().item())
            mask_max = float(mask.max().item())
            mask_min = float(mask.min().item())

        output_fifo = torch.cat([state["output_buffer"], committed], dim=-1)
        returned = output_fifo[..., :stride]
        new_output_buffer = output_fifo[..., stride:]

        new_state = {
            "encoder_buffer": new_encoder_buffer,
            "block_states": new_block_states,
            "decoder_overlap": new_decoder_overlap,
            "skip_buffer": new_skip_buffer,
            "block_startup_remaining": new_startup,
            "block_input_buffers": new_input_buffers,
            "output_buffer": new_output_buffer,
            "mask_mean": mask_mean,
            "mask_max": mask_max,
            "mask_min": mask_min,
        }
        return returned, new_state

    def streaming_step(
        self,
        mixture_chunk: torch.Tensor,
        speaker_embedding: torch.Tensor,
        state: dict,
        s4d_state_decay: float = 1.0,
    ) -> tuple[torch.Tensor, dict]:
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
        outs = []
        new_state = state
        mask_means = []
        mask_maxs = []
        mask_mins = []
        for hop in mixture_chunk.split(stride, dim=-1):
            out, new_state = self.streaming_hop(
                hop,
                speaker_embedding,
                new_state,
                s4d_state_decay=s4d_state_decay,
            )
            outs.append(out)
            mask_means.append(new_state.get("mask_mean", 0.0))
            mask_maxs.append(new_state.get("mask_max", 0.0))
            mask_mins.append(new_state.get("mask_min", 0.0))
        # Aggregate mask stats across all hops in the chunk.
        if mask_means:
            new_state["mask_mean"] = sum(mask_means) / len(mask_means)
            new_state["mask_max"] = max(mask_maxs)
            new_state["mask_min"] = min(mask_mins)
        return torch.cat(outs, dim=-1), new_state
