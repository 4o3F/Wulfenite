"""DeepFilterNet2 backbone adapted for Wulfenite PSE."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .deep_filtering import DfOp
from .erb import erb_fb, erb_fb_inverse
from .modules import Conv2dNormAct, ConvTranspose2dNormAct, GroupedGRU, GroupedLinear


@dataclass
class DfNetStreamState:
    """Streaming state for frame-wise DFNet2 inference."""

    erb_feat_buf: torch.Tensor
    df_feat_buf: torch.Tensor
    enc_state: torch.Tensor
    erb_dec_state: torch.Tensor
    df_dec_state: torch.Tensor
    df_spec_bufs: tuple[torch.Tensor, ...]


class Encoder(nn.Module):
    """DFNet2 encoder with optional conditioning input."""

    def __init__(
        self,
        *,
        erb_bins: int = 24,
        df_bins: int = 96,
        conv_ch: int = 16,
        emb_hidden_dim: int = 256,
        condition_dim: int = 0,
        gru_groups: int = 1,
        lin_groups: int = 1,
        lsnr_min: float = -15.0,
        lsnr_max: float = 35.0,
    ) -> None:
        super().__init__()
        if erb_bins % 4 != 0:
            raise ValueError(f"erb_bins must be divisible by 4, got {erb_bins}")
        if df_bins % 2 != 0:
            raise ValueError(f"df_bins must be divisible by 2, got {df_bins}")
        self.erb_bins = erb_bins
        self.df_bins = df_bins
        self.conv_ch = conv_ch
        self.condition_dim = condition_dim
        self.emb_in_dim = conv_ch * erb_bins // 4
        self.emb_hidden_dim = emb_hidden_dim
        self.lsnr_min = lsnr_min
        self.lsnr_max = lsnr_max

        self.erb_conv0 = Conv2dNormAct(
            1,
            conv_ch,
            kernel_size=(3, 3),
            separable=True,
            bias=False,
        )
        self.erb_conv1 = Conv2dNormAct(
            conv_ch,
            conv_ch,
            kernel_size=(1, 3),
            fstride=2,
            separable=True,
            bias=False,
        )
        self.erb_conv2 = Conv2dNormAct(
            conv_ch,
            conv_ch,
            kernel_size=(1, 3),
            fstride=2,
            separable=True,
            bias=False,
        )
        self.erb_conv3 = Conv2dNormAct(
            conv_ch,
            conv_ch,
            kernel_size=(1, 3),
            fstride=1,
            separable=True,
            bias=False,
        )

        self.df_conv0 = Conv2dNormAct(
            2,
            conv_ch,
            kernel_size=(3, 3),
            separable=True,
            bias=False,
        )
        self.df_conv1 = Conv2dNormAct(
            conv_ch,
            conv_ch,
            kernel_size=(1, 3),
            fstride=2,
            separable=True,
            bias=False,
        )
        self.df_fc_emb = GroupedLinear(
            conv_ch * (df_bins // 2),
            self.emb_in_dim,
            groups=lin_groups,
        )
        self.emb_gru = GroupedGRU(
            self.emb_in_dim + condition_dim,
            emb_hidden_dim,
            num_layers=1,
            batch_first=True,
            groups=gru_groups,
            shuffle=True,
            add_outputs=True,
        )
        self.lsnr_fc = nn.Sequential(nn.Linear(emb_hidden_dim, 1), nn.Sigmoid())

    def _conditioning_to_frames(
        self,
        conditioning: torch.Tensor | None,
        frames: int,
    ) -> torch.Tensor | None:
        if self.condition_dim == 0:
            return None
        if conditioning is None:
            raise ValueError("conditioning is required for this encoder")
        if conditioning.dim() == 2:
            if conditioning.size(-1) != self.condition_dim:
                raise ValueError(
                    f"expected conditioning dim {self.condition_dim}, got {conditioning.size(-1)}"
                )
            conditioning = conditioning.unsqueeze(1).expand(-1, frames, -1)
        elif conditioning.dim() == 3:
            if conditioning.size(-1) != self.condition_dim:
                raise ValueError(
                    f"expected conditioning dim {self.condition_dim}, got {conditioning.size(-1)}"
                )
            if conditioning.size(1) != frames:
                raise ValueError(
                    f"conditioning frames={conditioning.size(1)} do not match T={frames}"
                )
        else:
            raise ValueError(
                "conditioning must be [B, D] or [B, T, D], "
                f"got {tuple(conditioning.shape)}"
            )
        return conditioning

    def forward(
        self,
        feat_erb: torch.Tensor,
        feat_df: torch.Tensor,
        conditioning: torch.Tensor | None = None,
        state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if feat_erb.dim() != 4 or feat_df.dim() != 4:
            raise ValueError("feat_erb and feat_df must be [B, C, T, F]")
        e0 = self.erb_conv0(feat_erb)
        e1 = self.erb_conv1(e0)
        e2 = self.erb_conv2(e1)
        e3 = self.erb_conv3(e2)

        c0 = self.df_conv0(feat_df)
        c1 = self.df_conv1(c0)
        cemb = c1.permute(0, 2, 3, 1).flatten(2)
        cemb = self.df_fc_emb(cemb)
        emb = e3.permute(0, 2, 3, 1).flatten(2) + cemb
        cond = self._conditioning_to_frames(conditioning, emb.size(1))
        if cond is not None:
            emb = torch.cat((emb, cond), dim=-1)
        emb, state_out = self.emb_gru(emb, state)
        lsnr = self.lsnr_fc(emb) * (self.lsnr_max - self.lsnr_min) + self.lsnr_min
        return e0, e1, e2, e3, emb, c0, lsnr, state_out


class ErbDecoder(nn.Module):
    """ERB mask decoder."""

    def __init__(
        self,
        *,
        erb_bins: int = 24,
        conv_ch: int = 16,
        emb_hidden_dim: int = 256,
        gru_groups: int = 1,
        lin_groups: int = 1,
    ) -> None:
        super().__init__()
        if erb_bins % 4 != 0:
            raise ValueError(f"erb_bins must be divisible by 4, got {erb_bins}")
        low_res_bins = erb_bins // 4
        self.conv_ch = conv_ch
        self.low_res_bins = low_res_bins
        self.emb_gru = GroupedGRU(
            emb_hidden_dim,
            emb_hidden_dim,
            num_layers=1,
            batch_first=True,
            groups=gru_groups,
            shuffle=True,
            add_outputs=True,
        )
        self.fc_emb = nn.Sequential(
            GroupedLinear(emb_hidden_dim, conv_ch * low_res_bins, groups=lin_groups),
            nn.ReLU(),
        )
        self.conv3p = Conv2dNormAct(conv_ch, conv_ch, kernel_size=1, bias=False, separable=True)
        self.convt3 = Conv2dNormAct(
            conv_ch,
            conv_ch,
            kernel_size=(1, 3),
            bias=False,
            separable=True,
        )
        self.conv2p = Conv2dNormAct(conv_ch, conv_ch, kernel_size=1, bias=False, separable=True)
        self.convt2 = ConvTranspose2dNormAct(
            conv_ch,
            conv_ch,
            kernel_size=(1, 3),
            fstride=2,
            bias=False,
            separable=True,
        )
        self.conv1p = Conv2dNormAct(conv_ch, conv_ch, kernel_size=1, bias=False, separable=True)
        self.convt1 = ConvTranspose2dNormAct(
            conv_ch,
            conv_ch,
            kernel_size=(1, 3),
            fstride=2,
            bias=False,
            separable=True,
        )
        self.conv0p = Conv2dNormAct(conv_ch, conv_ch, kernel_size=1, bias=False, separable=True)
        self.conv0_out = Conv2dNormAct(
            conv_ch,
            1,
            kernel_size=(1, 3),
            bias=True,
            separable=False,
            norm_layer=None,
            activation_layer=nn.Sigmoid,
        )

    def forward(
        self,
        emb: torch.Tensor,
        e3: torch.Tensor,
        e2: torch.Tensor,
        e1: torch.Tensor,
        e0: torch.Tensor,
        state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, _, frames, low_res_bins = e3.shape
        emb, state_out = self.emb_gru(emb, state)
        emb = self.fc_emb(emb).view(batch, frames, low_res_bins, self.conv_ch).permute(0, 3, 1, 2)
        d3 = self.convt3(self.conv3p(e3) + emb)
        d2 = self.convt2(self.conv2p(e2) + d3)
        d1 = self.convt1(self.conv1p(e1) + d2)
        gains = self.conv0_out(self.conv0p(e0) + d1)
        return gains, state_out


class DfDecoder(nn.Module):
    """Deep-filter coefficient decoder."""

    def __init__(
        self,
        *,
        df_bins: int = 96,
        df_order: int = 5,
        conv_ch: int = 16,
        emb_hidden_dim: int = 256,
        hidden_dim: int = 256,
        num_layers: int = 3,
        gru_groups: int = 1,
    ) -> None:
        super().__init__()
        self.df_bins = df_bins
        self.df_order = df_order
        self.df_out_ch = df_order * 2
        self.df_gru = GroupedGRU(
            emb_hidden_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            groups=gru_groups,
            shuffle=True,
            add_outputs=True,
        )
        self.df_out = nn.Sequential(
            nn.Linear(hidden_dim, df_bins * self.df_out_ch),
            nn.Tanh(),
        )
        self.df_fc_a = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())
        self.df_convp = Conv2dNormAct(
            conv_ch,
            self.df_out_ch,
            kernel_size=(1, 1),
            bias=False,
            separable=False,
        )

    def forward(
        self,
        emb: torch.Tensor,
        c0: torch.Tensor,
        state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, frames, _ = emb.shape
        hid, state_out = self.df_gru(emb, state)
        bias = self.df_convp(c0).permute(0, 2, 3, 1)
        alpha = self.df_fc_a(hid)
        coefs = self.df_out(hid).view(batch, frames, self.df_bins, self.df_out_ch) + bias
        coefs = coefs.view(batch, frames, self.df_bins, self.df_order, 2).permute(0, 1, 3, 2, 4)
        return coefs, alpha, state_out


class DfNet(nn.Module):
    """DeepFilterNet2 base model."""

    def __init__(
        self,
        *,
        sample_rate: int = 16000,
        win_size: int = 320,
        hop_size: int = 160,
        fft_size: int = 320,
        erb_bins: int = 24,
        df_bins: int = 96,
        df_order: int = 5,
        df_lookahead: int = 0,
        lookahead_frames: int = 2,
        conv_ch: int = 16,
        emb_hidden_dim: int = 256,
        df_hidden_dim: int = 256,
        df_num_layers: int = 3,
        gru_groups: int = 1,
        lin_groups: int = 1,
        condition_dim: int = 0,
        df_iters: int = 2,
    ) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.win_size = win_size
        self.hop_size = hop_size
        self.fft_size = fft_size
        self.freq_bins = fft_size // 2 + 1
        self.erb_bins = erb_bins
        self.df_bins = df_bins
        self.df_order = df_order
        self.df_lookahead = df_lookahead
        self.lookahead_frames = lookahead_frames
        self.conv_ch = conv_ch
        self.emb_hidden_dim = emb_hidden_dim
        self.condition_dim = condition_dim
        self.df_iters = df_iters

        fb = erb_fb(
            n_freqs=self.freq_bins,
            nb_bands=erb_bins,
            sample_rate=sample_rate,
            min_nb_freqs=2,
        )
        self.erb_fb: torch.Tensor
        self.register_buffer("erb_fb", fb)
        self.erb_inv_fb: torch.Tensor
        self.register_buffer("erb_inv_fb", erb_fb_inverse(fb))
        # A rectangular analysis window keeps center=False reconstruction simple
        # for both batch ISTFT and the streaming overlap-add path.
        self.window: torch.Tensor
        self.register_buffer("window", torch.ones(win_size), persistent=False)

        self.enc = Encoder(
            erb_bins=erb_bins,
            df_bins=df_bins,
            conv_ch=conv_ch,
            emb_hidden_dim=emb_hidden_dim,
            condition_dim=condition_dim,
            gru_groups=gru_groups,
            lin_groups=lin_groups,
        )
        self.erb_dec = ErbDecoder(
            erb_bins=erb_bins,
            conv_ch=conv_ch,
            emb_hidden_dim=emb_hidden_dim,
            gru_groups=gru_groups,
            lin_groups=lin_groups,
        )
        self.df_dec = DfDecoder(
            df_bins=df_bins,
            df_order=df_order,
            conv_ch=conv_ch,
            emb_hidden_dim=emb_hidden_dim,
            hidden_dim=df_hidden_dim,
            num_layers=df_num_layers,
            gru_groups=gru_groups,
        )
        self.df_op = DfOp(
            df_bins=df_bins,
            df_order=df_order,
            df_lookahead=df_lookahead,
            method="real_unfold",
        )

    def waveform_to_spec(
        self,
        waveform: torch.Tensor,
    ) -> tuple[torch.Tensor, int]:
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        if waveform.dim() != 2:
            raise ValueError(
                f"waveform must be [T] or [B, T], got {tuple(waveform.shape)}"
            )
        length = waveform.size(-1)
        if length <= self.fft_size:
            padded_length = self.fft_size
        else:
            steps = math.ceil((length - self.fft_size) / self.hop_size)
            padded_length = self.fft_size + max(0, steps) * self.hop_size
        pad_right = padded_length - length
        padded = F.pad(waveform, (0, pad_right))
        spec = torch.stft(
            padded,
            n_fft=self.fft_size,
            hop_length=self.hop_size,
            win_length=self.win_size,
            window=self.window.to(device=waveform.device, dtype=waveform.dtype),
            center=False,
            return_complex=True,
        ).transpose(1, 2)
        return torch.view_as_real(spec).unsqueeze(1), pad_right

    def spec_to_waveform(
        self,
        spec: torch.Tensor,
        *,
        length: int | None = None,
    ) -> torch.Tensor:
        if spec.dim() != 5 or spec.size(1) != 1 or spec.size(-1) != 2:
            raise ValueError(
                "spec must have shape [B, 1, T, F, 2], "
                f"got {tuple(spec.shape)}"
            )
        complex_spec = torch.view_as_complex(spec.squeeze(1).contiguous()).transpose(1, 2)
        return torch.istft(
            complex_spec,
            n_fft=self.fft_size,
            hop_length=self.hop_size,
            win_length=self.win_size,
            window=self.window.to(device=spec.device, dtype=spec.dtype),
            center=False,
            length=length,
        )

    def compute_features(
        self,
        spec: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if spec.dim() != 5 or spec.size(1) != 1 or spec.size(-1) != 2:
            raise ValueError(
                "spec must have shape [B, 1, T, F, 2], "
                f"got {tuple(spec.shape)}"
            )
        mag = torch.linalg.vector_norm(spec, dim=-1)
        erb_feat = torch.matmul(mag, self.erb_fb.transpose(0, 1))
        erb_feat = torch.log1p(erb_feat)
        df_feat = spec[..., : self.df_bins, :].squeeze(1).permute(0, 3, 1, 2)
        return erb_feat, df_feat

    def _apply_erb_mask(
        self,
        spec: torch.Tensor,
        gains: torch.Tensor,
    ) -> torch.Tensor:
        expanded = torch.matmul(gains.squeeze(1), self.erb_inv_fb).unsqueeze(1).unsqueeze(-1)
        return spec * expanded

    def _conditioning_to_frames(
        self,
        conditioning: torch.Tensor | None,
        frames: int,
    ) -> torch.Tensor | None:
        if self.condition_dim == 0:
            return None
        if conditioning is None:
            raise ValueError("conditioning is required for this model")
        if conditioning.dim() == 2:
            if conditioning.size(-1) != self.condition_dim:
                raise ValueError(
                    f"expected conditioning dim {self.condition_dim}, got {conditioning.size(-1)}"
                )
            conditioning = conditioning.unsqueeze(1).expand(-1, frames, -1)
        elif conditioning.dim() == 3:
            if conditioning.size(1) != frames:
                raise ValueError(
                    f"conditioning frames={conditioning.size(1)} do not match T={frames}"
                )
            if conditioning.size(-1) != self.condition_dim:
                raise ValueError(
                    f"expected conditioning dim {self.condition_dim}, got {conditioning.size(-1)}"
                )
        else:
            raise ValueError(
                "conditioning must be [B, D] or [B, T, D], "
                f"got {tuple(conditioning.shape)}"
            )
        return conditioning

    def forward(
        self,
        spec: torch.Tensor,
        conditioning: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        erb_feat, df_feat = self.compute_features(spec)
        e0, e1, e2, e3, emb, c0, lsnr, _ = self.enc(erb_feat, df_feat, conditioning)
        gains, _ = self.erb_dec(emb, e3, e2, e1, e0)
        enhanced = self._apply_erb_mask(spec, gains)
        coefs, alpha, _ = self.df_dec(emb, c0)
        for _ in range(self.df_iters):
            enhanced = self.df_op(enhanced, coefs, alpha)
        return enhanced, gains, lsnr, alpha

    def init_stream_state(
        self,
        batch_size: int,
        *,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> DfNetStreamState:
        return DfNetStreamState(
            erb_feat_buf=torch.zeros(batch_size, 1, 2, self.erb_bins, device=device, dtype=dtype),
            df_feat_buf=torch.zeros(batch_size, 2, 2, self.df_bins, device=device, dtype=dtype),
            enc_state=self.enc.emb_gru.get_h0(batch_size, device=device, dtype=dtype),
            erb_dec_state=self.erb_dec.emb_gru.get_h0(batch_size, device=device, dtype=dtype),
            df_dec_state=self.df_dec.df_gru.get_h0(batch_size, device=device, dtype=dtype),
            df_spec_bufs=tuple(
                torch.zeros(
                    batch_size,
                    1,
                    self.df_order,
                    self.df_bins,
                    2,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(self.df_iters)
            ),
        )

    def _conv0_frame(
        self,
        module: nn.Module,
        history: torch.Tensor,
        current: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        stacked = torch.cat((history, current), dim=2)
        output = module(stacked)[:, :, -1:, :]
        history = stacked[:, :, -2:, :]
        return output, history

    def _df_step(
        self,
        masked_frame: torch.Tensor,
        coefs: torch.Tensor,
        alpha: torch.Tensor,
        spec_buf: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        spec_buf = torch.roll(spec_buf, shifts=-1, dims=2)
        spec_buf[:, :, -1] = masked_frame[..., : self.df_bins, :].squeeze(2)
        real = spec_buf[..., 0] * coefs[..., 0] - spec_buf[..., 1] * coefs[..., 1]
        imag = spec_buf[..., 1] * coefs[..., 0] + spec_buf[..., 0] * coefs[..., 1]
        filtered = torch.stack((real, imag), dim=-1).sum(dim=2).unsqueeze(2)
        mix = alpha.view(masked_frame.size(0), 1, 1, 1, 1)
        output = masked_frame.clone()
        output[..., : self.df_bins, :] = (
            filtered * mix + masked_frame[..., : self.df_bins, :] * (1.0 - mix)
        )
        return output, spec_buf

    def stream_step(
        self,
        spec_frame: torch.Tensor,
        conditioning: torch.Tensor | None,
        state: DfNetStreamState,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, DfNetStreamState]:
        if spec_frame.shape[2] != 1:
            raise ValueError(
                f"spec_frame must contain exactly one frame, got {tuple(spec_frame.shape)}"
            )
        erb_feat, df_feat = self.compute_features(spec_frame)
        e0, state.erb_feat_buf = self._conv0_frame(self.enc.erb_conv0, state.erb_feat_buf, erb_feat)
        e1 = self.enc.erb_conv1(e0)
        e2 = self.enc.erb_conv2(e1)
        e3 = self.enc.erb_conv3(e2)

        c0, state.df_feat_buf = self._conv0_frame(self.enc.df_conv0, state.df_feat_buf, df_feat)
        c1 = self.enc.df_conv1(c0)
        cemb = self.enc.df_fc_emb(c1.permute(0, 2, 3, 1).flatten(2))
        emb = e3.permute(0, 2, 3, 1).flatten(2) + cemb
        cond = self._conditioning_to_frames(conditioning, 1)
        if cond is not None:
            emb = torch.cat((emb, cond), dim=-1)
        emb, state.enc_state = self.enc.emb_gru(emb, state.enc_state)
        lsnr = self.enc.lsnr_fc(emb) * (self.enc.lsnr_max - self.enc.lsnr_min) + self.enc.lsnr_min

        gains, state.erb_dec_state = self.erb_dec(emb, e3, e2, e1, e0, state.erb_dec_state)
        masked = self._apply_erb_mask(spec_frame, gains)
        coefs, alpha, state.df_dec_state = self.df_dec(emb, c0, state.df_dec_state)
        enhanced = masked
        next_bufs: list[torch.Tensor] = []
        for idx in range(self.df_iters):
            enhanced, buf = self._df_step(
                enhanced, coefs[:, 0], alpha[:, 0], state.df_spec_bufs[idx],
            )
            next_bufs.append(buf)
        state.df_spec_bufs = tuple(next_bufs)
        return enhanced, gains, lsnr, alpha, state


__all__ = [
    "DfNetStreamState",
    "Encoder",
    "ErbDecoder",
    "DfDecoder",
    "DfNet",
]
