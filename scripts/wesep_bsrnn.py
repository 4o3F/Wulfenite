from __future__ import annotations

import argparse
import json
import math
import time
from functools import lru_cache
from pathlib import Path

import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F


EPS = 1e-8
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_CHECKPOINT = "vendor/wesep/pretrained/english/avg_model.pt"


def read_wav_mono(path: str | Path) -> tuple[torch.Tensor, int]:
    audio, sample_rate = sf.read(str(path), always_2d=True, dtype="float32")
    audio = torch.from_numpy(audio).transpose(0, 1).contiguous()
    if audio.size(0) > 1:
        audio = audio.mean(dim=0, keepdim=True)
    return audio.squeeze(0), sample_rate


def write_wav_mono(path: str | Path, audio: torch.Tensor, sample_rate: int) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    audio = audio.detach().cpu().clamp(-1.0, 1.0 - 1.0 / 32768.0)
    sf.write(str(path), audio.numpy(), sample_rate, subtype="PCM_16")


def resample_mono(audio: torch.Tensor, orig_sr: int, target_sr: int) -> torch.Tensor:
    if orig_sr == target_sr:
        return audio
    new_length = int(round(audio.numel() * target_sr / orig_sr))
    audio = audio.view(1, 1, -1)
    audio = F.interpolate(audio, size=new_length, mode="linear", align_corners=False)
    return audio.view(-1)


def upper_power_of_two(n: int) -> int:
    return 1 << (n - 1).bit_length()


def hz_to_mel(freq: float) -> float:
    return 1127.0 * math.log(1.0 + freq / 700.0)


def mel_to_hz(mel_freq: float) -> float:
    return 700.0 * (math.exp(mel_freq / 1127.0) - 1.0)


@lru_cache(maxsize=None)
def build_mel_filter_bank(
    sample_rate: int,
    frame_length: int,
    num_mel_bins: int,
    low_freq: float = 20.0,
    high_freq: float | None = None,
) -> torch.Tensor:
    if high_freq is None:
        high_freq = sample_rate / 2.0

    n_fft = upper_power_of_two(frame_length)
    num_fft_bins = n_fft // 2
    fft_bin_width = sample_rate / n_fft

    mel_low = hz_to_mel(low_freq)
    mel_high = hz_to_mel(high_freq)
    mel_delta = (mel_high - mel_low) / (num_mel_bins + 1)

    filters = torch.zeros(num_mel_bins, num_fft_bins, dtype=torch.float32)
    for mel_bin in range(num_mel_bins):
        left_mel = mel_low + mel_bin * mel_delta
        center_mel = mel_low + (mel_bin + 1) * mel_delta
        right_mel = mel_low + (mel_bin + 2) * mel_delta
        for fft_bin in range(num_fft_bins):
            mel = hz_to_mel(fft_bin * fft_bin_width)
            if mel <= left_mel or mel >= right_mel:
                continue
            if mel <= center_mel:
                weight = (mel - left_mel) / (center_mel - left_mel)
            else:
                weight = (right_mel - mel) / (right_mel - center_mel)
            filters[mel_bin, fft_bin] = weight
    return filters


def compute_log_mel_fbank(
    waveform: torch.Tensor,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    num_mel_bins: int = 80,
    frame_length_ms: int = 25,
    frame_shift_ms: int = 10,
) -> torch.Tensor:
    waveform = waveform.to(torch.float32)
    if waveform.dim() != 1:
        waveform = waveform.view(-1)

    frame_length = int(sample_rate * frame_length_ms / 1000)
    frame_shift = int(sample_rate * frame_shift_ms / 1000)
    n_fft = upper_power_of_two(frame_length)

    if waveform.numel() < frame_length:
        waveform = F.pad(waveform, (0, frame_length - waveform.numel()))

    num_frames = 1 + (waveform.numel() - frame_length) // frame_shift
    frames = waveform[: frame_length + (num_frames - 1) * frame_shift].unfold(0, frame_length, frame_shift)
    frames = frames - frames.mean(dim=1, keepdim=True)

    emphasized = frames.clone()
    emphasized[:, 1:] = emphasized[:, 1:] - 0.97 * frames[:, :-1]
    emphasized[:, 0] = emphasized[:, 0] - 0.97 * frames[:, 0]

    window = torch.hamming_window(frame_length, periodic=False, dtype=waveform.dtype, device=waveform.device)
    emphasized = emphasized * window
    if n_fft > frame_length:
        emphasized = F.pad(emphasized, (0, n_fft - frame_length))

    spectrum = torch.fft.rfft(emphasized, n=n_fft, dim=-1)
    power = spectrum.real.square() + spectrum.imag.square()
    power = power[:, : n_fft // 2]

    mel_fb = build_mel_filter_bank(sample_rate, frame_length, num_mel_bins).to(power.device, power.dtype)
    feats = power @ mel_fb.transpose(0, 1)
    feats = torch.log(feats.clamp_min(torch.finfo(feats.dtype).eps))
    feats = feats - feats.mean(dim=0, keepdim=True)
    return feats


class LinearLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class SpeakerFuseLayer(nn.Module):
    def __init__(self, embed_dim: int, feat_dim: int, fuse_type: str = "multiply"):
        super().__init__()
        if fuse_type not in {"concat", "additive", "multiply"}:
            raise ValueError(f"Unsupported fuse type: {fuse_type}")

        self.fuse_type = fuse_type
        if fuse_type == "concat":
            self.fc = LinearLayer(embed_dim + feat_dim, feat_dim)
        else:
            self.fc = LinearLayer(embed_dim, feat_dim)

    def forward(self, x: torch.Tensor, embed: torch.Tensor) -> torch.Tensor:
        if self.fuse_type == "concat":
            embed_t = embed.expand(-1, x.size(1), -1, x.size(3))
            y = torch.cat([x, embed_t], dim=2)
            y = torch.transpose(y, 2, 3)
            return torch.transpose(self.fc(y), 2, 3).contiguous()

        embed_t = embed.expand(-1, x.size(1), -1, x.size(3))
        embed_t = torch.transpose(embed_t, 2, 3)
        transformed = torch.transpose(self.fc(embed_t), 2, 3)
        if self.fuse_type == "additive":
            return x + transformed
        return x * transformed


class ResRNN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, bidirectional: bool = True):
        super().__init__()
        eps = torch.finfo(torch.float32).eps
        self.norm = nn.GroupNorm(1, input_size, eps)
        self.rnn = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.proj = nn.Linear(hidden_size * 2, input_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rnn_output, _ = self.rnn(self.norm(x).transpose(1, 2).contiguous())
        batch_size, seq_len, hidden_dim = rnn_output.shape
        rnn_output = self.proj(rnn_output.reshape(batch_size * seq_len, hidden_dim))
        rnn_output = rnn_output.view(batch_size, seq_len, -1)
        return x + rnn_output.transpose(1, 2).contiguous()


class BSNet(nn.Module):
    def __init__(self, in_channel: int, nband: int, bidirectional: bool = True):
        super().__init__()
        self.nband = nband
        self.feature_dim = in_channel // nband
        self.band_rnn = ResRNN(self.feature_dim, self.feature_dim * 2, bidirectional=bidirectional)
        self.band_comm = ResRNN(self.feature_dim, self.feature_dim * 2, bidirectional=bidirectional)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, seq_len = x.shape
        band_output = self.band_rnn(x.view(batch_size * self.nband, self.feature_dim, -1))
        band_output = band_output.view(batch_size, self.nband, -1, seq_len)
        band_output = band_output.permute(0, 3, 2, 1).contiguous().view(batch_size * seq_len, -1, self.nband)
        output = self.band_comm(band_output)
        output = output.view(batch_size, seq_len, -1, self.nband).permute(0, 3, 2, 1).contiguous()
        return output.view(batch_size, self.nband * self.feature_dim, seq_len)


class FuseSeparation(nn.Module):
    def __init__(
        self,
        nband: int,
        num_repeat: int,
        feature_dim: int,
        spk_emb_dim: int,
        spk_fuse_type: str,
        multi_fuse: bool,
    ):
        super().__init__()
        self.multi_fuse = multi_fuse
        self.nband = nband
        self.feature_dim = feature_dim
        self.separation = nn.ModuleList()

        self.separation.append(
            SpeakerFuseLayer(embed_dim=spk_emb_dim, feat_dim=feature_dim, fuse_type=spk_fuse_type)
        )
        for _ in range(num_repeat):
            self.separation.append(BSNet(nband * feature_dim, nband))

    def forward(self, x: torch.Tensor, spk_embedding: torch.Tensor) -> torch.Tensor:
        if self.multi_fuse:
            raise ValueError("This standalone script supports the released multi_fuse=False checkpoint only.")

        batch_size = x.size(0)
        x = self.separation[0](x, spk_embedding)
        x = x.view(batch_size, self.nband * self.feature_dim, -1)
        for separator in self.separation[1:]:
            x = separator(x)
        return x.view(batch_size, self.nband, self.feature_dim, -1)


class Conv1dReluBn(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            bias=bias,
        )
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(F.relu(self.conv(x)))


class Res2Conv1dReluBn(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        dilation: int,
        bias: bool = True,
        scale: int = 4,
    ):
        super().__init__()
        if channels % scale != 0:
            raise ValueError(f"{channels} must be divisible by {scale}")

        self.scale = scale
        self.width = channels // scale
        self.nums = scale if scale == 1 else scale - 1
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _ in range(self.nums):
            self.convs.append(
                nn.Conv1d(
                    self.width,
                    self.width,
                    kernel_size,
                    stride,
                    padding,
                    dilation,
                    bias=bias,
                )
            )
            self.bns.append(nn.BatchNorm1d(self.width))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = []
        split_x = torch.split(x, self.width, dim=1)
        state = split_x[0]
        for idx, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            if idx >= 1:
                state = state + split_x[idx]
            state = bn(F.relu(conv(state)))
            out.append(state)
        if self.scale != 1:
            out.append(split_x[self.nums])
        return torch.cat(out, dim=1)


class SEConnect(nn.Module):
    def __init__(self, channels: int, se_bottleneck_dim: int = 128):
        super().__init__()
        self.linear1 = nn.Linear(channels, se_bottleneck_dim)
        self.linear2 = nn.Linear(se_bottleneck_dim, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = x.mean(dim=2)
        pooled = F.relu(self.linear1(pooled))
        pooled = torch.sigmoid(self.linear2(pooled))
        return x * pooled.unsqueeze(2)


class SERes2Block(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        dilation: int,
        scale: int,
    ):
        super().__init__()
        self.se_res2block = nn.Sequential(
            Conv1dReluBn(channels, channels, kernel_size=1, stride=1, padding=0),
            Res2Conv1dReluBn(
                channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                scale=scale,
            ),
            Conv1dReluBn(channels, channels, kernel_size=1, stride=1, padding=0),
            SEConnect(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.se_res2block(x)


class ASTP(nn.Module):
    def __init__(self, in_dim: int, bottleneck_dim: int = 128, global_context_att: bool = False):
        super().__init__()
        self.in_dim = in_dim
        self.global_context_att = global_context_att
        attn_in_dim = in_dim * 3 if global_context_att else in_dim
        self.linear1 = nn.Conv1d(attn_in_dim, bottleneck_dim, kernel_size=1)
        self.linear2 = nn.Conv1d(bottleneck_dim, in_dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.global_context_att:
            context_mean = x.mean(dim=-1, keepdim=True).expand_as(x)
            context_std = torch.sqrt(torch.var(x, dim=-1, keepdim=True) + 1e-7).expand_as(x)
            x_in = torch.cat([x, context_mean, context_std], dim=1)
        else:
            x_in = x

        alpha = torch.tanh(self.linear1(x_in))
        alpha = torch.softmax(self.linear2(alpha), dim=2)
        mean = torch.sum(alpha * x, dim=2)
        var = torch.sum(alpha * (x**2), dim=2) - mean**2
        std = torch.sqrt(var.clamp_min(1e-7))
        return torch.cat([mean, std], dim=1)


class ECAPATDNN(nn.Module):
    def __init__(
        self,
        channels: int = 512,
        feat_dim: int = 80,
        embed_dim: int = 192,
        global_context_att: bool = True,
    ):
        super().__init__()
        self.layer1 = Conv1dReluBn(feat_dim, channels, kernel_size=5, padding=2)
        self.layer2 = SERes2Block(channels, kernel_size=3, stride=1, padding=2, dilation=2, scale=8)
        self.layer3 = SERes2Block(channels, kernel_size=3, stride=1, padding=3, dilation=3, scale=8)
        self.layer4 = SERes2Block(channels, kernel_size=3, stride=1, padding=4, dilation=4, scale=8)
        self.conv = nn.Conv1d(channels * 3, channels * 3, kernel_size=1)
        self.pool = ASTP(in_dim=channels * 3, global_context_att=global_context_att)
        self.bn = nn.BatchNorm1d(channels * 6)
        self.linear = nn.Linear(channels * 6, embed_dim)

    def _get_frame_level_feat(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = x.permute(0, 2, 1)
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        out = torch.cat([out2, out3, out4], dim=1)
        out = self.conv(out)
        return out, out4

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out, out4 = self._get_frame_level_feat(x)
        out = F.relu(out)
        out = self.bn(self.pool(out))
        out = self.linear(out)
        return out4, out


class WeSepBSRNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.sr = 16000
        self.win = 512
        self.stride = 128
        self.enc_dim = self.win // 2 + 1
        self.feature_dim = 128
        self.spk_emb_dim = 192

        bandwidth_100 = math.floor(100 / (self.sr / 2.0) * self.enc_dim)
        bandwidth_200 = math.floor(200 / (self.sr / 2.0) * self.enc_dim)
        bandwidth_500 = math.floor(500 / (self.sr / 2.0) * self.enc_dim)
        bandwidth_2k = math.floor(2000 / (self.sr / 2.0) * self.enc_dim)

        self.band_width = [bandwidth_100] * 15
        self.band_width += [bandwidth_200] * 10
        self.band_width += [bandwidth_500] * 5
        self.band_width += [bandwidth_2k]
        self.band_width.append(self.enc_dim - sum(self.band_width))
        self.nband = len(self.band_width)

        eps = torch.finfo(torch.float32).eps
        self.spk_model = ECAPATDNN(channels=512, feat_dim=80, embed_dim=192, global_context_att=True)

        self.BN = nn.ModuleList()
        for band_width in self.band_width:
            self.BN.append(
                nn.Sequential(
                    nn.GroupNorm(1, band_width * 2, eps),
                    nn.Conv1d(band_width * 2, self.feature_dim, 1),
                )
            )

        self.separator = FuseSeparation(
            nband=self.nband,
            num_repeat=6,
            feature_dim=self.feature_dim,
            spk_emb_dim=self.spk_emb_dim,
            spk_fuse_type="multiply",
            multi_fuse=False,
        )

        self.mask = nn.ModuleList()
        for band_width in self.band_width:
            self.mask.append(
                nn.Sequential(
                    nn.GroupNorm(1, self.feature_dim, eps),
                    nn.Conv1d(self.feature_dim, self.feature_dim * 4, 1),
                    nn.Tanh(),
                    nn.Conv1d(self.feature_dim * 4, self.feature_dim * 4, 1),
                    nn.Tanh(),
                    nn.Conv1d(self.feature_dim * 4, band_width * 4, 1),
                )
            )

    def forward(self, mixture: torch.Tensor, enrollment_feat: torch.Tensor) -> torch.Tensor:
        if mixture.dim() != 2:
            raise ValueError(f"Expected mixture shape [B, T], got {tuple(mixture.shape)}")
        if enrollment_feat.dim() != 3:
            raise ValueError(f"Expected enrollment features shape [B, T, 80], got {tuple(enrollment_feat.shape)}")

        batch_size, num_samples = mixture.shape
        window = torch.hann_window(self.win, dtype=mixture.dtype, device=mixture.device)
        spec = torch.stft(
            mixture,
            n_fft=self.win,
            hop_length=self.stride,
            window=window,
            return_complex=True,
        )

        spec_ri = torch.stack([spec.real, spec.imag], dim=1)
        subband_spec = []
        subband_mix_spec = []
        band_idx = 0
        for band_width in self.band_width:
            subband_spec.append(spec_ri[:, :, band_idx : band_idx + band_width].contiguous())
            subband_mix_spec.append(spec[:, band_idx : band_idx + band_width])
            band_idx += band_width

        subband_feature = []
        for band, bn in zip(subband_spec, self.BN):
            subband_feature.append(bn(band.view(batch_size, band.size(1) * band.size(2), -1)))
        subband_feature = torch.stack(subband_feature, dim=1)

        spk_embedding = self.spk_model(enrollment_feat)[-1]
        spk_embedding = spk_embedding.unsqueeze(1).unsqueeze(3)
        sep_output = self.separator(subband_feature, spk_embedding)

        separated_bands = []
        for idx, mask_head in enumerate(self.mask):
            band_out = mask_head(sep_output[:, idx]).view(batch_size, 2, 2, self.band_width[idx], -1)
            mask = band_out[:, 0] * torch.sigmoid(band_out[:, 1])
            mask_real = mask[:, 0]
            mask_imag = mask[:, 1]
            est_real = subband_mix_spec[idx].real * mask_real - subband_mix_spec[idx].imag * mask_imag
            est_imag = subband_mix_spec[idx].real * mask_imag + subband_mix_spec[idx].imag * mask_real
            separated_bands.append(torch.complex(est_real, est_imag))

        est_spec = torch.cat(separated_bands, dim=1)
        output = torch.istft(
            est_spec.view(batch_size, self.enc_dim, -1),
            n_fft=self.win,
            hop_length=self.stride,
            window=window,
            length=num_samples,
        )
        return output


def load_released_checkpoint(model: nn.Module, checkpoint_path: str | Path) -> None:
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    if isinstance(checkpoint, dict) and "models" in checkpoint:
        state_dict = checkpoint["models"][0]
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict, strict=True)


def build_model(device: torch.device, checkpoint_path: str | Path) -> WeSepBSRNN:
    model = WeSepBSRNN()
    load_released_checkpoint(model, checkpoint_path)
    model.to(device)
    model.eval()
    return model


def normalize_output(audio: torch.Tensor) -> torch.Tensor:
    peak = audio.abs().max().clamp_min(EPS)
    return audio / peak * 0.9


def si_sdr(estimate: torch.Tensor, reference: torch.Tensor) -> float:
    estimate = estimate.view(-1).to(torch.float32)
    reference = reference.view(-1).to(torch.float32)
    reference_energy = torch.sum(reference**2).clamp_min(EPS)
    projection = torch.sum(estimate * reference) * reference / reference_energy
    noise = estimate - projection
    value = 10.0 * torch.log10(torch.sum(projection**2).clamp_min(EPS) / torch.sum(noise**2).clamp_min(EPS))
    return float(value.item())


def compare_waveforms(estimate: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    length = min(estimate.numel(), reference.numel())
    estimate = estimate[:length]
    reference = reference[:length]
    diff = estimate - reference
    mse = torch.mean(diff**2).item()
    mae = torch.mean(diff.abs()).item()
    max_abs = torch.max(diff.abs()).item()
    rel_l2 = (torch.norm(diff) / torch.norm(reference).clamp_min(EPS)).item()
    return {
        "samples_compared": length,
        "mae": float(mae),
        "mse": float(mse),
        "max_abs": float(max_abs),
        "relative_l2": float(rel_l2),
        "si_sdr_db": si_sdr(estimate, reference),
    }


def separate(
    model: WeSepBSRNN,
    mixture_path: str | Path,
    enrollment_path: str | Path,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    mixture, mixture_sr = read_wav_mono(mixture_path)
    enrollment, enrollment_sr = read_wav_mono(enrollment_path)

    mixture = resample_mono(mixture, mixture_sr, DEFAULT_SAMPLE_RATE)
    enrollment = resample_mono(enrollment, enrollment_sr, DEFAULT_SAMPLE_RATE)
    enrollment_feat = compute_log_mel_fbank(enrollment, sample_rate=DEFAULT_SAMPLE_RATE).unsqueeze(0)

    mixture = mixture.unsqueeze(0).to(device)
    enrollment_feat = enrollment_feat.to(device)

    start = time.perf_counter()
    with torch.no_grad():
        separated = model(mixture, enrollment_feat)[0].cpu()
    elapsed = time.perf_counter() - start

    normalized = normalize_output(separated)
    report = {
        "sample_rate": DEFAULT_SAMPLE_RATE,
        "mixture_seconds": mixture.size(1) / DEFAULT_SAMPLE_RATE,
        "elapsed_seconds": elapsed,
        "rtf": elapsed / (mixture.size(1) / DEFAULT_SAMPLE_RATE),
        "faster_than_real_time": (mixture.size(1) / DEFAULT_SAMPLE_RATE) / elapsed,
    }
    return normalized, report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Single-file PyTorch-only reimplementation of the released WeSep BSRNN + ECAPA checkpoint."
    )
    parser.add_argument("--mixture", required=True, help="Path to the mixed waveform.")
    parser.add_argument("--enrollment", required=True, help="Path to the target-speaker enrollment waveform.")
    parser.add_argument("--output", required=True, help="Path for the separated waveform.")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT, help="Released WeSep checkpoint path.")
    parser.add_argument("--reference", help="Optional reference output to compare against.")
    parser.add_argument("--report-path", help="Optional JSON output path.")
    parser.add_argument("--device", default="cpu", help="Torch device, e.g. cpu or cuda.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    model = build_model(device, args.checkpoint)
    separated, report = separate(model, args.mixture, args.enrollment, device)
    write_wav_mono(args.output, separated, DEFAULT_SAMPLE_RATE)

    result = {
        "checkpoint": str(args.checkpoint),
        "mixture": str(args.mixture),
        "enrollment": str(args.enrollment),
        "output": str(args.output),
        **report,
    }

    if args.reference:
        reference, reference_sr = read_wav_mono(args.reference)
        if reference_sr != DEFAULT_SAMPLE_RATE:
            reference = resample_mono(reference, reference_sr, DEFAULT_SAMPLE_RATE)
        result["reference"] = str(args.reference)
        result["comparison"] = compare_waveforms(separated, reference)

    print(json.dumps(result, indent=2))
    if args.report_path:
        report_path = Path(args.report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
