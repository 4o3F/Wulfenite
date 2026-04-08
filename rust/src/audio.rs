// Model-agnostic audio utilities: STFT/iSTFT with cached FFT planners,
// log-mel FBank, resampling, WAV I/O. Kept after the v1 -> main reset
// because none of these are tied to a specific model architecture.
// `dead_code` is allowed at the module level until new code wires it in.
#![allow(dead_code)]

use std::{borrow::Cow, f32::consts::PI, fs, path::Path, sync::Arc};

use anyhow::{Context, Result, ensure};
use hound::{SampleFormat, WavReader, WavSpec, WavWriter};
use rustfft::{Fft, FftPlanner, num_complex::Complex32};

const EPS: f32 = 1e-8;

pub const DEFAULT_SAMPLE_RATE: usize = 16_000;
pub const DEFAULT_NUM_MEL_BINS: usize = 80;
pub const DEFAULT_FRAME_LENGTH_MS: usize = 25;
pub const DEFAULT_FRAME_SHIFT_MS: usize = 10;

#[derive(Debug, Clone)]
pub struct MonoAudio {
    pub sample_rate: usize,
    pub samples: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct FeatureMatrix {
    pub frames: usize,
    pub bins: usize,
    pub data: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct ComplexSpectrogram {
    pub freq_bins: usize,
    pub frames: usize,
    pub real: Vec<f32>,
    pub imag: Vec<f32>,
}

/// Reusable STFT/iSTFT state — owns the FFT plans, hann window, and scratch
/// buffers so streaming callers do not pay planner/allocation cost per chunk.
///
/// `n_fft` and `hop_length` are fixed for the lifetime of the state. Use one
/// state per (n_fft, hop_length) pair.
pub struct StftState {
    n_fft: usize,
    hop_length: usize,
    fft_forward: Arc<dyn Fft<f32>>,
    fft_inverse: Arc<dyn Fft<f32>>,
    window: Vec<f32>,
    fwd_scratch: Vec<Complex32>,
    inv_scratch: Vec<Complex32>,
    frame_buf: Vec<Complex32>,
    padded: Vec<f32>,
    out_time: Vec<f32>,
    envelope: Vec<f32>,
}

impl StftState {
    pub fn new(n_fft: usize, hop_length: usize) -> Self {
        assert!(n_fft > 0, "n_fft must be greater than zero");
        assert!(hop_length > 0, "hop_length must be greater than zero");
        let mut planner = FftPlanner::<f32>::new();
        let fft_forward = planner.plan_fft_forward(n_fft);
        let fft_inverse = planner.plan_fft_inverse(n_fft);
        let fwd_scratch = vec![Complex32::new(0.0, 0.0); fft_forward.get_inplace_scratch_len()];
        let inv_scratch = vec![Complex32::new(0.0, 0.0); fft_inverse.get_inplace_scratch_len()];
        let window = hann_window(n_fft, true);
        Self {
            n_fft,
            hop_length,
            fft_forward,
            fft_inverse,
            window,
            fwd_scratch,
            inv_scratch,
            frame_buf: vec![Complex32::new(0.0, 0.0); n_fft],
            padded: Vec::new(),
            out_time: Vec::new(),
            envelope: Vec::new(),
        }
    }

    /// Forward STFT into the provided spectrogram buffer. Reuses internal
    /// scratch and writes results into `out` via in-place resize.
    pub fn stft_into(&mut self, waveform: &[f32], out: &mut ComplexSpectrogram) -> Result<()> {
        ensure!(!waveform.is_empty(), "waveform must not be empty");

        let n_fft = self.n_fft;
        let hop_length = self.hop_length;
        let pad = n_fft / 2;
        reflect_pad_into(waveform, pad, &mut self.padded);
        let frames = 1 + (self.padded.len() - n_fft) / hop_length;
        let freq_bins = n_fft / 2 + 1;

        out.freq_bins = freq_bins;
        out.frames = frames;
        out.real.clear();
        out.real.resize(freq_bins * frames, 0.0);
        out.imag.clear();
        out.imag.resize(freq_bins * frames, 0.0);

        for frame_idx in 0..frames {
            let start = frame_idx * hop_length;
            let frame = &self.padded[start..start + n_fft];
            for (slot, (sample, window)) in self
                .frame_buf
                .iter_mut()
                .zip(frame.iter().zip(self.window.iter()))
            {
                *slot = Complex32::new(sample * window, 0.0);
            }
            self.fft_forward
                .process_with_scratch(&mut self.frame_buf, &mut self.fwd_scratch);

            for freq in 0..freq_bins {
                let value = self.frame_buf[freq];
                out.real[freq * frames + frame_idx] = value.re;
                out.imag[freq * frames + frame_idx] = value.im;
            }
        }

        Ok(())
    }

    /// Inverse STFT into the provided output buffer. Reuses internal scratch
    /// and writes the time-domain signal into `out` (resized to `output_len`).
    pub fn istft_into(
        &mut self,
        spectrogram: &ComplexSpectrogram,
        output_len: usize,
        out: &mut Vec<f32>,
    ) -> Result<()> {
        let n_fft = self.n_fft;
        let hop_length = self.hop_length;
        ensure!(
            spectrogram.freq_bins == n_fft / 2 + 1,
            "expected {} frequency bins, got {}",
            n_fft / 2 + 1,
            spectrogram.freq_bins
        );

        let total_len = n_fft + hop_length * spectrogram.frames.saturating_sub(1);
        self.out_time.clear();
        self.out_time.resize(total_len, 0.0);
        self.envelope.clear();
        self.envelope.resize(total_len, 0.0);

        let inv_scale = 1.0 / n_fft as f32;
        for frame_idx in 0..spectrogram.frames {
            for slot in self.frame_buf.iter_mut() {
                *slot = Complex32::new(0.0, 0.0);
            }
            for freq in 0..spectrogram.freq_bins {
                self.frame_buf[freq] = Complex32::new(
                    spectrogram.real[freq * spectrogram.frames + frame_idx],
                    spectrogram.imag[freq * spectrogram.frames + frame_idx],
                );
            }
            for freq in 1..spectrogram.freq_bins.saturating_sub(1) {
                self.frame_buf[n_fft - freq] = self.frame_buf[freq].conj();
            }

            self.fft_inverse
                .process_with_scratch(&mut self.frame_buf, &mut self.inv_scratch);

            let start = frame_idx * hop_length;
            for index in 0..n_fft {
                let sample = self.frame_buf[index].re * inv_scale;
                let weighted = sample * self.window[index];
                self.out_time[start + index] += weighted;
                self.envelope[start + index] += self.window[index] * self.window[index];
            }
        }

        for (sample, weight) in self.out_time.iter_mut().zip(self.envelope.iter()) {
            if *weight > EPS {
                *sample /= *weight;
            }
        }

        let pad = n_fft / 2;
        out.clear();
        out.reserve(output_len);
        if pad < self.out_time.len() {
            let available = self.out_time.len() - pad;
            let take = output_len.min(available);
            out.extend_from_slice(&self.out_time[pad..pad + take]);
        }
        if out.len() < output_len {
            out.resize(output_len, 0.0);
        }

        Ok(())
    }
}

pub fn read_wav_mono(path: &Path) -> Result<MonoAudio> {
    let mut reader =
        WavReader::open(path).with_context(|| format!("failed to open wav {}", path.display()))?;
    let spec = reader.spec();
    ensure!(
        spec.channels > 0,
        "wav {} has zero channels",
        path.display()
    );

    let channels = spec.channels as usize;
    let interleaved = match spec.sample_format {
        SampleFormat::Float => reader
            .samples::<f32>()
            .collect::<std::result::Result<Vec<_>, _>>()
            .with_context(|| format!("failed to read float samples from {}", path.display()))?,
        SampleFormat::Int => {
            let scale = (1_i64 << (spec.bits_per_sample.saturating_sub(1) as u32)) as f32;
            reader
                .samples::<i32>()
                .map(|sample| sample.map(|value| value as f32 / scale))
                .collect::<std::result::Result<Vec<_>, _>>()
                .with_context(|| format!("failed to read PCM samples from {}", path.display()))?
        }
    };

    ensure!(
        !interleaved.is_empty(),
        "wav {} contains no samples",
        path.display()
    );

    let frames = interleaved.len() / channels;
    let mut mono = vec![0.0; frames];
    for (index, sample) in interleaved.into_iter().enumerate() {
        mono[index / channels] += sample;
    }
    if channels > 1 {
        let scale = 1.0 / channels as f32;
        for sample in mono.iter_mut() {
            *sample *= scale;
        }
    }

    Ok(MonoAudio {
        sample_rate: spec.sample_rate as usize,
        samples: mono,
    })
}

pub fn write_wav_mono(path: &Path, samples: &[f32], sample_rate: usize) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }

    let spec = WavSpec {
        channels: 1,
        sample_rate: sample_rate as u32,
        bits_per_sample: 16,
        sample_format: SampleFormat::Int,
    };
    let mut writer = WavWriter::create(path, spec)
        .with_context(|| format!("failed to create wav {}", path.display()))?;

    for &sample in samples {
        let clamped = sample.clamp(-1.0, 1.0 - 1.0 / 32768.0);
        let pcm = (clamped * 32768.0).round() as i16;
        writer
            .write_sample(pcm)
            .with_context(|| format!("failed to write wav {}", path.display()))?;
    }

    writer
        .finalize()
        .with_context(|| format!("failed to finalize wav {}", path.display()))
}

/// Borrowing variant of [`resample_mono`]: returns the input slice unchanged
/// when the source and target sample rates match. Used by the streaming worker
/// to avoid an allocation/copy on the equal-rate fast path.
pub fn resample_mono_borrowed<'a>(
    samples: &'a [f32],
    orig_sr: usize,
    target_sr: usize,
) -> Cow<'a, [f32]> {
    if orig_sr == target_sr {
        Cow::Borrowed(samples)
    } else {
        Cow::Owned(resample_mono(samples, orig_sr, target_sr))
    }
}

pub fn resample_mono(samples: &[f32], orig_sr: usize, target_sr: usize) -> Vec<f32> {
    if orig_sr == target_sr {
        return samples.to_vec();
    }
    if samples.is_empty() {
        return Vec::new();
    }

    let new_len = ((samples.len() as f64) * (target_sr as f64) / (orig_sr as f64)).round() as usize;
    let new_len = new_len.max(1);
    let input_len = samples.len() as f32;
    let output_len = new_len as f32;
    let mut output = Vec::with_capacity(new_len);

    for index in 0..new_len {
        let position = (index as f32 + 0.5) * input_len / output_len - 0.5;
        let left = position.floor() as isize;
        let right = left + 1;
        let weight_right = position - left as f32;
        let left_idx = left.clamp(0, samples.len() as isize - 1) as usize;
        let right_idx = right.clamp(0, samples.len() as isize - 1) as usize;
        output.push(samples[left_idx] * (1.0 - weight_right) + samples[right_idx] * weight_right);
    }

    output
}

pub fn compute_log_mel_fbank(
    waveform: &[f32],
    sample_rate: usize,
    num_mel_bins: usize,
) -> FeatureMatrix {
    let mut waveform = waveform.to_vec();
    let frame_length = sample_rate * DEFAULT_FRAME_LENGTH_MS / 1000;
    let frame_shift = sample_rate * DEFAULT_FRAME_SHIFT_MS / 1000;
    let n_fft = upper_power_of_two(frame_length);

    if waveform.len() < frame_length {
        waveform.resize(frame_length, 0.0);
    }

    let num_frames = 1 + (waveform.len() - frame_length) / frame_shift;
    let usable = frame_length + (num_frames - 1) * frame_shift;
    waveform.truncate(usable);

    let window = hamming_window(frame_length, false);
    let mel_filter_bank = build_mel_filter_bank(sample_rate, frame_length, num_mel_bins);
    let num_fft_bins = n_fft / 2;
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(n_fft);
    let mut feats = vec![0.0; num_frames * num_mel_bins];

    for frame_idx in 0..num_frames {
        let start = frame_idx * frame_shift;
        let frame = &waveform[start..start + frame_length];
        let mean = frame.iter().sum::<f32>() / frame_length as f32;

        let mut emphasized = vec![0.0; n_fft];
        if !frame.is_empty() {
            emphasized[0] = (frame[0] - mean) * (1.0 - 0.97) * window[0];
        }
        for index in 1..frame_length {
            let centered = frame[index] - mean;
            let prev_centered = frame[index - 1] - mean;
            emphasized[index] = (centered - 0.97 * prev_centered) * window[index];
        }

        let mut spectrum: Vec<_> = emphasized
            .into_iter()
            .map(|sample| Complex32::new(sample, 0.0))
            .collect();
        fft.process(&mut spectrum);

        let mut power = vec![0.0; num_fft_bins];
        for bin in 0..num_fft_bins {
            let value = spectrum[bin];
            power[bin] = value.re.mul_add(value.re, value.im * value.im);
        }

        for mel_bin in 0..num_mel_bins {
            let filter = &mel_filter_bank[mel_bin * num_fft_bins..(mel_bin + 1) * num_fft_bins];
            let energy = power
                .iter()
                .zip(filter.iter())
                .map(|(power, weight)| power * weight)
                .sum::<f32>()
                .max(f32::EPSILON);
            feats[frame_idx * num_mel_bins + mel_bin] = energy.ln();
        }
    }

    for mel_bin in 0..num_mel_bins {
        let mean = (0..num_frames)
            .map(|frame| feats[frame * num_mel_bins + mel_bin])
            .sum::<f32>()
            / num_frames as f32;
        for frame in 0..num_frames {
            feats[frame * num_mel_bins + mel_bin] -= mean;
        }
    }

    FeatureMatrix {
        frames: num_frames,
        bins: num_mel_bins,
        data: feats,
    }
}

pub fn normalize_output(samples: &mut [f32]) {
    let peak = samples
        .iter()
        .fold(0.0_f32, |peak, sample| peak.max(sample.abs()))
        .max(EPS);
    let scale = 0.9 / peak;
    for sample in samples.iter_mut() {
        *sample *= scale;
    }
}

fn upper_power_of_two(value: usize) -> usize {
    value.next_power_of_two()
}

fn hz_to_mel(freq: f32) -> f32 {
    1127.0 * (1.0 + freq / 700.0).ln()
}

fn build_mel_filter_bank(sample_rate: usize, frame_length: usize, num_mel_bins: usize) -> Vec<f32> {
    let n_fft = upper_power_of_two(frame_length);
    let num_fft_bins = n_fft / 2;
    let fft_bin_width = sample_rate as f32 / n_fft as f32;
    let mel_low = hz_to_mel(20.0);
    let mel_high = hz_to_mel(sample_rate as f32 / 2.0);
    let mel_delta = (mel_high - mel_low) / (num_mel_bins + 1) as f32;
    let mut filters = vec![0.0; num_mel_bins * num_fft_bins];

    for mel_bin in 0..num_mel_bins {
        let left_mel = mel_low + mel_bin as f32 * mel_delta;
        let center_mel = mel_low + (mel_bin + 1) as f32 * mel_delta;
        let right_mel = mel_low + (mel_bin + 2) as f32 * mel_delta;

        for fft_bin in 0..num_fft_bins {
            let mel = hz_to_mel(fft_bin as f32 * fft_bin_width);
            let weight = if mel <= left_mel || mel >= right_mel {
                0.0
            } else if mel <= center_mel {
                (mel - left_mel) / (center_mel - left_mel)
            } else {
                (right_mel - mel) / (right_mel - center_mel)
            };
            filters[mel_bin * num_fft_bins + fft_bin] = weight;
        }
    }

    filters
}

fn hamming_window(len: usize, periodic: bool) -> Vec<f32> {
    if len == 1 {
        return vec![1.0];
    }

    let denom = if periodic {
        len as f32
    } else {
        (len - 1) as f32
    };
    (0..len)
        .map(|index| 0.54 - 0.46 * (2.0 * PI * index as f32 / denom).cos())
        .collect()
}

fn hann_window(len: usize, periodic: bool) -> Vec<f32> {
    if len == 1 {
        return vec![1.0];
    }

    let denom = if periodic {
        len as f32
    } else {
        (len - 1) as f32
    };
    (0..len)
        .map(|index| 0.5 - 0.5 * (2.0 * PI * index as f32 / denom).cos())
        .collect()
}

/// Reflect-pad `input` into `output`, reusing existing capacity. Mirrors
/// `numpy.pad(..., mode="reflect")` semantics for `pad > 0`.
fn reflect_pad_into(input: &[f32], pad: usize, output: &mut Vec<f32>) {
    output.clear();
    if pad == 0 {
        output.extend_from_slice(input);
        return;
    }
    if input.len() == 1 {
        output.reserve(input.len() + pad * 2);
        for _ in 0..pad {
            output.push(input[0]);
        }
        output.push(input[0]);
        for _ in 0..pad {
            output.push(input[0]);
        }
        return;
    }

    output.reserve(input.len() + pad * 2);
    for index in (-(pad as isize))..(input.len() as isize + pad as isize) {
        output.push(input[reflect_index(index, input.len())]);
    }
}

fn reflect_index(mut index: isize, len: usize) -> usize {
    let len = len as isize;
    while index < 0 || index >= len {
        if index < 0 {
            index = -index;
        } else {
            index = 2 * len - 2 - index;
        }
    }
    index as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn log_mel_shape_matches_reference_layout() {
        let waveform = vec![0.1; 1_600];
        let feats = compute_log_mel_fbank(&waveform, DEFAULT_SAMPLE_RATE, DEFAULT_NUM_MEL_BINS);

        assert_eq!(feats.frames, 8);
        assert_eq!(feats.bins, 80);
        assert_eq!(feats.data.len(), feats.frames * feats.bins);
    }

    #[test]
    fn stft_and_istft_round_trip_waveform() {
        let sample_rate = 16_000.0_f32;
        let waveform: Vec<f32> = (0..4_000)
            .map(|index| (2.0 * PI * 440.0 * index as f32 / sample_rate).sin() * 0.25)
            .collect();

        let mut fwd = StftState::new(512, 128);
        let mut inv = StftState::new(512, 128);
        let mut spec = ComplexSpectrogram {
            freq_bins: 0,
            frames: 0,
            real: Vec::new(),
            imag: Vec::new(),
        };
        fwd.stft_into(&waveform, &mut spec).unwrap();
        let mut reconstructed = Vec::new();
        inv.istft_into(&spec, waveform.len(), &mut reconstructed)
            .unwrap();

        let mae = waveform
            .iter()
            .zip(reconstructed.iter())
            .map(|(lhs, rhs)| (lhs - rhs).abs())
            .sum::<f32>()
            / waveform.len() as f32;

        assert!(mae < 1e-3, "mae={mae}");
    }
}
