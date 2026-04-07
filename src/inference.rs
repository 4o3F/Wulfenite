use std::{
    cell::RefCell,
    mem,
    path::{Path, PathBuf},
    time::Instant,
};

use anyhow::{Context, Result, ensure};
use burn::tensor::{Tensor, TensorData, backend::Backend};

use crate::{
    audio::{
        ComplexSpectrogram, DEFAULT_NUM_MEL_BINS, DEFAULT_SAMPLE_RATE, StftState,
        compute_log_mel_fbank, normalize_output, read_wav_mono, resample_mono, write_wav_mono,
    },
    loader::{self, BackendKind, CpuBackend, GpuFp16Backend, GpuFp32Backend, PrecisionKind},
    model::wesep::{WeSepBsrnn, WeSepBsrnnConfig},
};

#[derive(Debug, Clone)]
pub struct SeparateOptions {
    pub enrollment: PathBuf,
    pub mixture: PathBuf,
    pub weight: PathBuf,
    pub output: PathBuf,
    pub backend: BackendKind,
    pub precision: PrecisionKind,
}

#[derive(Debug, Clone)]
pub struct SeparationReport {
    pub output_path: PathBuf,
    pub sample_rate: usize,
    pub mixture_seconds: f64,
    pub elapsed_seconds: f64,
    pub rtf: f64,
    pub faster_than_real_time: f64,
}

#[derive(Debug, Clone)]
pub(crate) struct ProcessReport {
    pub(crate) waveform: Vec<f32>,
    pub(crate) elapsed_seconds: f64,
}

pub(crate) struct SeparatorSession<B: Backend> {
    device: B::Device,
    model: WeSepBsrnn<B>,
    enrollment_feat: Tensor<B, 3>,
    // Reused across `process_waveform` calls so the streaming worker does not
    // pay FFT planner / scratch allocation cost on every chunk. Wrapped in
    // `RefCell` for interior mutability — `process_waveform` takes `&self` and
    // the session is never shared across threads concurrently (the streaming
    // worker uses `block_in_place`, the offline path is sync).
    stft_fwd: RefCell<StftState>,
    stft_inv: RefCell<StftState>,
    spec_buf: RefCell<ComplexSpectrogram>,
    out_buf: RefCell<Vec<f32>>,
}

pub fn separate_to_file(options: SeparateOptions) -> Result<SeparationReport> {
    match (options.backend, options.precision) {
        (BackendKind::Cpu, PrecisionKind::Auto | PrecisionKind::Fp32) => {
            separate_with_backend::<CpuBackend>(&options)
        }
        (BackendKind::Cpu, PrecisionKind::Fp16) => {
            loader::unsupported_precision(options.backend, options.precision)
        }
        (BackendKind::Gpu, PrecisionKind::Auto) => separate_with_auto_gpu_backend(&options),
        (BackendKind::Gpu, PrecisionKind::Fp32) => {
            separate_with_backend::<GpuFp32Backend>(&options)
        }
        (BackendKind::Gpu, PrecisionKind::Fp16) => {
            separate_with_checked_backend::<GpuFp16Backend>(&options)
        }
    }
    .with_context(|| {
        format!(
            "failed to separate {} with enrollment {} using {:?} backend and {:?} precision",
            options.mixture.display(),
            options.enrollment.display(),
            options.backend,
            options.precision
        )
    })
}

fn separate_with_checked_backend<B: Backend>(
    options: &SeparateOptions,
) -> Result<SeparationReport> {
    ensure_backend_dtype_available::<B>(options.backend)?;
    separate_with_backend::<B>(options)
}

fn separate_with_auto_gpu_backend(options: &SeparateOptions) -> Result<SeparationReport> {
    if backend_supports_f16::<GpuFp16Backend>() {
        separate_with_backend::<GpuFp16Backend>(options)
    } else {
        separate_with_backend::<GpuFp32Backend>(options)
    }
}

fn separate_with_backend<B: Backend>(options: &SeparateOptions) -> Result<SeparationReport> {
    let config = WeSepBsrnnConfig::default();
    let mixture = load_and_resample_wav(&options.mixture, config.sample_rate)?;
    let enrollment = load_and_resample_wav(&options.enrollment, config.sample_rate)?;
    let device = Default::default();
    let separator = SeparatorSession::<B>::load(&options.weight, enrollment, device)?;

    let mut result = separator.process_waveform(&mixture)?;
    normalize_output(&mut result.waveform);
    write_wav_mono(&options.output, &result.waveform, DEFAULT_SAMPLE_RATE)?;

    let mixture_seconds = mixture.len() as f64 / DEFAULT_SAMPLE_RATE as f64;
    Ok(SeparationReport {
        output_path: options.output.clone(),
        sample_rate: DEFAULT_SAMPLE_RATE,
        mixture_seconds,
        elapsed_seconds: result.elapsed_seconds,
        rtf: result.elapsed_seconds / mixture_seconds,
        faster_than_real_time: mixture_seconds / result.elapsed_seconds,
    })
}

pub(crate) fn ensure_backend_dtype_available<B: Backend>(backend: BackendKind) -> Result<()> {
    ensure!(
        backend_supports_f16::<B>(),
        "{backend:?} backend does not support f16 precision on the selected device"
    );
    Ok(())
}

pub(crate) fn backend_supports_f16<B: Backend>() -> bool {
    let device = Default::default();
    B::supports_dtype(&device, burn::tensor::DType::F16)
}

impl<B: Backend> SeparatorSession<B> {
    pub(crate) fn load(weight: &Path, enrollment: Vec<f32>, device: B::Device) -> Result<Self> {
        let config = WeSepBsrnnConfig::default();
        let model = loader::load_model::<B>(weight, &device)?;
        let enrollment_feat =
            compute_log_mel_fbank(&enrollment, config.sample_rate, DEFAULT_NUM_MEL_BINS);
        let enrollment_feat = Tensor::<B, 3>::from_data(
            TensorData::new(
                enrollment_feat.data,
                [1, enrollment_feat.frames, enrollment_feat.bins],
            ),
            &device,
        );

        let stft_fwd = StftState::new(config.win, config.stride);
        let stft_inv = StftState::new(config.win, config.stride);

        Ok(Self {
            device,
            model,
            enrollment_feat,
            stft_fwd: RefCell::new(stft_fwd),
            stft_inv: RefCell::new(stft_inv),
            spec_buf: RefCell::new(ComplexSpectrogram {
                freq_bins: 0,
                frames: 0,
                real: Vec::new(),
                imag: Vec::new(),
            }),
            out_buf: RefCell::new(Vec::new()),
        })
    }

    pub(crate) fn process_waveform(&self, mixture: &[f32]) -> Result<ProcessReport> {
        ensure!(!mixture.is_empty(), "mixture waveform must not be empty");

        // Forward STFT into the cached spectrogram buffer (no fresh FFT
        // planner rebuild — those live in `stft_fwd`).
        let (freq_bins, frames, real_vec, imag_vec) = {
            let mut spec = self.spec_buf.borrow_mut();
            self.stft_fwd.borrow_mut().stft_into(mixture, &mut spec)?;
            // Move the freshly-filled vecs out of the cached buffer into the
            // tensors. Burn copies host data to device internally, so the
            // previous `.clone()` was a redundant second host copy. After
            // `mem::take`, `spec.real`/`spec.imag` are empty `Vec`s — the
            // next call's `clear() + resize()` reuses the prior allocation.
            (
                spec.freq_bins,
                spec.frames,
                mem::take(&mut spec.real),
                mem::take(&mut spec.imag),
            )
        };

        let mixture_real = Tensor::<B, 3>::from_data(
            TensorData::new(real_vec, [1, freq_bins, frames]),
            &self.device,
        );
        let mixture_imag = Tensor::<B, 3>::from_data(
            TensorData::new(imag_vec, [1, freq_bins, frames]),
            &self.device,
        );

        let start = Instant::now();
        let estimated =
            self.model
                .forward(mixture_real, mixture_imag, self.enrollment_feat.clone());
        let elapsed_seconds = start.elapsed().as_secs_f64();
        let output_real = estimated
            .real
            .into_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .context("failed to read estimated real spectrogram from backend")?;
        let output_imag = estimated
            .imag
            .into_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .context("failed to read estimated imaginary spectrogram from backend")?;

        let estimated_spec = ComplexSpectrogram {
            freq_bins,
            frames,
            real: output_real,
            imag: output_imag,
        };

        let mut out = self.out_buf.borrow_mut();
        self.stft_inv
            .borrow_mut()
            .istft_into(&estimated_spec, mixture.len(), &mut out)?;

        Ok(ProcessReport {
            waveform: out.clone(),
            elapsed_seconds,
        })
    }
}

pub(crate) fn load_and_resample_wav(path: &Path, target_sample_rate: usize) -> Result<Vec<f32>> {
    let audio = read_wav_mono(path)?;
    Ok(resample_mono(
        &audio.samples,
        audio.sample_rate,
        target_sample_rate,
    ))
}
