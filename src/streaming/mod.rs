use std::{f32::consts::PI, io, path::PathBuf, sync::Arc, time::Instant};

use anyhow::{Context, Result, bail, ensure};
use burn::tensor::backend::Backend;
use cpal::{
    FromSample, Sample, SampleFormat, SizedSample,
    traits::{DeviceTrait, HostTrait, StreamTrait},
};
use ringbuf::{HeapRb, traits::*};
use tokio::{sync::Notify, task};
use tracing::{debug, info, warn};

use crate::{
    audio::resample_mono_borrowed,
    inference::{self, SeparatorSession},
    loader::{self, BackendKind, CpuBackend, GpuFp16Backend, GpuFp32Backend, PrecisionKind},
    model::wesep::WeSepBsrnnConfig,
};

pub const DEFAULT_STREAM_CHUNK_SECONDS: f32 = 1.2;
pub const DEFAULT_STREAM_MAX_BUFFER_SECONDS: f32 = 5.0;
pub const DEFAULT_STREAM_OVERLAP_RATIO: f32 = 0.75;

const MIN_STREAM_OVERLAP_RATIO: f32 = 0.70;
const MAX_STREAM_OVERLAP_RATIO: f32 = 0.85;
const OUTPUT_WEIGHT_EPSILON: f32 = 1e-6;

#[derive(Debug, Clone)]
pub struct StreamOptions {
    pub enrollment: PathBuf,
    pub weight: PathBuf,
    pub input_device: Option<String>,
    pub output_device: Option<String>,
    pub backend: BackendKind,
    pub precision: PrecisionKind,
    pub chunk_seconds: f32,
    pub max_buffer_seconds: f32,
    pub overlap_ratio: f32,
}

pub async fn stream_from_microphone(options: StreamOptions) -> Result<()> {
    ensure!(
        options.chunk_seconds > 0.0,
        "chunk_seconds must be greater than zero"
    );
    ensure!(
        options.max_buffer_seconds >= options.chunk_seconds,
        "max_buffer_seconds must be greater than or equal to chunk_seconds"
    );
    ensure!(
        (MIN_STREAM_OVERLAP_RATIO..=MAX_STREAM_OVERLAP_RATIO).contains(&options.overlap_ratio),
        "overlap_ratio must be between {:.2} and {:.2}",
        MIN_STREAM_OVERLAP_RATIO,
        MAX_STREAM_OVERLAP_RATIO
    );

    match (options.backend, options.precision) {
        (BackendKind::Cpu, PrecisionKind::Auto | PrecisionKind::Fp32) => {
            stream_with_backend::<CpuBackend>(&options).await
        }
        (BackendKind::Cpu, PrecisionKind::Fp16) => {
            loader::unsupported_precision(options.backend, options.precision)
        }
        (BackendKind::Gpu, PrecisionKind::Auto) => {
            if inference::backend_supports_f16::<GpuFp16Backend>() {
                stream_with_backend::<GpuFp16Backend>(&options).await
            } else {
                stream_with_backend::<GpuFp32Backend>(&options).await
            }
        }
        (BackendKind::Gpu, PrecisionKind::Fp32) => {
            stream_with_backend::<GpuFp32Backend>(&options).await
        }
        (BackendKind::Gpu, PrecisionKind::Fp16) => {
            inference::ensure_backend_dtype_available::<GpuFp16Backend>(options.backend)?;
            stream_with_backend::<GpuFp16Backend>(&options).await
        }
    }
    .with_context(|| {
        format!(
            "failed to start microphone stream with enrollment {} using {:?} backend and {:?} precision",
            options.enrollment.display(),
            options.backend,
            options.precision
        )
    })
}

async fn stream_with_backend<B: Backend>(options: &StreamOptions) -> Result<()> {
    let config = WeSepBsrnnConfig::default();
    let enrollment = inference::load_and_resample_wav(&options.enrollment, config.sample_rate)?;
    let device = Default::default();
    let separator = SeparatorSession::<B>::load(&options.weight, enrollment, device)?;

    let host = cpal::default_host();
    let input_device = select_input_device(&host, options.input_device.as_deref())?;
    let output_device = select_output_device(&host, options.output_device.as_deref())?;
    let input_supported = input_device
        .default_input_config()
        .context("failed to query default input config")?;
    let output_supported = output_device
        .default_output_config()
        .context("failed to query default output config")?;

    let input_config: cpal::StreamConfig = input_supported.config();
    let output_config: cpal::StreamConfig = output_supported.config();
    let input_channels = input_config.channels as usize;
    let output_channels = output_config.channels as usize;
    let input_rate = input_config.sample_rate as usize;
    let output_rate = output_config.sample_rate as usize;
    let chunk_input_samples = ((options.chunk_seconds * input_rate as f32).round() as usize).max(1);
    let hop_input_samples =
        (((1.0 - options.overlap_ratio) * chunk_input_samples as f32).round() as usize).max(1);
    let max_input_samples = ((options.max_buffer_seconds * input_rate as f32).round() as usize)
        .max(chunk_input_samples);
    let chunk_model_samples =
        predict_resampled_len(chunk_input_samples, input_rate, config.sample_rate);
    let chunk_output_samples =
        predict_resampled_len(chunk_model_samples, config.sample_rate, output_rate);
    let hop_output_samples = predict_resampled_len(hop_input_samples, input_rate, output_rate);
    let max_output_samples = predict_resampled_len(max_input_samples, input_rate, output_rate)
        .max(chunk_output_samples * 2);

    info!(
        "streaming mic -> speaker with windowed inference; chunk {:.2}s, hop {:.2}s, overlap {:.0}%, expect at least {:.2}s algorithmic latency",
        options.chunk_seconds,
        hop_input_samples as f32 / input_rate as f32,
        options.overlap_ratio * 100.0,
        options.chunk_seconds
    );
    info!(
        "input device: {} | {} Hz | {} ch | {:?}",
        describe_device(&input_device),
        input_rate,
        input_channels,
        input_supported.sample_format()
    );
    info!(
        "output device: {} | {} Hz | {} ch | {:?}",
        describe_device(&output_device),
        output_rate,
        output_channels,
        output_supported.sample_format()
    );

    let input_ready = Arc::new(Notify::new());
    let shutdown = Arc::new(Notify::new());
    let input_rb = HeapRb::<f32>::new(max_input_samples);
    let output_rb = HeapRb::<f32>::new(max_output_samples);
    let (input_prod, mut input_cons) = input_rb.split();
    let (mut output_prod, output_cons) = output_rb.split();

    let worker_input_ready = Arc::clone(&input_ready);
    let worker_shutdown = Arc::clone(&shutdown);
    let model_sample_rate = config.sample_rate;
    let worker: task::JoinHandle<Result<()>> = tokio::spawn(async move {
        let overlap_window = build_overlap_window(chunk_output_samples);
        // Fixed-size sliding window. Filled from the head until the first
        // chunk is ready, then rolled left in-place via `copy_within` for
        // every subsequent hop. Avoids `VecDeque` rotation cost and the
        // per-chunk `iter().collect::<Vec<_>>()` copy that the previous
        // implementation paid.
        let mut input_window = vec![0.0_f32; chunk_input_samples];
        let mut window_filled: usize = 0;
        // Output overlap-add ring buffers. Length is fixed; `oa_head` tracks
        // the rotating start of the window inside the flat buffers.
        let mut output_mix = vec![0.0_f32; chunk_output_samples];
        let mut output_weight = vec![0.0_f32; chunk_output_samples];
        let mut oa_head: usize = 0;
        let mut ready_buf: Vec<f32> = Vec::with_capacity(hop_output_samples);
        let mut hop_buffer = vec![0.0; hop_input_samples];

        'worker: loop {
            while input_cons.occupied_len() < hop_input_samples {
                tokio::select! {
                    _ = worker_input_ready.notified() => {}
                    _ = worker_shutdown.notified() => break 'worker Ok(()),
                }
            }

            let popped = input_cons.pop_slice(&mut hop_buffer);
            if popped == 0 {
                continue;
            }

            // Append `popped` samples to the sliding window. While the window
            // is still warming up we fill from the tail; once full we shift
            // the existing samples left by `popped` and write the new ones at
            // the end. `popped` is bounded by `hop_input_samples`, which is
            // <= `chunk_input_samples` by construction, so a single shift
            // suffices.
            if window_filled < chunk_input_samples {
                let take = (chunk_input_samples - window_filled).min(popped);
                input_window[window_filled..window_filled + take]
                    .copy_from_slice(&hop_buffer[..take]);
                window_filled += take;
                if take < popped {
                    let rem = popped - take;
                    input_window.copy_within(rem.., 0);
                    input_window[chunk_input_samples - rem..]
                        .copy_from_slice(&hop_buffer[take..popped]);
                }
            } else {
                input_window.copy_within(popped.., 0);
                input_window[chunk_input_samples - popped..]
                    .copy_from_slice(&hop_buffer[..popped]);
            }

            if window_filled < chunk_input_samples {
                continue;
            }

            let t0 = Instant::now();
            let chunk_16k =
                resample_mono_borrowed(&input_window, input_rate, model_sample_rate);
            let t1 = Instant::now();
            let result = task::block_in_place(|| separator.process_waveform(&chunk_16k))?;
            let t2 = Instant::now();
            let output_chunk =
                resample_mono_borrowed(&result.waveform, model_sample_rate, output_rate);
            let t3 = Instant::now();
            overlap_add_chunk(
                &mut output_mix,
                &mut output_weight,
                &mut oa_head,
                &output_chunk,
                &overlap_window,
                hop_output_samples,
                &mut ready_buf,
            );
            let t4 = Instant::now();
            debug!(
                resample_in_us = (t1 - t0).as_micros() as u64,
                process_us = (t2 - t1).as_micros() as u64,
                resample_out_us = (t3 - t2).as_micros() as u64,
                overlap_add_us = (t4 - t3).as_micros() as u64,
                "stream chunk timing"
            );
            let written = output_prod.push_slice(&ready_buf);
            if written < ready_buf.len() {
                warn!(
                    "output ring buffer overrun: dropped {} samples",
                    ready_buf.len() - written
                );
            }
        }
    });

    let input_stream = build_input_stream(
        &input_device,
        &input_config,
        input_supported.sample_format(),
        input_channels,
        input_prod,
        Arc::clone(&input_ready),
    )?;
    let output_stream = build_output_stream(
        &output_device,
        &output_config,
        output_supported.sample_format(),
        output_channels,
        output_cons,
    )?;

    input_stream
        .play()
        .context("failed to start input stream")?;
    output_stream
        .play()
        .context("failed to start output stream")?;

    println!("Live TSE stream started. Press Enter to stop.");
    task::spawn_blocking(|| -> Result<()> {
        let mut line = String::new();
        io::stdin()
            .read_line(&mut line)
            .context("failed while waiting for stop input")?;
        Ok(())
    })
    .await
    .map_err(|_| anyhow::anyhow!("stdin wait task panicked"))??;

    shutdown.notify_waiters();
    input_ready.notify_waiters();
    drop(input_stream);
    drop(output_stream);

    worker
        .await
        .map_err(|_| anyhow::anyhow!("stream worker task panicked"))??;

    Ok(())
}

fn build_input_stream<P>(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    sample_format: SampleFormat,
    channels: usize,
    input_prod: P,
    input_ready: Arc<Notify>,
) -> Result<cpal::Stream>
where
    P: Producer<Item = f32> + Send + 'static,
{
    let err_fn = |err| eprintln!("input stream error: {err}");

    let stream = match sample_format {
        SampleFormat::F32 => {
            let mut input_prod = input_prod;
            let input_ready = Arc::clone(&input_ready);
            let mut mono_buf: Vec<f32> = Vec::with_capacity(2048);
            device.build_input_stream(
                config,
                move |data: &[f32], _| {
                    push_input_samples(
                        data,
                        channels,
                        &mut mono_buf,
                        &mut input_prod,
                        &input_ready,
                    )
                },
                err_fn,
                None,
            )
        }
        SampleFormat::I16 => {
            let mut input_prod = input_prod;
            let input_ready = Arc::clone(&input_ready);
            let mut mono_buf: Vec<f32> = Vec::with_capacity(2048);
            device.build_input_stream(
                config,
                move |data: &[i16], _| {
                    push_input_samples(
                        data,
                        channels,
                        &mut mono_buf,
                        &mut input_prod,
                        &input_ready,
                    )
                },
                err_fn,
                None,
            )
        }
        SampleFormat::U16 => {
            let mut input_prod = input_prod;
            let input_ready = Arc::clone(&input_ready);
            let mut mono_buf: Vec<f32> = Vec::with_capacity(2048);
            device.build_input_stream(
                config,
                move |data: &[u16], _| {
                    push_input_samples(
                        data,
                        channels,
                        &mut mono_buf,
                        &mut input_prod,
                        &input_ready,
                    )
                },
                err_fn,
                None,
            )
        }
        SampleFormat::I32 => {
            let mut input_prod = input_prod;
            let input_ready = Arc::clone(&input_ready);
            let mut mono_buf: Vec<f32> = Vec::with_capacity(2048);
            device.build_input_stream(
                config,
                move |data: &[i32], _| {
                    push_input_samples(
                        data,
                        channels,
                        &mut mono_buf,
                        &mut input_prod,
                        &input_ready,
                    )
                },
                err_fn,
                None,
            )
        }
        SampleFormat::U32 => {
            let mut input_prod = input_prod;
            let input_ready = Arc::clone(&input_ready);
            let mut mono_buf: Vec<f32> = Vec::with_capacity(2048);
            device.build_input_stream(
                config,
                move |data: &[u32], _| {
                    push_input_samples(
                        data,
                        channels,
                        &mut mono_buf,
                        &mut input_prod,
                        &input_ready,
                    )
                },
                err_fn,
                None,
            )
        }
        SampleFormat::I8 => {
            let mut input_prod = input_prod;
            let input_ready = Arc::clone(&input_ready);
            let mut mono_buf: Vec<f32> = Vec::with_capacity(2048);
            device.build_input_stream(
                config,
                move |data: &[i8], _| {
                    push_input_samples(
                        data,
                        channels,
                        &mut mono_buf,
                        &mut input_prod,
                        &input_ready,
                    )
                },
                err_fn,
                None,
            )
        }
        SampleFormat::U8 => {
            let mut input_prod = input_prod;
            let input_ready = Arc::clone(&input_ready);
            let mut mono_buf: Vec<f32> = Vec::with_capacity(2048);
            device.build_input_stream(
                config,
                move |data: &[u8], _| {
                    push_input_samples(
                        data,
                        channels,
                        &mut mono_buf,
                        &mut input_prod,
                        &input_ready,
                    )
                },
                err_fn,
                None,
            )
        }
        SampleFormat::F64 => {
            let mut input_prod = input_prod;
            let input_ready = Arc::clone(&input_ready);
            let mut mono_buf: Vec<f32> = Vec::with_capacity(2048);
            device.build_input_stream(
                config,
                move |data: &[f64], _| {
                    push_input_samples(
                        data,
                        channels,
                        &mut mono_buf,
                        &mut input_prod,
                        &input_ready,
                    )
                },
                err_fn,
                None,
            )
        }
        unsupported => {
            return Err(anyhow::anyhow!(
                "unsupported input sample format {unsupported:?}"
            ));
        }
    }
    .context("failed to build input stream")?;

    Ok(stream)
}

fn build_output_stream<C>(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    sample_format: SampleFormat,
    channels: usize,
    output_cons: C,
) -> Result<cpal::Stream>
where
    C: Consumer<Item = f32> + Send + 'static,
{
    let err_fn = |err| eprintln!("output stream error: {err}");

    let stream = match sample_format {
        SampleFormat::F32 => {
            let mut output_cons = output_cons;
            device.build_output_stream(
                config,
                move |data: &mut [f32], _| write_output_samples(data, channels, &mut output_cons),
                err_fn,
                None,
            )
        }
        SampleFormat::I16 => {
            let mut output_cons = output_cons;
            device.build_output_stream(
                config,
                move |data: &mut [i16], _| write_output_samples(data, channels, &mut output_cons),
                err_fn,
                None,
            )
        }
        SampleFormat::U16 => {
            let mut output_cons = output_cons;
            device.build_output_stream(
                config,
                move |data: &mut [u16], _| write_output_samples(data, channels, &mut output_cons),
                err_fn,
                None,
            )
        }
        SampleFormat::I32 => {
            let mut output_cons = output_cons;
            device.build_output_stream(
                config,
                move |data: &mut [i32], _| write_output_samples(data, channels, &mut output_cons),
                err_fn,
                None,
            )
        }
        SampleFormat::U32 => {
            let mut output_cons = output_cons;
            device.build_output_stream(
                config,
                move |data: &mut [u32], _| write_output_samples(data, channels, &mut output_cons),
                err_fn,
                None,
            )
        }
        SampleFormat::I8 => {
            let mut output_cons = output_cons;
            device.build_output_stream(
                config,
                move |data: &mut [i8], _| write_output_samples(data, channels, &mut output_cons),
                err_fn,
                None,
            )
        }
        SampleFormat::U8 => {
            let mut output_cons = output_cons;
            device.build_output_stream(
                config,
                move |data: &mut [u8], _| write_output_samples(data, channels, &mut output_cons),
                err_fn,
                None,
            )
        }
        SampleFormat::F64 => {
            let mut output_cons = output_cons;
            device.build_output_stream(
                config,
                move |data: &mut [f64], _| write_output_samples(data, channels, &mut output_cons),
                err_fn,
                None,
            )
        }
        unsupported => {
            return Err(anyhow::anyhow!(
                "unsupported output sample format {unsupported:?}"
            ));
        }
    }
    .context("failed to build output stream")?;

    Ok(stream)
}

/// Downmix and forward `input` into the ringbuf, reusing the caller-supplied
/// `mono` scratch buffer. The buffer lives in the cpal stream callback's
/// closure (one per stream) so this function performs no allocation on the
/// real-time audio thread after the first warm-up call — which is essential
/// to avoid jitter and xruns on low-latency devices.
fn push_input_samples<T, P>(
    input: &[T],
    channels: usize,
    mono: &mut Vec<f32>,
    input_prod: &mut P,
    input_ready: &Arc<Notify>,
) where
    T: Sample,
    f32: FromSample<T>,
    P: Producer<Item = f32>,
{
    let chans = channels.max(1);
    mono.clear();
    mono.reserve(input.len() / chans);
    for frame in input.chunks(chans) {
        let sum = frame
            .iter()
            .fold(0.0_f32, |sum, sample| sum + f32::from_sample(*sample));
        mono.push(sum / frame.len() as f32);
    }

    let written = input_prod.push_slice(mono);
    if written > 0 {
        input_ready.notify_one();
    }
    if written < mono.len() {
        eprintln!(
            "input ring buffer overrun: dropped {} microphone samples",
            mono.len() - written
        );
    }
}

fn write_output_samples<T, C>(output: &mut [T], channels: usize, output_cons: &mut C)
where
    T: SizedSample + FromSample<f32>,
    C: Consumer<Item = f32>,
{
    for frame in output.chunks_mut(channels.max(1)) {
        let sample = output_cons.try_pop().unwrap_or(0.0).clamp(-1.0, 1.0);
        let value = T::from_sample(sample);
        for output_sample in frame.iter_mut() {
            *output_sample = value;
        }
    }
}

fn predict_resampled_len(input_len: usize, orig_rate: usize, target_rate: usize) -> usize {
    if input_len == 0 {
        return 0;
    }
    if orig_rate == target_rate {
        return input_len;
    }

    (((input_len as f64) * target_rate as f64 / orig_rate as f64).round() as usize).max(1)
}

fn select_input_device(host: &cpal::Host, requested: Option<&str>) -> Result<cpal::Device> {
    match requested {
        Some(name) => find_named_device(
            host.input_devices()
                .context("failed to enumerate input devices")?,
            name,
            "input",
        ),
        None => host
            .default_input_device()
            .context("no default input device available"),
    }
}

fn select_output_device(host: &cpal::Host, requested: Option<&str>) -> Result<cpal::Device> {
    match requested {
        Some(name) => find_named_device(
            host.output_devices()
                .context("failed to enumerate output devices")?,
            name,
            "output",
        ),
        None => host
            .default_output_device()
            .context("no default output device available"),
    }
}

fn find_named_device<I>(devices: I, requested: &str, kind: &str) -> Result<cpal::Device>
where
    I: IntoIterator<Item = cpal::Device>,
{
    let requested = requested.trim();
    ensure!(!requested.is_empty(), "{kind}_device must not be empty");

    let requested_folded = requested.to_ascii_lowercase();
    let mut exact_match = None;
    let mut partial_matches = Vec::new();
    let mut available_names = Vec::new();

    for device in devices {
        let name = describe_device(&device);
        let folded = name.to_ascii_lowercase();
        available_names.push(name.clone());

        if folded == requested_folded {
            exact_match = Some(device);
            break;
        }

        if folded.contains(&requested_folded) {
            partial_matches.push((name, device));
        }
    }

    if let Some(device) = exact_match {
        return Ok(device);
    }

    match partial_matches.len() {
        1 => Ok(partial_matches
            .pop()
            .expect("partial_matches length checked")
            .1),
        0 => {
            log_available_devices(kind, &available_names);
            bail!(
                "no {kind} device matched '{requested}'. available {kind} devices: {}",
                format_device_list(&available_names)
            )
        }
        _ => {
            let matches = partial_matches
                .into_iter()
                .map(|(name, _)| name)
                .collect::<Vec<_>>();
            log_available_devices(kind, &available_names);
            bail!(
                "multiple {kind} devices matched '{requested}': {}",
                format_device_list(&matches)
            )
        }
    }
}

fn format_device_list(names: &[String]) -> String {
    if names.is_empty() {
        "<none>".to_string()
    } else {
        names.join(", ")
    }
}

fn log_available_devices(kind: &str, names: &[String]) {
    if names.is_empty() {
        warn!("no {kind} devices are currently available");
    } else {
        warn!("available {kind} devices: {}", format_device_list(names));
    }
}

fn build_overlap_window(len: usize) -> Vec<f32> {
    match len {
        0 => Vec::new(),
        1 => vec![1.0],
        _ => (0..len)
            .map(|index| {
                let phase = 2.0 * PI * (index as f32 + 0.5) / len as f32;
                0.5 - 0.5 * phase.cos()
            })
            .collect(),
    }
}

/// Accumulate `chunk * window` into the rotating overlap-add buffers and
/// drain `hop_samples` finished samples into `ready`.
///
/// `output_mix` and `output_weight` are flat ring buffers of fixed length
/// (`chunk_output_samples`); `head` is the rotating start index. After this
/// call, `head` advances by `hop_samples` (modulo the buffer length) and the
/// drained slots are zeroed in place so the next chunk can accumulate into
/// them. The previous implementation used `VecDeque::get_mut` and
/// `pop_front + push_back` per sample, which paid head-offset bookkeeping
/// twice per element; this version is straight-line over `&mut [f32]`.
fn overlap_add_chunk(
    output_mix: &mut [f32],
    output_weight: &mut [f32],
    head: &mut usize,
    chunk: &[f32],
    window: &[f32],
    hop_samples: usize,
    ready: &mut Vec<f32>,
) {
    debug_assert_eq!(chunk.len(), window.len());
    debug_assert_eq!(output_mix.len(), output_weight.len());
    let len = output_mix.len();
    if len == 0 {
        ready.clear();
        return;
    }

    // Accumulate chunk * window into the ring starting at *head.
    // Hot loop — keep it scalar; SIMD is a follow-up if profiling demands.
    let chunk_len = chunk.len().min(len);
    for i in 0..chunk_len {
        let idx = (*head + i) % len;
        let w = window[i];
        output_mix[idx] += chunk[i] * w;
        output_weight[idx] += w;
    }

    // Drain hop_samples finished samples from the head and zero them so the
    // ring slots are ready for the next chunk's accumulation.
    let drain = hop_samples.min(len);
    ready.clear();
    ready.reserve(drain);
    for _ in 0..drain {
        let i = *head;
        let w = output_weight[i];
        ready.push(if w > OUTPUT_WEIGHT_EPSILON {
            output_mix[i] / w
        } else {
            0.0
        });
        output_mix[i] = 0.0;
        output_weight[i] = 0.0;
        *head = (*head + 1) % len;
    }
}

fn describe_device(device: &cpal::Device) -> String {
    device
        .description()
        .map(|description| description.name().to_string())
        .unwrap_or_else(|_| "<unknown>".to_string())
}

impl Default for StreamOptions {
    fn default() -> Self {
        Self {
            enrollment: PathBuf::new(),
            weight: PathBuf::new(),
            input_device: None,
            output_device: None,
            backend: BackendKind::Cpu,
            precision: PrecisionKind::Auto,
            chunk_seconds: DEFAULT_STREAM_CHUNK_SECONDS,
            max_buffer_seconds: DEFAULT_STREAM_MAX_BUFFER_SECONDS,
            overlap_ratio: DEFAULT_STREAM_OVERLAP_RATIO,
        }
    }
}
