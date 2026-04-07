use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail, ensure};
use clap::ValueEnum;

use burn::{
    backend::{NdArray, Wgpu},
    store::{ModuleSnapshot, SafetensorsStore},
    tensor::{DType, backend::Backend, f16},
};

use crate::model::wesep::{WeSepBsrnn, WeSepBsrnnConfig};

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
pub enum BackendKind {
    Cpu,
    Gpu,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
pub enum PrecisionKind {
    Auto,
    Fp32,
    Fp16,
}

pub type CpuBackend = NdArray<f32>;
pub type GpuFp32Backend = Wgpu<f32>;
pub type GpuFp16Backend = Wgpu<f16>;

#[derive(Debug, Clone)]
pub struct TryLoadOptions {
    pub weight: PathBuf,
    pub backend: BackendKind,
    pub precision: PrecisionKind,
}

#[derive(Debug, Clone)]
pub struct LoadReport {
    pub converted_path: PathBuf,
    pub applied: usize,
    pub missing: usize,
    pub unused: usize,
    pub errors: usize,
    pub diagnostics: String,
}

pub fn try_load_checkpoint(options: TryLoadOptions) -> Result<LoadReport> {
    match (options.backend, options.precision) {
        (BackendKind::Cpu, PrecisionKind::Auto | PrecisionKind::Fp32) => {
            load_with_backend::<CpuBackend>(&options.weight)
        }
        (BackendKind::Cpu, PrecisionKind::Fp16) => {
            unsupported_precision(options.backend, options.precision)
        }
        (BackendKind::Gpu, PrecisionKind::Auto) => load_with_auto_gpu_backend(&options.weight),
        (BackendKind::Gpu, PrecisionKind::Fp32) => {
            load_with_backend::<GpuFp32Backend>(&options.weight)
        }
        (BackendKind::Gpu, PrecisionKind::Fp16) => load_with_checked_backend::<GpuFp16Backend>(
            &options.weight,
            DType::F16,
            options.backend,
        ),
    }
    .with_context(|| {
        format!(
            "failed to load checkpoint {} with {:?} backend and {:?} precision",
            options.weight.display(),
            options.backend,
            options.precision
        )
    })
}

pub fn load_model<B: Backend>(path: &Path, device: &B::Device) -> Result<WeSepBsrnn<B>> {
    let (model, _) = load_model_with_report(path, device)?;
    Ok(model)
}

fn load_model_with_report<B: Backend>(
    path: &Path,
    device: &B::Device,
) -> Result<(WeSepBsrnn<B>, LoadReport)> {
    let extension = path
        .extension()
        .and_then(|ext| ext.to_str())
        .unwrap_or_default();
    ensure!(
        extension == "safetensors",
        "expected a converted .safetensors checkpoint, got {}",
        path.display()
    );

    let mut model = WeSepBsrnnConfig::default().init::<B>(device);
    let mut store = SafetensorsStore::from_file(path)
        .allow_partial(true)
        .validate(false);
    let result = model
        .load_from(&mut store)
        .with_context(|| format!("failed to apply checkpoint tensors from {}", path.display()))?;

    let report = LoadReport {
        converted_path: path.to_path_buf(),
        applied: result.applied.len(),
        missing: result.missing.len(),
        unused: result.unused.len(),
        errors: result.errors.len(),
        diagnostics: format!("{result}"),
    };

    ensure!(
        report.missing == 0 && report.errors == 0,
        "{}",
        format_load_report(&report)
    );

    Ok((model, report))
}

fn load_with_backend<B: Backend>(path: &Path) -> Result<LoadReport> {
    let device = Default::default();
    load_model_with_report::<B>(path, &device).map(|(_, report)| report)
}

fn load_with_checked_backend<B: Backend>(
    path: &Path,
    dtype: DType,
    backend: BackendKind,
) -> Result<LoadReport> {
    let device = Default::default();
    ensure!(
        B::supports_dtype(&device, dtype),
        "{backend:?} backend does not support {dtype:?} precision on the selected device"
    );
    load_model_with_report::<B>(path, &device).map(|(_, report)| report)
}

fn load_with_auto_gpu_backend(path: &Path) -> Result<LoadReport> {
    let device = Default::default();
    if GpuFp16Backend::supports_dtype(&device, DType::F16) {
        load_model_with_report::<GpuFp16Backend>(path, &device).map(|(_, report)| report)
    } else {
        load_model_with_report::<GpuFp32Backend>(path, &device).map(|(_, report)| report)
    }
}

pub fn unsupported_precision<T>(backend: BackendKind, precision: PrecisionKind) -> Result<T> {
    match (backend, precision) {
        (BackendKind::Cpu, PrecisionKind::Fp16) => bail!(
            "cpu backend uses burn-ndarray, which does not support f16; use --precision fp32 or switch to --backend gpu"
        ),
        _ => bail!("unsupported precision {precision:?} for backend {backend:?}"),
    }
}

fn format_load_report(report: &LoadReport) -> String {
    format!(
        "checkpoint load did not fully match\nconverted: {}\napplied: {}\nmissing: {}\nunused: {}\nerrors: {}\n{}",
        report.converted_path.display(),
        report.applied,
        report.missing,
        report.unused,
        report.errors,
        report.diagnostics
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_non_safetensors_files() {
        let err = try_load_checkpoint(TryLoadOptions {
            weight: PathBuf::from("/tmp/model.pt"),
            backend: BackendKind::Cpu,
            precision: PrecisionKind::Auto,
        })
        .unwrap_err();

        assert!(err.chain().any(|cause| {
            cause
                .to_string()
                .contains("expected a converted .safetensors checkpoint")
        }));
    }

    #[test]
    fn rejects_cpu_fp16_precision() {
        let err = try_load_checkpoint(TryLoadOptions {
            weight: PathBuf::from("/tmp/model.safetensors"),
            backend: BackendKind::Cpu,
            precision: PrecisionKind::Fp16,
        })
        .unwrap_err();

        assert!(err.chain().any(|cause| {
            cause
                .to_string()
                .contains("cpu backend uses burn-ndarray, which does not support f16")
        }));
    }
}
