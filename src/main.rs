use std::path::PathBuf;

use anyhow::Result;
use clap::{Parser, Subcommand};
use tracing::{Level, info, level_filters::LevelFilter};
use tracing_subscriber::EnvFilter;
mod audio;
mod inference;
mod loader;
mod model;
mod streaming;

#[derive(Debug, Parser)]
#[command(name = "wulfenite")]
#[command(about = "Load a converted WeSep safetensors checkpoint with Burn")]
struct Cli {
    #[arg(long, global = true, value_enum, default_value_t = loader::BackendKind::Cpu)]
    backend: loader::BackendKind,
    #[arg(long, global = true, value_enum, default_value_t = loader::PrecisionKind::Auto)]
    precision: loader::PrecisionKind,
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    Load {
        weight: PathBuf,
    },
    Separate {
        #[arg(long)]
        enrollment: PathBuf,
        #[arg(long)]
        mixture: PathBuf,
        #[arg(long)]
        weight: PathBuf,
        #[arg(long)]
        output: PathBuf,
    },
    Stream {
        #[arg(long)]
        enrollment: PathBuf,
        #[arg(long)]
        weight: PathBuf,
        #[arg(long)]
        input_device: Option<String>,
        #[arg(long)]
        output_device: Option<String>,
        #[arg(long, default_value_t = streaming::DEFAULT_STREAM_CHUNK_SECONDS)]
        chunk_seconds: f32,
        #[arg(long, default_value_t = streaming::DEFAULT_STREAM_MAX_BUFFER_SECONDS)]
        max_buffer_seconds: f32,
        #[arg(long, default_value_t = streaming::DEFAULT_STREAM_OVERLAP_RATIO)]
        overlap_ratio: f32,
    },
}

#[tokio::main(flavor = "multi_thread")]
async fn main() -> Result<()> {
    init_tracing();
    let cli = Cli::parse();
    match cli.command {
        Command::Load { weight } => {
            let report = loader::try_load_checkpoint(loader::TryLoadOptions {
                weight,
                backend: cli.backend,
                precision: cli.precision,
            })?;
            info!(
                "checkpoint loaded successfully from {}",
                report.converted_path.display()
            );
            info!(
                "converted: {}\napplied: {}\nmissing: {}\nunused: {}\nerrors: {}",
                report.converted_path.display(),
                report.applied,
                report.missing,
                report.unused,
                report.errors
            );
        }
        Command::Separate {
            enrollment,
            mixture,
            weight,
            output,
        } => {
            let report = inference::separate_to_file(inference::SeparateOptions {
                enrollment,
                mixture,
                weight,
                output,
                backend: cli.backend,
                precision: cli.precision,
            })?;
            info!(
                "separated waveform written to {}",
                report.output_path.display()
            );
            info!(
                "output: {}\nsample_rate: {}\nmixture_seconds: {:.3}\nelapsed_seconds: {:.3}\nrtf: {:.3}\nfaster_than_real_time: {:.3}",
                report.output_path.display(),
                report.sample_rate,
                report.mixture_seconds,
                report.elapsed_seconds,
                report.rtf,
                report.faster_than_real_time
            );
        }
        Command::Stream {
            enrollment,
            weight,
            input_device,
            output_device,
            chunk_seconds,
            max_buffer_seconds,
            overlap_ratio,
        } => {
            streaming::stream_from_microphone(streaming::StreamOptions {
                enrollment,
                weight,
                input_device,
                output_device,
                backend: cli.backend,
                precision: cli.precision,
                chunk_seconds,
                max_buffer_seconds,
                overlap_ratio,
            })
            .await?;
        }
    }

    Ok(())
}

fn init_tracing() {
    tracing_subscriber::fmt().with_level(true).init();
}
