// Wulfenite — engineering / deployment crate.
//
// This is a clean slate after the TSE branch was archived on `v2`.
// The new model architecture is being designed; this binary intentionally
// has no model code yet. Reusable model-agnostic infrastructure lives in
// the `audio` module (STFT, resampling, WAV I/O, FFT planner caching).

use anyhow::Result;
use clap::{Parser, Subcommand};
use tracing::info;

// audio is a model-agnostic utility library (STFT, resample, WAV I/O,
// FFT planner cache). Will be wired in by future architecture work.
mod audio;

#[derive(Debug, Parser)]
#[command(name = "wulfenite")]
#[command(about = "Wulfenite real-time speech enhancement runtime scaffold")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Print build info; placeholder until the new architecture lands.
    Version,
}

fn main() -> Result<()> {
    init_tracing();
    let cli = Cli::parse();
    match cli.command {
        Command::Version => {
            info!(
                "wulfenite {} — engineering crate, awaiting new model architecture",
                env!("CARGO_PKG_VERSION")
            );
        }
    }
    Ok(())
}

fn init_tracing() {
    tracing_subscriber::fmt().with_level(true).init();
}
