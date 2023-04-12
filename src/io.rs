pub use anyhow::{anyhow, Result};

pub use ndarray::prelude::*;
pub use std::path::Path;

use ffmpeg_sidecar::command::{ffmpeg_is_installed, FfmpegCommand};
use ffmpeg_sidecar::paths::sidecar_dir;
use indicatif::{ProgressBar, ProgressStyle};

pub fn ensure_ffmpeg(verbose: bool) {
    if !ffmpeg_is_installed() {
        if verbose {
            println!(
                "No ffmpeg installation found, downloading one to {}...",
                &sidecar_dir().unwrap().display()
            );
        }
        ffmpeg_sidecar::download::auto_download().unwrap();
    }
}

pub fn make_video(
    pattern: &str,
    outfile: &str,
    fps: u64,
    num_frames: u64,
    pbar_style: Option<ProgressStyle>,
) {
    let pbar = if let Some(style) = pbar_style {
        ProgressBar::new(num_frames).with_style(style)
    } else {
        ProgressBar::hidden()
    };

    let cmd = format!(
        "-framerate {fps} -f image2 -i {pattern} -y -vcodec libx264 -crf 22 -pix_fmt yuv420p {outfile}"
    );

    let mut ffmpeg_runner = FfmpegCommand::new().args(cmd.split(' ')).spawn().unwrap();
    ffmpeg_runner
        .iter()
        .unwrap()
        .filter_progress()
        .for_each(|progress| pbar.set_position(progress.frame as u64));
    pbar.finish_and_clear();
}
