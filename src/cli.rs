pub use clap::Parser;

/// Perform Lucas-Kanade homography estimation between two images and animate optimization process.
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub struct Args {
    /// Path of images to register, expects two.
    #[arg(short, long, num_args(2))]
    pub input: Vec<String>,

    /// Path of output video, defaults to "out.mp4".
    #[arg(short, long)]
    pub output: Option<String>,

    /// Downscale images before optimaization, defaults to 1.0 (no downscaling).
    #[arg(long, default_value_t=1.0)]
    pub downscale: f32,

    /// Number of LK iterations to use, defaults to 250.
    #[arg(long, default_value_t=250)]
    pub iterations: i32,

    /// Only use every `viz_step` frame for vizualization, defaults to 10.
    #[arg(long, default_value_t=10)]
    pub viz_step: usize,

    /// Controls framerate of video preview, defaults to 24.
    #[arg(long, default_value_t=24)]
    pub viz_fps: u64,
}
