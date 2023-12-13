pub use clap::Parser;
use clap::{Args, Subcommand, ValueEnum};

#[derive(Parser, Debug, Clone)]
#[command(author, version, about, long_about = None)]
/// Register two or more images and animate optimization process.
pub struct Cli {
    #[command(subcommand)]
    pub command: Option<Commands>,

    /// Path of images to register, two are expected when matching images
    #[arg(short, long, num_args(..2), global=true)]
    pub input: Vec<String>,

    /// Path of output
    #[arg(short, long, default_value = "out.png", global = true)]
    pub output: Option<String>,

    /// Only use every `viz_step` frame for vizualization
    #[arg(long, default_value_t = 10, global = true)]
    pub viz_step: usize,

    /// Controls framerate of video preview
    #[arg(long, default_value_t = 24, global = true)]
    pub viz_fps: u64,

    /// Path of video output showing optimization process [default: does not save]
    #[arg(short, long, global = true)]
    pub viz_output: Option<String>,

    /// Output directory to save individual frames to [default: tmpdir]
    #[arg(short = 'd', long, global = true)]
    pub img_dir: Option<String>,
}

#[derive(Args, Debug, Clone)]
/// Register two or more images and animate optimization process.
pub struct LKArgs {
    /// If enabled, use multi-level / hierarchical matching
    #[arg(long, default_value_t = false)]
    pub multi: bool,

    /// Downscale images before optimaization
    #[arg(long, default_value_t = 1.0)]
    pub downscale: f32,

    /// Number of LK iterations to use
    #[arg(long, default_value_t = 250)]
    pub iterations: i32,

    /// Stop optimization process when updates have an L1 norm less than this value
    #[arg(long, default_value_t = 1e-3)]
    pub early_stop: f32,

    /// Maximum number of levels to use (only used when `--multi`)
    #[arg(long, default_value_t = 8)]
    pub max_lvls: u32,
}

#[derive(Subcommand, Debug, Clone)]
pub enum Commands {
    /// Perform Lucas-Kanade homography estimation between two images and animate optimization process.
    LK(LKArgs),

    /// Estimate pairwise homographies and interpolate to all frames.
    Pano(LKArgs),
}

#[derive(ValueEnum, Clone, Copy, Debug)]
pub enum Transform {
    Identity,
    Rot90,
    Rot180,
    Rot270,
    FlipUD,
    FlipLR,
}
