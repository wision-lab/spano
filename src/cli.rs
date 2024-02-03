pub use clap::Parser;
use clap::{Args, Subcommand};
use photoncube2video::transforms::Transform;

fn validate_normalized(p: &str) -> Result<f32, String> {
    let value = p.parse::<f32>().map_err(|_| "Invalid value")?;
    if (0.0..=1.0).contains(&value) {
        Ok(value)
    } else {
        Err("Value must be between 0.0 and 1.0".to_string())
    }
}

#[derive(Parser, Debug, Clone)]
#[command(author, version, about, long_about = None, args_conflicts_with_subcommands = true)]
/// Register two or more images and animate optimization process.
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,

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

    /// Apply transformations to each frame (these can be composed)
    #[arg(short, long, value_enum, num_args(0..), global=true)]
    pub transform: Vec<Transform>,
}

#[derive(Args, Debug, Clone)]
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

    /// Stop optimization process when parameter updates have an L1 norm less than this value
    #[arg(long, default_value_t = 1e-3)]
    pub early_stop: f32,

    /// Controls window size over which parameter updates are averaged, this average is then used in `early_stop`
    #[arg(long, default_value_t = 10)]
    pub patience: usize,

    /// Maximum number of levels to use (only used when `--multi`)
    #[arg(long, default_value_t = 8)]
    pub max_lvls: u32,

    /// Save parameters of optimization to file
    #[arg(long)]
    pub params_path: Option<String>,
}

#[derive(Args, Debug, Clone)]
pub struct PanoArgs {
    #[command(flatten)]
    pub lk_args: LKArgs,

    /// Index of binary frame at which to start the preview from (inclusive)
    #[arg(short, long, default_value = None)]
    pub start: Option<isize>,

    /// Index of binary frame at which to stop the preview at (exclusive)
    #[arg(short, long, default_value = None)]
    pub end: Option<isize>,

    /// Normalized index (i.e [0, 1]) of `with-respect-to` frame, frame with identity warp
    #[arg(long, default_value_t = 0.5, value_parser=validate_normalized)]
    pub wrt: f32,

    /// Number of frames to average together
    #[arg(long, default_value_t = 256)]
    pub burst_size: usize,

    /// If enabled, invert the SPAD's response (Bernoulli process)
    #[arg(long, action)]
    pub invert_response: bool,

    /// If enabled, apply sRGB tonemapping to output
    #[arg(long, action)]
    pub tonemap2srgb: bool,

    /// If enabled, swap columns that are out of order and crop to 254x496
    #[arg(long, action)]
    pub colorspad_fix: bool,

    /// Path of color filter array to use for demosaicing
    #[arg(long, default_value = None)]
    pub cfa_path: Option<String>,

    /// Path of inpainting mask to use for filtering out dead/hot pixels
    #[arg(long, num_args(0..))]
    pub inpaint_path: Vec<String>,
}

#[derive(Subcommand, Debug, Clone)]
pub enum Commands {
    /// Perform Lucas-Kanade homography estimation between two images and animate optimization process.
    LK(LKArgs),

    /// Estimate pairwise homographies and interpolate to all frames.
    Pano(PanoArgs),
}
