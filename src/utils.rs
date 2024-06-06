use std::{
    collections::HashMap,
    fs::{create_dir_all, remove_dir_all},
    path::Path,
};

use anyhow::Result;
use conv::ValueInto;
use image::{
    imageops::{resize, FilterType},
    io::Reader as ImageReader,
    EncodableLayout, Pixel, PixelWithColorType, Primitive, Rgb,
};
use imageproc::definitions::{Clamp, Image};
use indicatif::{ProgressBar, ProgressStyle};
use itertools::Itertools;
use photoncube2video::{
    ffmpeg::{ensure_ffmpeg, make_video},
    transforms::annotate,
};
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};

use crate::{kernels::Backend, warps::Mapping};

/// Conditionally setup a progressbar
pub fn get_pbar(len: usize, message: Option<&str>) -> ProgressBar {
    if let Some(msg) = message {
        ProgressBar::new(len as u64)
            .with_style(
                ProgressStyle::with_template(
                    "{msg} ETA:{eta}, [{elapsed_precise}] {wide_bar:.cyan/blue} {pos:>6}/{len:6}",
                )
                .expect("Invalid progress style."),
            )
            .with_message(msg.to_owned())
    } else {
        ProgressBar::hidden()
    }
}

#[allow(clippy::too_many_arguments)]
pub fn animate_warp<B: Backend>(
    img_path: &str,
    params_history: Vec<Vec<f32>>,
    img_dir: &str,
    scale: f32,
    fps: Option<u64>,
    step: Option<usize>,
    out_path: Option<&str>,
    message: Option<&str>,
) -> Result<()> {
    let img = ImageReader::open(img_path)?.decode()?.into_rgb8();
    let (w, h) = img.dimensions();

    // Clear dir, and make sure it exists
    if Path::new(&img_dir).is_dir() {
        remove_dir_all(img_dir)?;
    }
    create_dir_all(img_dir)?;

    let num_frames = params_history.len() / step.unwrap_or(100);
    let pbar = get_pbar(num_frames, message);
    params_history
        .into_par_iter()
        .step_by(step.unwrap_or(100))
        .enumerate()
        .for_each(|(i, params)| {
            let out = Mapping::<B>::from_params(params)
                .inverse()
                .rescale(1.0 / scale)
                .warp_image(&img, (h as usize, w as usize), Some(Rgb([128, 128, 128])));

            let path = Path::new(&img_dir).join(format!("frame{:06}.png", i));
            out.save(&path)
                .unwrap_or_else(|_| panic!("Could not save frame at {}!", &path.display()));
            pbar.inc(1);
        });
    pbar.finish_and_clear();

    ensure_ffmpeg(true);
    make_video(
        Path::new(&img_dir).join("frame%06d.png").to_str().unwrap(),
        out_path.unwrap_or("out.mp4"),
        fps.unwrap_or(25u64),
        num_frames as u64,
        Some("Making Video..."),
        None,
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn animate_hierarchical_warp<B: Backend>(
    img_path: &str,
    all_params_history: HashMap<u32, Vec<Vec<f32>>>,
    global_scale: f32,
    img_dir: &str,
    fps: Option<u64>,
    step: Option<usize>,
    out_path: Option<&str>,
    message: Option<&str>,
) -> Result<()> {
    let img = ImageReader::open(img_path)?.decode()?.into_rgb8();
    let (w, h) = img.dimensions();
    let mut offset = 0;

    // Clear dir, and make sure it exists
    if Path::new(&img_dir).is_dir() {
        remove_dir_all(img_dir)?;
    }
    create_dir_all(img_dir)?;

    let num_frames: usize = all_params_history
        .values()
        .map(|v| v.len() / step.unwrap_or(100))
        .sum();
    let pbar = get_pbar(num_frames, message);

    for (scale, params_history) in all_params_history.into_iter().sorted_by_key(|x| x.0).rev() {
        // This is not super efficient, but it's debug/viz code...
        let scale = scale as f32 * global_scale;
        let resized = resize(
            &resize(
                &img,
                (w as f32 / scale).round() as u32,
                (h as f32 / scale).round() as u32,
                FilterType::CatmullRom,
            ),
            w,
            h,
            FilterType::CatmullRom,
        );

        offset += params_history
            .into_par_iter()
            .step_by(step.unwrap_or(100))
            .enumerate()
            .map(|(i, params)| {
                let mut out = Mapping::<B>::from_params(params)
                    .inverse()
                    .rescale(1.0 / scale)
                    .warp_image(
                        &resized,
                        (h as usize, w as usize),
                        Some(Rgb([128, 128, 128])),
                    );
                annotate(
                    &mut out,
                    &format!("Scale: 1/{:.2}", scale),
                    Rgb([252, 186, 3]),
                );

                let path = Path::new(&img_dir).join(format!("frame{:06}.png", i + offset));
                out.save(&path)
                    .unwrap_or_else(|_| panic!("Could not save frame at {}!", &path.display()));
                pbar.inc(1)
            })
            .count();
    }
    pbar.finish_and_clear();

    ensure_ffmpeg(true);
    make_video(
        Path::new(&img_dir).join("frame%06d.png").to_str().unwrap(),
        out_path.unwrap_or("out.mp4"),
        fps.unwrap_or(25u64),
        num_frames as u64,
        Some("Making Video..."),
        None,
    );
    Ok(())
}

pub fn stabilized_video<P, B>(
    mappings: &[Mapping<B>],
    frames: &[Image<P>],
    img_dir: &str,
    fps: Option<u64>,
    step: Option<usize>,
    out_path: Option<&str>,
    message: Option<&str>,
) -> Result<()>
where
    P: Pixel + PixelWithColorType + Send + Sync,
    <P as Pixel>::Subpixel:
        num_traits::Zero + Clone + Copy + ValueInto<f32> + Send + Sync + Clamp<f32>,
    [<P as Pixel>::Subpixel]: EncodableLayout,
    f32: From<<P as Pixel>::Subpixel>,
    B: Backend,
{
    // Clear dir, and make sure it exists
    if Path::new(&img_dir).is_dir() {
        remove_dir_all(img_dir)?;
    }
    create_dir_all(img_dir)?;

    let sizes: Vec<_> = frames
        .iter()
        .map(|f| (f.width() as usize, f.height() as usize))
        .unique()
        .collect();
    let (extent, offset) = Mapping::maximum_extent(mappings, &sizes[..]);
    let [canvas_w, canvas_h]: [f32] = extent.into_data().convert().value[..] else {
        unreachable!("Canvas should have width and height")
    };
    let pbar = get_pbar(frames.len() / step.unwrap_or(100), message);

    (mappings, frames)
        .into_par_iter()
        .step_by(step.unwrap_or(100))
        .enumerate()
        .for_each(|(i, (map, frame))| {
            let img = map.transform(None, Some(offset.clone())).warp_image(
                frame,
                (canvas_h.ceil() as usize, canvas_w.ceil() as usize),
                Some(*P::from_slice(
                    &vec![<P as Pixel>::Subpixel::DEFAULT_MIN_VALUE; P::CHANNEL_COUNT as usize],
                )),
            );

            let path = Path::new(&img_dir).join(format!("frame{:06}.png", i));
            img.save(&path)
                .unwrap_or_else(|_| panic!("Could not save frame at {}!", &path.display()));
            pbar.inc(1);
        });
    pbar.finish_and_clear();

    ensure_ffmpeg(true);
    make_video(
        Path::new(&img_dir).join("frame%06d.png").to_str().unwrap(),
        out_path.unwrap_or("out.mp4"),
        fps.unwrap_or(25u64),
        frames.len() as u64,
        Some("Stabilizing..."),
        None,
    );
    Ok(())
}
