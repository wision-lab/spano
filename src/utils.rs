use std::{
    collections::HashMap,
    fs::{create_dir_all, remove_dir_all},
    path::{Path, PathBuf},
};

use anyhow::{anyhow, Result};
use conv::ValueInto;
use image::{
    imageops::{resize, FilterType},
    EncodableLayout, Luma, Pixel, PixelWithColorType, Primitive, Rgb, RgbImage,
};
use imageproc::definitions::{Clamp, Image};
use indicatif::{ProgressBar, ProgressStyle};
use itertools::Itertools;
use photoncube2video::{
    ffmpeg::{ensure_ffmpeg, make_video},
    signals::DeferredSignal,
    transforms::{annotate, array3_to_image, gray_to_rgbimage},
};
use pyo3::prelude::*;
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};
use tempfile::tempdir;

use crate::{lk::pyarray_to_im_bridge, warps::Mapping};

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
pub fn animate_warp<P: AsRef<Path>>(
    img: &RgbImage,
    params_history: HashMap<u32, Vec<Vec<f32>>>,
    img_dir: Option<P>,
    scale: f32,
    fps: Option<u64>,
    step: Option<usize>,
    out_path: Option<P>,
    message: Option<&str>,
) -> Result<()> {
    if img_dir.is_none() && out_path.is_none() {
        return Err(anyhow!(
            "At least one of `img_dir`, `out_path` should be set."
        ));
    }

    let (w, h) = img.dimensions();
    let mut offset = 0;

    // Clear dir, and make sure it exists
    let tmp_dir = tempdir()?;
    let img_dir = img_dir
        .map(|p| {
            let path: &Path = p.as_ref();
            path.to_owned()
        })
        .unwrap_or(tmp_dir.path().to_path_buf());

    if img_dir.is_dir() {
        remove_dir_all(&img_dir)?;
    }
    create_dir_all(&img_dir)?;

    let num_frames: usize = params_history
        .values()
        .map(|v| v.len() / step.unwrap_or(100))
        .sum();
    let pbar = get_pbar(num_frames, message);

    for (current_scale, p_hist) in params_history.into_iter().sorted_by_key(|x| x.0).rev() {
        // This is not super efficient, but it's debug/viz code...
        let resized = resize(
            &resize(
                img,
                (w as f32 / current_scale as f32).round() as u32,
                (h as f32 / current_scale as f32).round() as u32,
                FilterType::CatmullRom,
            ),
            w,
            h,
            FilterType::CatmullRom,
        );

        offset += p_hist
            .into_par_iter()
            .step_by(step.unwrap_or(100))
            .enumerate()
            .map(|(i, params)| {
                let mut out = Mapping::from_params(params)
                    .inverse()
                    .rescale(1.0 / current_scale as f32)
                    .warp_image(&resized, (h as usize, w as usize), Some(Rgb([0, 0, 0])));

                annotate(
                    &mut out,
                    &format!("Scale: 1/{:.2}", current_scale as f32 * scale),
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

    if let Some(out_path) = out_path {
        ensure_ffmpeg(true);
        make_video(
            Path::new(&img_dir).join("frame%06d.png").to_str().unwrap(),
            out_path,
            fps.unwrap_or(25u64),
            num_frames as u64,
            None,
            None,
            Some("Making Video..."),
            None,
        );
    }
    Ok(())
}

#[pyfunction]
#[pyo3(
    name = "animate_warp",
    signature = (img, params_history, img_dir=None, scale=1.0, fps=25, step=100, out_path=None, message=None)
)]
#[allow(clippy::too_many_arguments)]
pub fn animate_warp_py<'py>(
    py: Python<'py>,
    img: &Bound<'py, PyAny>,
    params_history: HashMap<u32, Vec<Vec<f32>>>,
    img_dir: Option<PathBuf>,
    scale: f32,
    fps: Option<u64>,
    step: Option<usize>,
    out_path: Option<PathBuf>,
    message: Option<&str>,
) -> PyResult<()> {
    let _defer = DeferredSignal::new(py, "SIGINT")?;

    let img = pyarray_to_im_bridge(img)?;
    let (_h, _w, c) = img.dim();

    let img = match c {
        1 => gray_to_rgbimage(&array3_to_image::<Luma<_>>(img.to_owned())),
        3 => array3_to_image(img.to_owned()),
        _ => return Err(anyhow!("Expected one or three channels, got {c}.").into()),
    };

    animate_warp(
        &img,
        params_history,
        img_dir,
        scale,
        fps,
        step,
        out_path,
        message,
    )?;
    Ok(())
}

pub fn stabilized_video<P, R: AsRef<Path>>(
    mappings: &[Mapping],
    frames: &[Image<P>],
    img_dir: Option<R>,
    fps: Option<u64>,
    step: Option<usize>,
    out_path: Option<R>,
    message: Option<&str>,
) -> Result<()>
where
    P: Pixel + PixelWithColorType + Send + Sync,
    <P as Pixel>::Subpixel:
        num_traits::Zero + Clone + Copy + ValueInto<f32> + Send + Sync + Clamp<f32>,
    [<P as Pixel>::Subpixel]: EncodableLayout,
    f32: From<<P as Pixel>::Subpixel>,
{
    if img_dir.is_none() && out_path.is_none() {
        return Err(anyhow!(
            "At least one of `img_dir`, `out_path` should be set."
        ));
    }

    // Clear dir, and make sure it exists
    let tmp_dir = tempdir()?;
    let img_dir = img_dir
        .map(|p| {
            let path: &Path = p.as_ref();
            path.to_owned()
        })
        .unwrap_or(tmp_dir.path().to_path_buf());

    if img_dir.is_dir() {
        remove_dir_all(&img_dir)?;
    }
    create_dir_all(&img_dir)?;

    let sizes: Vec<_> = frames
        .iter()
        .map(|f| (f.width() as usize, f.height() as usize))
        .unique()
        .collect();
    let (extent, offset) = Mapping::maximum_extent(mappings, &sizes[..]);
    let [canvas_w, canvas_h] = extent.to_vec()[..] else {
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

    if let Some(out_path) = out_path {
        ensure_ffmpeg(true);
        make_video(
            Path::new(&img_dir).join("frame%06d.png").to_str().unwrap(),
            out_path,
            fps.unwrap_or(25u64),
            frames.len() as u64,
            None,
            None,
            Some("Stabilizing..."),
            None,
        );
    }
    Ok(())
}
