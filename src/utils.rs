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
    EncodableLayout, Pixel, PixelWithColorType, Rgb,
};
use imageproc::definitions::{Clamp, Image};
use itertools::Itertools;
use photoncube2video::{
    ffmpeg::{ensure_ffmpeg, make_video},
    transforms::annotate,
};
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};

use crate::warps::{warp_image, Mapping};

pub fn animate_warp(
    img_path: &str,
    params_history: Vec<Vec<f32>>,
    img_dir: &str,
    scale: f32,
    fps: Option<u64>,
    step: Option<usize>,
    out_path: Option<&str>,
) -> Result<()> {
    let img = ImageReader::open(img_path)?.decode()?.into_rgb8();
    let (w, h) = img.dimensions();

    // Clear dir, and make sure it exists
    if Path::new(&img_dir).is_dir() {
        remove_dir_all(img_dir)?;
    }
    create_dir_all(img_dir)?;

    params_history
        .par_iter()
        .step_by(step.unwrap_or(100))
        .enumerate()
        .for_each(|(i, params)| {
            let out = warp_image(
                &Mapping::from_params(params).inverse().rescale(1.0 / scale),
                &img,
                (h as usize, w as usize),
                Some(Rgb([128, 128, 128])),
            );

            let path = Path::new(&img_dir).join(format!("frame{:06}.png", i));
            out.save(&path)
                .unwrap_or_else(|_| panic!("Could not save frame at {}!", &path.display()));
        });

    ensure_ffmpeg(true);
    make_video(
        Path::new(&img_dir).join("frame%06d.png").to_str().unwrap(),
        out_path.unwrap_or("out.mp4"),
        fps.unwrap_or(25u64),
        0,
        None,
        None,
    );
    Ok(())
}

pub fn animate_hierarchical_warp(
    img_path: &str,
    all_params_history: HashMap<u32, Vec<Vec<f32>>>,
    global_scale: f32,
    img_dir: &str,
    fps: Option<u64>,
    step: Option<usize>,
    out_path: Option<&str>,
) -> Result<()> {
    let img = ImageReader::open(img_path)?.decode()?.into_rgb8();
    let (w, h) = img.dimensions();
    let mut offset = 0;

    // Clear dir, and make sure it exists
    if Path::new(&img_dir).is_dir() {
        remove_dir_all(img_dir)?;
    }
    create_dir_all(img_dir)?;

    for (scale, params_history) in all_params_history.iter().sorted_by_key(|x| x.0).rev() {
        // This is not super efficient, but it's debug/viz code...
        let scale = *scale as f32 * global_scale;
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

        // for params in params_history.iter().step_by(step.unwrap_or(10)) {
        offset += params_history
            .par_iter()
            .step_by(step.unwrap_or(100))
            .enumerate()
            .map(|(i, params)| {
                let mut out = warp_image(
                    &Mapping::from_params(params).inverse().rescale(1.0 / scale),
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
            })
            .count();
    }

    ensure_ffmpeg(true);
    make_video(
        Path::new(&img_dir).join("frame%06d.png").to_str().unwrap(),
        out_path.unwrap_or("out.mp4"),
        fps.unwrap_or(25u64),
        0,
        None,
        None,
    );
    Ok(())
}

pub fn stabilized_video<P>(
    mappings: &[Mapping],
    frames: &[Image<P>],
    img_dir: &str,
    fps: Option<u64>,
    step: Option<usize>,
    out_path: Option<&str>,
) -> Result<()>
where
    P: Pixel + PixelWithColorType + Send + Sync,
    <P as Pixel>::Subpixel:
        num_traits::Zero + Clone + Copy + ValueInto<f32> + Send + Sync + Clamp<f32>,
    [<P as Pixel>::Subpixel]: EncodableLayout,
    f32: From<<P as Pixel>::Subpixel>,
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
    let [canvas_w, canvas_h] = extent.to_vec()[..] else {
        unreachable!("Canvas should have width and height")
    };

    mappings
        .par_iter()
        .step_by(step.unwrap_or(100))
        .enumerate()
        .for_each(|(i, map)| {
            let img = warp_image(
                &map.transform(None, Some(offset.clone())),
                &frames[i],
                (canvas_h.ceil() as usize, canvas_w.ceil() as usize),
                None,
            );

            let path = Path::new(&img_dir).join(format!("frame{:06}.png", i));
            img.save(&path)
                .unwrap_or_else(|_| panic!("Could not save frame at {}!", &path.display()));
        });

    ensure_ffmpeg(true);
    make_video(
        Path::new(&img_dir).join("frame%06d.png").to_str().unwrap(),
        out_path.unwrap_or("out.mp4"),
        fps.unwrap_or(25u64),
        0,
        None,
        None,
    );
    Ok(())
}
