use std::collections::HashMap;
use std::fs::{create_dir_all, remove_dir_all};
use std::path::{Path, PathBuf};

use anyhow::Result;
use glob::glob;
use image::imageops::{resize, FilterType};
use image::{io::Reader as ImageReader, Rgb};
use itertools::Itertools;
use natord::compare;

use crate::blend::interpolate_bilinear_with_bkg;
use crate::ffmpeg::{ensure_ffmpeg, make_video};
use crate::transforms::annotate;
use crate::warps::{warp_image, Mapping};

pub fn sorted_glob(path: &Path, pattern: &str) -> Result<Vec<String>> {
    let paths: Vec<PathBuf> =
        glob(path.join(pattern).to_str().unwrap())?.collect::<Result<Vec<PathBuf>, _>>()?;
    let paths: Vec<&str> = paths
        .iter()
        .map(|v| v.to_str())
        .collect::<Option<Vec<&str>>>()
        .unwrap();
    let mut paths: Vec<String> = paths.iter().map(|p| p.to_string()).collect();
    paths.sort_by(|a, b| compare(a, b));

    Ok(paths)
}

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

    if Path::new(&img_dir).is_dir() {
        remove_dir_all(img_dir)?;
    }
    create_dir_all(img_dir)?;

    for (i, params) in params_history
        .iter()
        .step_by(step.unwrap_or(100))
        .enumerate()
    {
        let get_pixel = |x, y| interpolate_bilinear_with_bkg(&img, x, y, Rgb([128, 128, 128]));
        let out = warp_image(
            &Mapping::from_params(params).inverse().rescale(1.0 / scale),
            get_pixel,
            w as usize,
            h as usize,
        );

        let path = Path::new(&img_dir).join(format!("frame{:06}.png", i));
        out.save(&path)
            .unwrap_or_else(|_| panic!("Could not save frame at {}!", &path.display()));
    }

    ensure_ffmpeg(true);
    make_video(
        Path::new(&img_dir).join("frame%06d.png").to_str().unwrap(),
        out_path.unwrap_or("out.mp4"),
        fps.unwrap_or(25u64),
        0,
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
    let mut i = 0;

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
        let get_pixel = |x, y| interpolate_bilinear_with_bkg(&resized, x, y, Rgb([128, 128, 128]));

        for params in params_history.iter().step_by(step.unwrap_or(10)) {
            let mut out = warp_image(
                &Mapping::from_params(params).inverse().rescale(1.0 / scale),
                get_pixel,
                w as usize,
                h as usize,
            );
            annotate(&mut out, &format!("Scale: 1/{:.2}", scale));

            let path = Path::new(&img_dir).join(format!("frame{:06}.png", i));
            out.save(&path)
                .unwrap_or_else(|_| panic!("Could not save frame at {}!", &path.display()));
            i += 1;
        }
    }

    ensure_ffmpeg(true);
    make_video(
        Path::new(&img_dir).join("frame%06d.png").to_str().unwrap(),
        out_path.unwrap_or("out.mp4"),
        fps.unwrap_or(25u64),
        0,
        None,
    );
    Ok(())
}
