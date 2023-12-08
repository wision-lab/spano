use std::collections::HashMap;
use std::fs::File;
use std::fs::{create_dir_all, remove_dir_all};
use std::io::BufReader;
use std::path::Path;
use std::convert::From;

use anyhow::Result;
use image::imageops::{resize, FilterType};
use image::{io::Reader as ImageReader, GrayImage, ImageBuffer, Rgb, RgbImage};
use imageproc::drawing::{draw_text_mut, text_size};
use itertools::Itertools;
use rusttype::{Font, Scale};
use ndarray::{Array2, Axis};

use crate::blend::interpolate_bilinear;
use crate::io::{ensure_ffmpeg, make_video};
use crate::warps::{warp, Mapping};


pub fn array2grayimage(frame: Array2<u8>) -> Option<GrayImage> {
    GrayImage::from_raw(
        frame.len_of(Axis(1)) as u32,
        frame.len_of(Axis(0)) as u32,
        frame.into_raw_vec(),
    )
}

pub fn annotate(frame: &mut RgbImage, text: &str) {
    let font = Vec::from(include_bytes!("DejaVuSans.ttf") as &[u8]);
    let font = Font::try_from_vec(font).unwrap();
    let scale = Scale { x: 20.0, y: 20.0 };

    draw_text_mut(frame, Rgb([252, 186, 3]), 5, 5, scale, &font, text);
    text_size(scale, &font, text);
}

pub fn animate_warp(
    img_path: &str,
    params_path: &str,
    img_dir: &str,
    scale: f32,
    fps: Option<u64>,
    step: Option<usize>,
    out_path: Option<&str>,
) -> Result<()> {
    let img = ImageReader::open(img_path)?.decode()?.into_rgb8();
    let (w, h) = img.dimensions();

    let file = File::open(params_path)?;
    let reader = BufReader::new(file);
    let params_history: Vec<Vec<f32>> = serde_json::from_reader(reader)?;

    if Path::new(&img_dir).is_dir() {
        remove_dir_all(img_dir)?;
    }
    create_dir_all(img_dir)?;

    for (i, params) in params_history
        .iter()
        .step_by(step.unwrap_or(100))
        .enumerate()
    {
        let mut out = ImageBuffer::new(w, h);
        let get_pixel = |x, y| interpolate_bilinear(&img, x, y).unwrap_or(Rgb([128, 128, 128]));
        warp(
            &mut out,
            Mapping::from_params(params).inverse()
                .transform(
                    Some(Mapping::scale(scale, scale)),
                    Some(Mapping::scale(1.0/scale, 1.0/scale)), 
                )
                .warpfn(),
            get_pixel,
        );

        let path = Path::new(&img_dir).join(format!("frame{:06}.png", i));
        out.save(&path)
            .unwrap_or_else(|_| panic!("Could not save frame at {}!", &path.display()));
    }

    ensure_ffmpeg(true);
    make_video(
        Path::new(&img_dir).join("frame%06d.png").to_str().unwrap(),
        &out_path.unwrap_or("out.mp4"),
        fps.unwrap_or(25u64),
        0,
        None,
    );
    Ok(())
}

pub fn animate_hierarchical_warp(
    img_path: &str,
    params_path: &str,
    img_dir: &str,
    fps: Option<u64>,
    step: Option<usize>,
    out_path: Option<&str>,
) -> Result<()> {
    let img = ImageReader::open(img_path)?.decode()?.into_rgb8();
    let (w, h) = img.dimensions();

    let file = File::open(params_path)?;
    let reader = BufReader::new(file);
    let all_params_history: HashMap<u32, Vec<Vec<f32>>> = serde_json::from_reader(reader)?;
    let mut i = 0;

    if Path::new(&img_dir).is_dir() {
        remove_dir_all(img_dir)?;
    }
    create_dir_all(img_dir)?;

    for (scale, params_history) in all_params_history.iter().sorted_by_key(|x| x.0).rev() {
        // This is not super efficient, but it's debug/viz code...
        let resized = resize(
            &resize(
                &img, (w as f32 / *scale as f32).round() as u32, 
                (h as f32 / *scale as f32).round() as u32, 
                FilterType::CatmullRom
            ),
            w, h, FilterType::CatmullRom
        );
        let get_pixel = |x, y| interpolate_bilinear(
            &resized, x, y).unwrap_or(Rgb([128, 128, 128])
        );

        for params in params_history.iter()
            .step_by(step.unwrap_or(10))
        {
            let mut out = ImageBuffer::new(w, h);
            warp(
                &mut out,
                Mapping::from_params(params).inverse()
                    .transform(
                        Some(Mapping::scale(*scale as f32, *scale as f32)),
                        Some(Mapping::scale(1.0/(*scale as f32), 1.0/(*scale as f32))), 
                    )
                    .warpfn(),
                get_pixel,
            );
            annotate(&mut out, &format!("Scale: 1/{}", *scale as u32));

            let path = Path::new(&img_dir).join(format!("frame{:06}.png", i));
            out.save(&path)
                .unwrap_or_else(|_| panic!("Could not save frame at {}!", &path.display()));
            i += 1;
        }
    }

    ensure_ffmpeg(true);
    make_video(
        Path::new(&img_dir).join("frame%06d.png").to_str().unwrap(),
        &out_path.unwrap_or("out.mp4"),
        fps.unwrap_or(25u64),
        0,
        None,
    );
    Ok(())
}
