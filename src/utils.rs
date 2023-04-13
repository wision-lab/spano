use std::fs::File;
use std::fs::{create_dir_all, remove_dir_all};
use std::io::BufReader;
use std::path::Path;

use anyhow::Result;
use image::imageops::{resize, FilterType};
use image::{io::Reader as ImageReader, GrayImage, ImageBuffer, Rgb};
use ndarray::{Array2, Axis};

use crate::blend::interpolate_bilinear;
use crate::io::{ensure_ffmpeg, make_video};
use crate::warps::{warp, Mapping};

#[allow(dead_code)]
pub fn array2grayimage(frame: Array2<u8>) -> Option<GrayImage> {
    GrayImage::from_raw(
        frame.len_of(Axis(1)) as u32,
        frame.len_of(Axis(0)) as u32,
        frame.into_raw_vec(),
    )
}

#[allow(dead_code)]
pub fn animate_warp(
    img_path: &str,
    params_path: &str,
    img_dir: &str,
    out_shape: Option<(u32, u32)>,
    fps: Option<u64>,
    step: Option<usize>,
) -> Result<()> {
    let (img, (w, h)) = {
        let img = ImageReader::open(img_path)?.decode()?.into_rgb8();

        if let Some((w, h)) = out_shape {
            (resize(&img, w, h, FilterType::CatmullRom), (w, h))
        } else {
            let (w, h) = img.dimensions();
            (img, (w, h))
        }
    };

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
        // warp(
        //     &mut out,
        //     Mapping::from_params(params).inverse().warpfn_centered(img.dimensions()),
        //     get_pixel,
        // );
        warp(&mut out, Mapping::from_params(params).inverse().warpfn(), get_pixel);

        let path = Path::new(&img_dir).join(format!("frame{:06}.png", i));
        out.save(&path)
            .unwrap_or_else(|_| panic!("Could not save frame at {}!", &path.display()));
    }

    ensure_ffmpeg(true);
    make_video(
        Path::new(&img_dir).join("frame%06d.png").to_str().unwrap(),
        "out.mp4",
        fps.unwrap_or(25u64),
        0,
        None,
    );
    Ok(())
}
