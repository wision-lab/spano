#![allow(dead_code)] // Todo: Remove

use anyhow::Result;
use blend::interpolate_bilinear;
use image::imageops::{resize, FilterType};
use image::{io::Reader as ImageReader, ImageBuffer, Rgb};

use warps::warp;

mod blend;
mod io;
mod lk;
mod utils;
mod warps;

use crate::lk::iclk;
use crate::utils::animate_warp;
use crate::warps::TransformationType;

#[allow(dead_code)]
fn print_type_of<T>(_: &T) {
    println!("{}", std::any::type_name::<T>())
}

fn main() -> Result<()> {
    // let dst = polygon_sdf_vec(
    //     array![[0.0, 0.0], [500.0, 100.0], [800.0, 900.0], [10.0, 990.0]]
    // );
    // let values: Vec<f32> = (0..1000).cartesian_product(0..1000).collect::<Vec<_>>().par_iter().map(
    //     |(x, y)| dst(*x as f32, *y as f32)
    // ).collect();

    // let max = values.par_iter().max_by(|x, y| x.partial_cmp(&y).unwrap()).unwrap();
    // let min = values.par_iter().min_by(|x, y| x.partial_cmp(&y).unwrap()).unwrap();
    // println!("{max}, {min}");

    // let out: ImageBuffer<Luma<u8>, Vec<u8>> = ImageBuffer::from_vec(
    //     1000, 1000,
    //     values.iter().map(|p|
    //         ((p-min) / (max-min) * 255.0) as u8
    //     ).collect()
    // ).unwrap();
    // out.save("pano.png")?;

    // -----------------------------------------------

    // let img1 = ImageReader::open("madison.jpg")?.decode()?.into_rgb8();
    // let mut out = ImageBuffer::new(1000, 1000);
    // let get_pixel = |x, y| interpolate_bilinear(&img1, x, y, Rgb([0, 0, 0]));
    // let mapping = Mapping::from_params(vec![-100.0, -100.0]);
    // warp(&mut out, mapping, get_pixel);
    // out.save("pano.png")?;

    // ----------------------------------------------
    let (w, h) = (128, 128);

    let img1 = ImageReader::open("A.png")?.decode()?.into_rgb8();
    let img1 = resize(&img1, w, h, FilterType::CatmullRom);

    let img2 = ImageReader::open("B.png")?.decode()?.into_rgb8();
    let img2 = resize(&img2, w, h, FilterType::CatmullRom);

    let mapping = iclk(&img1, &img2, TransformationType::Affine, Some(2500))?;
    // let mapping = Mapping::from_params(&vec![0.0, 0.0]);
    println!("{:?}", &mapping);

    let mut out = ImageBuffer::new(w, h);
    let get_pixel = |x, y| interpolate_bilinear(&img2, x, y).unwrap_or(Rgb([128, 0, 0]));
    warp(&mut out, mapping.warpfn(), get_pixel);
    // warp(
    //     &mut out,
    //     mapping.warpfn_centered(img2.dimensions()),
    //     get_pixel,
    // );
    out.save("out.png")?;

    img1.save("img1.png")?;
    img2.save("img2.png")?;

    animate_warp(
        "B.png",
        "params_hist.json",
        "tmp/",
        Some((w, h)),
        Some(5),
        Some(50),
    )?;

    // ----------------------------------------------

    // use std::fs::File;
    // use std::io::BufReader;
    // use std::path::Path;

    // let (w, h) = (512, 512);

    // let img2 = ImageReader::open("2-c.png")?.decode()?.into_rgb8();
    // // let img2 = resize(&img2, w, h, FilterType::CatmullRom);

    // let file = File::open("params_hist.json")?;
    // let reader = BufReader::new(file);
    // let params_history: Vec<Vec<f32>> = serde_json::from_reader(reader)?;
    // let img_dir = "tmp/".to_string();

    // for (i, params) in params_history.iter().step_by(1000).enumerate() {
    //     let mut out = ImageBuffer::new(w, h);
    //     let get_pixel = |x, y| interpolate_bilinear(&img2, x, y).unwrap_or(Rgb([128, 128, 128]));
    //     // warp(&mut out, mapping.warpfn(), get_pixel);
    //     warp(&mut out, Mapping::from_params(params).warpfn_centered(img2.dimensions()), get_pixel);

    //     let path = Path::new(&img_dir).join(format!("frame{:06}.png", i));
    //         out
    //             .save(&path)
    //             .unwrap_or_else(|_| panic!("Could not save frame at {}!", &path.display()));

    // }

    // ensure_ffmpeg(true);
    // make_video(
    //     Path::new(&img_dir).join("frame%06d.png").to_str().unwrap(),
    //     "out.mp4",
    //     25,
    //     300,
    //     None,
    // );

    // ----------------------------------------------

    Ok(())
}
