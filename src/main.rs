#![allow(dead_code)] // Todo: Remove

use anyhow::{anyhow, Result};
use blend::interpolate_bilinear;
use image::imageops::{resize, FilterType};
use image::{io::Reader as ImageReader, ImageBuffer, Rgb};
use std::fs;

mod blend;
mod io;
mod lk;
mod utils;
mod warps;
mod cli;

use crate::lk::{iclk, hierarchical_iclk};
use crate::warps::{warp, Mapping};
use crate::utils::{animate_warp, animate_hierarchical_warp};
use crate::cli::{Args, Parser};


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

    // use splines::impl_Interpolate;
    // use splines::{Interpolation, Key, Spline};
    // use nalgebra::SVector;
    // impl_Interpolate!(f32, SVector<f32, 8>, std::f32::consts::PI);

    // ----------------------------------------------
    // Parse arguments defined in struct
    let args = Args::parse();
    let [img1_path, img2_path, ..] = &args.input[..] else {
        // Note: This should never occur as clap validates the args... 
        return Err(anyhow!(
            "Exactly two inputs are required for --input."
        ));
    };

    // Load images and resize if needed
    let img1 = ImageReader::open(img1_path)?
        .decode()?
        .into_rgb8();
    let img2 = ImageReader::open(img2_path)?
        .decode()?
        .into_rgb8();
    let (w, h) = img1.dimensions();
    let (w_, h_) = img2.dimensions();

    if (h != h_) || (w != w_) {
        return Err(anyhow!(
            "Inputs need to be of same size."
        ));
    }
    let w = (w as f32 / args.downscale) as u32; 
    let h = (h as f32 / args.downscale) as u32; 
    let img1 = resize(&img1, w, h, FilterType::CatmullRom);
    let img2 = resize(&img2, w, h, FilterType::CatmullRom);

    // Register images
    let (mapping, params_history) = hierarchical_iclk(
        &img1,
        &img2,
        Mapping::from_params(&[0.0; 8]),
        Some(args.iterations),
        (25, 25),
        8,
        Some(0.1),
    )?;

    // let (mapping, params_history, dp_history) =
    //     iclk(&img1, &img2, Mapping::from_params(&[0.0; 8]), Some(args.iterations))?;

    // Save optimization results and history
    let serialized = serde_json::to_string_pretty(&params_history)?;
    fs::write("params_hist.json", serialized)?;

    println!("{:?}", &mapping.mat);

    let mut out = ImageBuffer::new(w, h);
    let get_pixel = |x, y| interpolate_bilinear(&img2, x, y).unwrap_or(Rgb([128, 0, 0]));
    warp(&mut out, mapping.warpfn(), get_pixel);
    
    out.save("out.png")?;
    img1.save("img1.png")?;
    img2.save("img2.png")?;

    animate_hierarchical_warp(
        img2_path,
        "params_hist.json",
        "tmp/",
        Some(args.viz_fps), // FPS
        Some(args.viz_step), // Step
        args.output.as_deref()
    )?;

    // animate_warp(
    //     img2_path,
    //     "params_hist.json",
    //     "tmp/",
    //     args.downscale,
    //     Some(args.viz_fps), // FPS
    //     Some(args.viz_step), // Step
    //     args.output.as_deref()
    // )?;

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
