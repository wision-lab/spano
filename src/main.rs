#![allow(dead_code)] // Todo: Remove

use anyhow::{anyhow, Result};
use blend::interpolate_bilinear;
use cli::{Commands, LKArgs};
use image::imageops::{resize, FilterType};
use image::{io::Reader as ImageReader, ImageBuffer, Rgb};
use std::fs::create_dir_all;
use tempfile::tempdir;

mod blend;
mod cli;
mod io;
mod lk;
mod utils;
mod warps;

use crate::cli::{Cli, Parser};
use crate::lk::{hierarchical_iclk, iclk};
use crate::utils::{animate_hierarchical_warp, animate_warp};
use crate::warps::{warp, Mapping};

fn print_type_of<T>(_: &T) {
    println!("{}", std::any::type_name::<T>())
}

fn match_imgpair(global_args: Cli, lk_args: LKArgs) -> Result<()> {
    let [img1_path, img2_path, ..] = &global_args.input[..] else {
        return Err(anyhow!("Exactly two inputs are required for --input."));
    };

    // Load images and resize if needed
    let img1 = ImageReader::open(img1_path)?.decode()?.into_rgb8();
    let img2 = ImageReader::open(img2_path)?.decode()?.into_rgb8();
    let (w, h) = img1.dimensions();
    let (w_, h_) = img2.dimensions();

    if (h != h_) || (w != w_) {
        return Err(anyhow!("Inputs need to be of same size."));
    }

    let w = (w as f32 / lk_args.downscale) as u32;
    let h = (h as f32 / lk_args.downscale) as u32;
    let img1 = resize(&img1, w, h, FilterType::CatmullRom);
    let img2 = resize(&img2, w, h, FilterType::CatmullRom);

    // Get img path or tempdir, ensure it exists.
    let tmp_dir = tempdir()?;
    let img_dir = global_args
        .img_dir
        .unwrap_or(tmp_dir.path().to_str().unwrap().to_owned());
    create_dir_all(&img_dir).ok();

    // Perform Matching
    let mapping = if !lk_args.multi {
        // Register images
        let (mapping, params_history) = iclk(
            &img1,
            &img2,
            Mapping::from_params(&[0.0; 8]),
            Some(lk_args.iterations),
            Some(lk_args.early_stop),
        )?;

        // Show Animation of optimization
        if global_args.viz_output.is_some() {
            animate_warp(
                img2_path,
                params_history,
                &img_dir,
                lk_args.downscale,
                Some(global_args.viz_fps),  // FPS
                Some(global_args.viz_step), // Step
                global_args.viz_output.as_deref(),
            )?;
        }
        mapping
    } else {
        // Register images
        let (mapping, params_history) = hierarchical_iclk(
            &img1,
            &img2,
            Mapping::from_params(&[0.0; 8]),
            Some(lk_args.iterations),
            (25, 25),
            lk_args.max_lvls,
            Some(lk_args.early_stop),
        )?;

        // Show Animation of optimization
        if global_args.viz_output.is_some() {
            animate_hierarchical_warp(
                img2_path,
                params_history,
                lk_args.downscale,
                &img_dir,
                Some(global_args.viz_fps),  // FPS
                Some(global_args.viz_step), // Step
                global_args.viz_output.as_deref(),
            )?;
        }
        mapping
    };

    let mut out = ImageBuffer::new(w, h);
    let get_pixel = |x, y| interpolate_bilinear(&img2, x, y).unwrap_or(Rgb([128, 0, 0]));
    warp(&mut out, mapping.warpfn(), get_pixel);
    out.save(global_args.output.unwrap_or("out.png".to_string()))?;

    println!("{:#?}", &mapping.mat);
    Ok(())
}

fn main() -> Result<()> {
    // Parse arguments defined in struct
    let args = Cli::parse();

    match &args.command {
        Some(Commands::LK(lk_args)) => match_imgpair(args.clone(), lk_args.clone()),
        None => Err(anyhow!("Only `LK` subcommand is currently implemented.")),
    }
}
