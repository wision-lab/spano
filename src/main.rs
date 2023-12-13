#![allow(dead_code)] // Todo: Remove
#![allow(unused_imports)]

use anyhow::{anyhow, Result};
use image::imageops::{resize, FilterType};
use image::Luma;
use image::{io::Reader as ImageReader, ImageBuffer, Rgb};
use indicatif::{ProgressIterator, ProgressStyle};
use itertools::Itertools;
use memmap2::{Mmap, MmapOptions};
use ndarray::{array, s, Array, Array1, Array2, Array3, ArrayView3, Axis, Slice};
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};
use std::fs::create_dir_all;
use std::iter::successors;
use std::path::Path;
use tempfile::tempdir;

mod blend;
mod cli;
mod ffmpeg;
mod io;
mod lk;
mod transforms;
mod utils;
mod warps;

use crate::blend::{interpolate_bilinear_with_bkg, polygon_sdf_vec};
use crate::cli::{Cli, Commands, LKArgs, Parser};
use crate::ffmpeg::make_video;
use crate::io::PhotonCube;
use crate::lk::{gradients, hierarchical_iclk, iclk, iclk_grayscale};
use crate::transforms::{array2grayimage, process_colorspad, unpack_single};
use crate::utils::{animate_hierarchical_warp, animate_warp};
use crate::warps::{warp, Mapping, TransformationType};

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
    let get_pixel = |x, y| interpolate_bilinear_with_bkg(&img2, x, y, Rgb([128, 0, 0]));
    warp(&mut out, mapping.warpfn(), get_pixel);
    out.save(global_args.output.unwrap_or("out.png".to_string()))?;

    println!("{:#?}", &mapping.mat);
    Ok(())
}

fn main() -> Result<()> {
    // Parse arguments defined in struct
    let args = Cli::parse();

    match &args.command {
        None => Err(anyhow!("Only `LK` subcommand is currently implemented.")),
        Some(Commands::LK(lk_args)) => match_imgpair(args.clone(), lk_args.clone()),
        Some(Commands::Pano(lk_args)) => {
            let cube = PhotonCube::load("../photoncube2video/data/binary.npy")?;
            let cube_view = cube.view()?;
            let cube_view = cube_view.slice_axis(Axis(0), Slice::new(0, Some(256 * 250), 1));
            let wrt_normalized_idx = 0.5;
            let burst_size = 256;

            // Create parallel iterator over all chunks of frames and process them
            let (virtual_exposures, sizes): (Vec<_>, Vec<(u32, u32)>) = cube_view
                .axis_chunks_iter(Axis(0), burst_size)
                // Make it parallel
                .into_par_iter()
                .map(|group| {
                    let frame = group
                        // Iterate over all bitplanes in group
                        .axis_iter(Axis(0))
                        // Unpack every frame in group as a f32 array
                        .map(|bitplane| unpack_single::<f32>(&bitplane, 1).unwrap())
                        // Sum frames together (.sum not implemented for this type)
                        .reduce(|acc, e| acc + e)
                        .unwrap()
                        // Compute mean values and save as uint8's
                        .mapv(|v| (v / (burst_size as f32) * 255.0).round() as u8);

                    // Convert to image and resize
                    let img = array2grayimage(process_colorspad(frame));
                    let w = (img.width() as f32 / lk_args.downscale).round() as u32;
                    let h = (img.height() as f32 / lk_args.downscale).round() as u32;
                    (resize(&img, w, h, FilterType::CatmullRom), (w, h))
                })
                // Force iteratpor to run to completion to get correct ordering
                .unzip();

            // Estimate pairwise registration
            let mut mappings: Vec<Mapping> = virtual_exposures
                .iter()
                .progress()
                .with_style(ProgressStyle::with_template(
                    "{msg} ETA:{eta}, [{elapsed_precise}] {wide_bar:.cyan/blue} {pos:>6}/{len:6}",
                )?)
                .with_message(format!("Lvl 1:").to_owned())
                .tuple_windows()
                .map(|(src, dst)| {
                    iclk_grayscale(
                        src,                             // im1_gray
                        dst,                             // im2_gray
                        gradients(&dst),                 // im2_grad,
                        Mapping::from_params(&[0.0; 8]), // init_mapping
                        Some(lk_args.iterations),        // max_iters
                        Some(lk_args.early_stop),        // stop_early
                        None,                            // message
                    )
                    // Drop param_history and rescale transform to full-size
                    .map(|(mapping, _)| {
                        mapping
                        // .transform(
                        //     Some(Mapping::scale(lk_args.downscale, lk_args.downscale)),
                        //     Some(Mapping::scale(
                        //         1.0 / lk_args.downscale,
                        //         1.0 / lk_args.downscale,
                        //     )),
                        // )
                    })
                })
                .collect::<Result<Vec<_>>>()?
                // Accumulate all pairwise warps
                // TODO: maybe impl Copy to minimize the clones here...
                .iter()
                .scan(Mapping::identity(), |acc, x| {
                    *acc = acc.transform(None, Some(x.clone()));
                    Some(acc.clone())
                })
                .collect();
            // Add in an identity warp to the start to have one warp per frame
            // TODO: This is slow, maybe chain-in an I warp or use BTree?
            mappings.insert(0, Mapping::identity());

            // Find middle warp and undo it
            let wrt_map = Mapping::interpolate_scalar(
                Array::linspace(0.0, 1.0, virtual_exposures.len()),
                &mappings,
                wrt_normalized_idx,
            );
            let mappings = mappings
                .iter()
                .map(|m| m.transform(Some(wrt_map.inverse()), None))
                .collect();

            // Validate all shapes are equal (re-write with dedup?)
            let size: (u32, u32) = sizes
                .iter()
                .tuple_windows()
                .all(|(a, b)| a == b)
                .then(|| sizes[0])
                .unwrap();
            let (extent, offset) = Mapping::total_extent(&mappings, size);
            // TODO: Unpack better??
            let [canvas_w, canvas_h] = extent.to_vec()[..] else {
                panic!()
            };
            println!("{:}, {:}, {:?}", &canvas_w, &canvas_h, &offset);

            for (i, map) in mappings.iter().step_by(args.viz_step).enumerate() {
                // let corners = map.transform(None, Some(offset.clone())).corners(size);
                // let dst = polygon_sdf_vec(array![
                //     [corners[0].0, corners[0].1],
                //     [corners[1].0, corners[1].1],
                //     [corners[2].0, corners[2].1],
                //     [corners[3].0, corners[3].1],
                // ]);
                // // let weights = ImageBuffer::<Luma<f32>, Vec<f32>>::from_fn(
                // //     canvas_w.ceil() as u32,
                // //     canvas_h.ceil() as u32,
                // //     |x, y| {
                // //         Luma([dst(x as f32, y as f32)])
                // //     }
                // // );

                // let weights = Array::from_shape_fn(
                //     (canvas_w.ceil() as usize, canvas_h.ceil() as usize),
                //     |(x, y)| dst(x as f32, y as f32),
                // );
                // let min = weights.fold(f32::INFINITY, |a, b| a.min(*b));
                // let max = weights.fold(-f32::INFINITY, |a, b| a.max(*b));

                // let img = array2grayimage::<u8>(
                //     weights.mapv(|v| ((v - min) / (max - min) * 255.0).round() as u8),
                // );

                let mut img = ImageBuffer::new(canvas_w.ceil() as u32, canvas_h.ceil() as u32);
                let get_pixel =
                    |x, y| interpolate_bilinear_with_bkg(&virtual_exposures[i], x, y, Luma([128]));
                warp(
                    &mut img,
                    map.transform(None, Some(offset.clone())).warpfn(),
                    get_pixel,
                );

                let path = Path::new(&"tmp/".to_string()).join(format!("frame{:06}.png", i));
                img.save(&path)
                    .unwrap_or_else(|_| panic!("Could not save frame at {}!", &path.display()));
            }

            make_video(
                Path::new(&"tmp/".to_string())
                    .join("frame%06d.png")
                    .to_str()
                    .unwrap(),
                "out.mp4",
                25u64,
                0,
                None,
            );

            // let raw_pano = ImageBuffer::<Luma<f32>, Vec<f32>>::new(canvas_w.ceil() as u32, canvas_h.ceil() as u32);
            // let weights = ImageBuffer::<Luma<f32>, Vec<f32>>::new(canvas_w.ceil() as u32, canvas_h.ceil() as u32);
            // let get_pixel = |x, y| interpolate_bilinear(&img2, x, y).unwrap_or(Rgb([128, 0, 0]));
            // warp(&mut out, mapping.warpfn(), get_pixel);
            // out.save(global_args.output.unwrap_or("out.png".to_string()))?;
            Ok(())
        }
    }
}
