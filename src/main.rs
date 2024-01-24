#![allow(dead_code)] // Todo: Remove
#![allow(unused_imports)]

use anyhow::{anyhow, Result};
use image::imageops::{grayscale, resize, FilterType};
use image::{io::Reader as ImageReader, ImageBuffer, Rgb};
use image::{GrayImage, Luma};
use indicatif::{ProgressIterator, ProgressStyle};
use itertools::Itertools;
use ndarray::{array, s, Array, Array3, Axis, Slice};
use nshare::ToNdarray3;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use std::fs::create_dir_all;
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

use cli::{Cli, Commands, LKArgs, Parser};
use ffmpeg::make_video;
use io::PhotonCube;
use lk::{gradients, hierarchical_iclk, iclk, iclk_grayscale};
use transforms::{array3_to_image, process_colorspad, unpack_single};
use utils::{animate_hierarchical_warp, animate_warp};
use warps::{warp_array3, warp_image, Mapping, TransformationType};

use crate::blend::distance_transform;
use crate::transforms::{apply_transform, array2_to_grayimage};

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

    let out = warp_image(
        &mapping,
        &img2,
        (h as usize, w as usize),
        Some(Rgb([128, 0, 0])),
    );
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
        Some(Commands::Warp(_)) => {
            let map = Mapping::from_matrix(
                array![
                    [0.47654548, -0.045553986, 4.847797],
                    [-0.14852144, 0.6426208, 2.1364543],
                    [-0.009891294, -0.0021317923, 0.88151735]
                ],
                TransformationType::Projective,
            )
            .rescale(1.0 / 16.0);

            let img = ImageReader::open("madison2.png")
                .unwrap()
                .decode()
                .unwrap()
                .into_rgb8();
            let (w, h) = img.dimensions();

            // Warp with warp_image: Slower but no jaggies
            let out = warp_image(&map, &img, (h as usize, w as usize), Some(Rgb([128, 0, 0])));
            out.save("out1.png")?;

            // Warp with warp_array3: hopefully faster...
            // Note: We cannot use `img.into_ndarray3()` as it results in CHW array
            let data = Array3::from_shape_vec((h as usize, w as usize, 3), img.into_raw())
                .unwrap()
                .mapv(|v| v as f32);
            let (out, _) = warp_array3(
                &map,
                &data,
                (h as usize, w as usize),
                Some(array![128.0, 0.0, 0.0]),
            );
            let out = array3_to_image::<Rgb<u8>>(out.mapv(|v| v as u8));
            out.save("out2.png")?;
            Ok(())
        }
        Some(Commands::Pano(lk_args)) => {
            let cube = PhotonCube::load("../photoncube2video/data/binary.npy")?;
            let cube_view = cube.view()?;
            let cube_view = cube_view.slice_axis(Axis(0), Slice::new(0, Some(256 * 250), 1));
            let wrt_normalized_idx = 0.5;
            let burst_size = 256;

            // Create parallel iterator over all chunks of frames and process them
            let (virtual_exposures, mut sizes): (Vec<_>, Vec<_>) = cube_view
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

                    // Convert to image and resize/transform
                    let img = array2_to_grayimage(process_colorspad(frame));
                    let img = resize(
                        &img,
                        (img.width() as f32 / lk_args.downscale).round() as u32,
                        (img.height() as f32 / lk_args.downscale).round() as u32,
                        FilterType::CatmullRom,
                    );
                    let img = apply_transform(img, &args.transform);
                    let (w, h) = (img.width() as usize, img.height() as usize);
                    (img, (w, h))
                })
                // Force iterator to run to completion to get correct ordering
                .unzip();

            // Estimate pairwise registration
            let mut mappings: Vec<Mapping> = virtual_exposures
                .iter()
                .progress()
                .with_style(ProgressStyle::with_template(
                    "{msg} ETA:{eta}, [{elapsed_precise}] {wide_bar:.cyan/blue} {pos:>6}/{len:6}",
                )?)
                .with_message("Lvl 1:".to_string())
                .tuple_windows()
                .map(|(src, dst)| {
                    iclk_grayscale(
                        src,                             // im1_gray
                        dst,                             // im2_gray
                        gradients(dst),                  // im2_grad,
                        Mapping::from_params(&[0.0; 2]), // init_mapping
                        Some(lk_args.iterations),        // max_iters
                        Some(lk_args.early_stop),        // stop_early
                        None,                            // message
                    )
                    // Drop param_history and rescale transform to full-size
                    .map(|(mapping, _)| {
                        mapping
                        // .rescale(1.0 / lk_args.downscale)
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
            let mappings: Vec<Mapping> = mappings
                .iter()
                .map(|m| m.transform(Some(wrt_map.inverse()), None))
                .collect();

            // Validate all shapes are equal
            sizes = sizes.into_iter().unique().collect();
            let [size]: [(usize, usize)] = sizes[..] else {
                return Err(anyhow!(
                    "Expected all frames to have same shapes but found shapes: {sizes:#?}"
                ));
            };
            let (extent, offset) = Mapping::maximum_extent(&mappings[..], size);
            let [canvas_w, canvas_h] = extent.to_vec()[..] else {
                unreachable!("Canvas should have width and height")
            };
            println!("{:}, {:}, {:?}", &canvas_w, &canvas_h, &offset);

            for (i, map) in mappings.iter().step_by(args.viz_step).enumerate() {
                let img = warp_image(
                    &map.transform(None, Some(offset.clone())),
                    &virtual_exposures[i],
                    (canvas_h.ceil() as usize, canvas_w.ceil() as usize),
                    Some(Luma([128])),
                );
                // let img = polygon_distance_transform(
                //     &map.corners(size), (canvas_h.ceil() as usize, canvas_w.ceil() as usize)
                // ).mapv(|v| (v*255.0).round() as u8);
                // let img = array2grayimage(img);

                let path = Path::new(&"tmp/".to_string()).join(format!("frame{:06}.png", i));
                img.save(&path)
                    .unwrap_or_else(|_| panic!("Could not save frame at {}!", &path.display()));
            }

            make_video(
                Path::new(&"tmp/".to_string())
                    .join("frame%06d.png")
                    .to_str()
                    .unwrap(),
                &args.viz_output.unwrap_or("out.mp4".to_string()),
                25u64,
                0,
                None,
            );

            // let canvas = Array3::<f32>::zeros((canvas_h.ceil() as usize, canvas_w.ceil() as usize, 2));
            // let weights = distance_transform((canvas_h.ceil() as usize, canvas_w.ceil() as usize));

            // let raw_pano = ImageBuffer::<Luma<f32>, Vec<f32>>::new(canvas_w.ceil() as u32, canvas_h.ceil() as u32);
            // let weights = ImageBuffer::<Luma<f32>, Vec<f32>>::new(canvas_w.ceil() as u32, canvas_h.ceil() as u32);
            // let get_pixel = |x, y| interpolate_bilinear(&img2, x, y).unwrap_or(Rgb([128, 0, 0]));
            // warp(&mut out, mapping.warpfn(), get_pixel);
            // out.save(global_args.output.unwrap_or("out.png".to_string()))?;
            Ok(())
        }
    }
}
