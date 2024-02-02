use std::{
    env,
    fs::{create_dir_all, write},
};

use anyhow::{anyhow, Result};
use image::{
    imageops::{resize, FilterType},
    io::Reader as ImageReader,
    Rgb,
};
use indicatif::ProgressIterator;
use itertools::Itertools;
use ndarray::{concatenate, s, Array2, Array3, Axis, NewAxis, Slice};
use photoncube2video::{
    cube::{PhotonCube, VirtualExposure},
    signals::DeferedSignal,
    transforms::{apply_transforms, array2_to_grayimage, process_colorspad, ref_image_to_array3},
};
use pyo3::prelude::*;
use rayon::iter::ParallelIterator;
use tempfile::tempdir;

use crate::{
    blend::distance_transform,
    cli::{Cli, Commands, LKArgs, Parser},
    lk::{hierarchical_iclk, iclk, pairwise_iclk},
    utils::{animate_hierarchical_warp, animate_warp, stabilized_video},
    warps::Mapping,
};

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

    // Perform Matching
    let (mapping, params_history_str, num_steps) = if !lk_args.multi {
        // Register images
        let (mapping, params_history) = iclk(
            &img1,
            &img2,
            Mapping::from_params(vec![0.0; 8]),
            Some(lk_args.iterations),
            Some(lk_args.early_stop),
            Some(lk_args.patience),
        )?;
        let num_steps = params_history.len();

        // Show Animation of optimization
        let params_history_str = serde_json::to_string_pretty(&params_history)?;
        if global_args.viz_output.is_some() {
            animate_warp(
                img2_path,
                params_history,
                &global_args.img_dir.unwrap(),
                lk_args.downscale,
                Some(global_args.viz_fps),  // FPS
                Some(global_args.viz_step), // Step
                global_args.viz_output.as_deref(),
            )?;
        }
        (mapping, params_history_str, num_steps - 1)
    } else {
        // Register images
        let (mapping, params_history) = hierarchical_iclk(
            &img1,
            &img2,
            Mapping::from_params(vec![0.0; 8]),
            Some(lk_args.iterations),
            (25, 25),
            lk_args.max_lvls,
            Some(lk_args.early_stop),
            Some(lk_args.patience),
            true,
        )?;
        let num_steps = params_history.values().map(|v| v.len()).sum();

        // Show Animation of optimization
        let params_history_str = serde_json::to_string_pretty(&params_history)?;
        if global_args.viz_output.is_some() {
            animate_hierarchical_warp(
                img2_path,
                params_history,
                lk_args.downscale,
                &global_args.img_dir.unwrap(),
                Some(global_args.viz_fps),  // FPS
                Some(global_args.viz_step), // Step
                global_args.viz_output.as_deref(),
            )?;
        }
        (mapping, params_history_str, num_steps)
    };

    println!(
        "Found following mapping in {:} steps:\n{:6.4}",
        num_steps - 1,
        &mapping.rescale(1.0 / lk_args.downscale).mat
    );
    if let Some(viz_path) = global_args.viz_output {
        println!("Saving animation to {viz_path}...");
    }
    if let Some(out_path) = global_args.output {
        let out = mapping.warp_image(&img2, (h as usize, w as usize), Some(Rgb([128, 0, 0])));
        out.save(&out_path)?;
        println!("Saving warped image to {out_path}...");
    }
    if let Some(params_path) = lk_args.params_path {
        write(params_path, params_history_str).expect("Unable to write params file.");
    }
    Ok(())
}

#[pyfunction]
pub fn cli_entrypoint(py: Python) -> Result<()> {
    // Start by telling python to not intercept CTRL+C signal,
    // Otherwise we won't get it here and will not be interruptable.
    // See: https://github.com/PyO3/pyo3/pull/3560
    let _defer = DeferedSignal::new(py, "SIGINT")?;

    // Parse arguments defined in struct
    // Since we're actually calling this via python, the first argument
    // is going to be the path to the python interpreter, so we skip it.
    // See: https://www.maturin.rs/bindings#both-binary-and-library
    let mut args = Cli::parse_from(env::args_os().skip(1));

    // Get img path or tempdir, ensure it exists.
    let tmp_dir = tempdir()?;
    let img_dir = args
        .img_dir
        .clone()
        .unwrap_or(tmp_dir.path().to_str().unwrap().to_owned());
    create_dir_all(&img_dir).ok();
    args.img_dir = Some(img_dir);

    match &args.command {
        None => Err(anyhow!("Only `LK` subcommand is currently implemented.")),
        Some(Commands::LK(lk_args)) => match_imgpair(args.clone(), lk_args.clone()),
        Some(Commands::Pano(pano_args)) => {
            let [cube_path, ..] = &args.input[..] else {
                return Err(anyhow!(
                    "Only one input is required for --input when forming Pano."
                ));
            };

            // Load and pre-process chunks of frames from photoncube
            // We unpack the bitplanes, avergae them in groups of `burst_size`,
            // Apply color-spad corrections, and optionally downscale.
            // Any transforms (i.e: flipud) can be applied here too.

            let cube = PhotonCube::open(cube_path)?;
            let view = cube.view()?;
            let slice = view.slice_axis(
                Axis(0),
                Slice::new(
                    pano_args.start.unwrap_or(0),
                    pano_args.end.or(Some(256 * 250)),
                    1,
                ),
            );
            let virtual_exposures: Vec<_> = slice
                .par_virtual_exposures(pano_args.burst_size, true)
                .map(|frame| {
                    let frame = frame.mapv(|x| (x * 255.0) as u8);
                    let mut img = array2_to_grayimage(process_colorspad(frame));
                    if pano_args.lk_args.downscale != 1.0 {
                        img = resize(
                            &img,
                            (img.width() as f32 / pano_args.lk_args.downscale).round() as u32,
                            (img.height() as f32 / pano_args.lk_args.downscale).round() as u32,
                            FilterType::CatmullRom,
                        );
                    }

                    apply_transforms(img, &args.transform[..])
                })
                .collect();

            // Estimate pairwise registration
            let init_mappings = vec![Mapping::identity(); virtual_exposures.len() - 1];
            let mappings: Vec<Mapping> = pairwise_iclk(
                &virtual_exposures,
                &init_mappings[..],
                1.0,
                pano_args.lk_args.iterations,
                (25, 25),
                pano_args.lk_args.max_lvls,
                pano_args.lk_args.early_stop,
                pano_args.lk_args.patience,
                Some(pano_args.wrt),
                Some("Lvl 1:"),
            )?;

            stabilized_video(
                &mappings,
                &virtual_exposures,
                &args.img_dir.unwrap(),
                Some(args.viz_fps),
                Some(args.viz_step),
                args.viz_output.as_deref(),
            )?;

            // Make canvas for panorama
            let sizes: Vec<_> = virtual_exposures
                .iter()
                .map(|f| (f.width() as usize, f.height() as usize))
                .unique()
                .collect();
            let (extent, offset) = Mapping::maximum_extent(&mappings[..], &sizes[..]);
            let [canvas_w, canvas_h] = extent.to_vec()[..] else {
                unreachable!("Canvas should have width and height")
            };
            let (canvas_h, canvas_w) = (canvas_h.ceil() as usize, canvas_w.ceil() as usize);
            println!(
                "Made Canvas of size {:}x{:}, with offset {:?}",
                &canvas_w,
                &canvas_h,
                &offset.get_params()
            );
            let mut canvas: Array3<f32> = Array3::zeros((canvas_h, canvas_w, 2));
            let mut valid: Array2<bool> = Array2::from_elem((canvas_h, canvas_w), false);

            let (size, _) = Mapping::maximum_extent(&[Mapping::identity()], &sizes[..]);
            let weights = distance_transform(
                size.map(|v| *v as usize)
                    .into_iter()
                    .collect_tuple()
                    .unwrap(),
            );
            let weights = weights.slice(s![.., .., NewAxis]);
            let merge = |dst: &mut [f32], src: &[f32]| {
                dst[0] += src[0] * src[1];
                dst[1] += src[1];
            };

            for (frame, map) in virtual_exposures.iter().zip(mappings).progress() {
                let frame = ref_image_to_array3(frame).mapv(|v| v as f32);
                let frame = concatenate(Axis(2), &[frame.view(), weights.view()])?;
                map.warp_array3_into(
                    &frame.as_standard_layout(),
                    &mut canvas,
                    &mut valid,
                    None,
                    None,
                    Some(merge),
                );
            }

            array2_to_grayimage(
                (canvas.slice(s![.., .., 0]).to_owned() / canvas.slice(s![.., .., 1]))
                    .mapv(|v| v as u8),
            )
            .save(&args.output.unwrap_or("out.png".to_string()))?;
            Ok(())
        }
    }
}
