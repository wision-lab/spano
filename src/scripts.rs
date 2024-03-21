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
use ndarray::{s, Array1, Axis, NewAxis, Slice};
use photoncube2video::{
    cube::PhotonCube,
    signals::DeferedSignal,
    transforms::{
        apply_transforms, array2_to_grayimage, interpolate_where_mask, process_colorspad,
        unpack_single,
    },
};
use rayon::{iter::{IntoParallelIterator, ParallelIterator}, slice::ParallelSlice};
use pyo3::prelude::*;
use tempfile::tempdir;

use crate::{
    blend::{merge_arrays, merge_images},
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
            Some("Matching..."),
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
        Commands::LK(lk_args) => match_imgpair(args.clone(), lk_args.clone()),
        Commands::Pano(pano_args) => {
            let [cube_path, ..] = &args.input[..] else {
                return Err(anyhow!(
                    "Only one input is required for --input when forming Pano."
                ));
            };

            // Load and pre-process chunks of frames from photoncube
            // We unpack the bitplanes, avergae them in groups of `burst_size`,
            // Apply color-spad corrections, and optionally downscale.
            // Any transforms (i.e: flipud) can be applied here too.
            let mut cube = PhotonCube::open(cube_path)?;
            if let Some(cfa_path) = &pano_args.cfa_path {
                cube.load_cfa(&cfa_path)?;
            }
            for inpaint_path in pano_args.inpaint_path.iter() {
                cube.load_mask(inpaint_path)?;
            }

            let view = cube.view()?;
            let slice = view.slice_axis(
                Axis(0),
                Slice::new(pano_args.start.unwrap_or(0), pano_args.end, 1),
            );

            let (_, h, w) = view.dim();
            let (lvls_h, lvls_w) = (
                f32::log2(
                    h as f32
                        / pano_args.lk_args.downscale as f32
                        / pano_args.lk_args.min_size as f32,
                )
                .ceil(),
                f32::log2(
                    w as f32
                        / pano_args.lk_args.downscale as f32
                        / pano_args.lk_args.min_size as f32
                        * 8.0,
                )
                .ceil(),
            );
            let num_lvls = (lvls_h as u32)
                .min(lvls_w as u32)
                .min(pano_args.lk_args.max_lvls);
            println!("{h}, {w}, {num_lvls}");

            let num_ves = view.len_of(Axis(0)) / pano_args.burst_size;
            let mut mappings: Vec<Mapping> = vec![Mapping::from_params(vec![0.0; 2]); num_ves - 1];
            let mut virtual_exposures: Vec<_> = vec![];

            for lvl in (1..=num_lvls).rev() {
                // Interpolate mappings to all bitplanes
                let acc_maps = Mapping::accumulate(mappings.clone());
                let interpd_maps = Mapping::interpolate_array(
                    Array1::linspace(0.0, 1.0, num_ves).to_vec(),
                    acc_maps,
                    Array1::linspace(0.0, 1.0, (num_ves-1) * pano_args.burst_size).to_vec(),
                );

                let downscale = 2 << lvl;

                virtual_exposures = (
                    slice.axis_chunks_iter(Axis(0), pano_args.burst_size),
                    interpd_maps.par_chunks(pano_args.burst_size)
                )
                    .into_par_iter()
                    .map(|(group, maps)| {
                        let frame = {
                            // Iterate over all bitplanes in group,
                            // Unpack every frame in group as a f32 array
                            // Apply any corrections and blend them together
                            let frames: Vec<_> = group
                                .axis_iter(Axis(0))
                                .map(|bitplane| {
                                    let mut frame = unpack_single::<f32>(&bitplane, 1).unwrap();

                                    // Apply any frame-level fixes (only for ColorSPAD at the moment)
                                    if pano_args.colorspad_fix {
                                        frame = process_colorspad(frame);
                                    }

                                    // Demosaic frame by interpolating white pixels
                                    if let Some(mask) = &cube.cfa_mask {
                                        frame = interpolate_where_mask(&frame, mask, true).unwrap();
                                    }

                                    // Inpaint any hot/dead pixels
                                    if let Some(mask) = &cube.inpaint_mask {
                                        frame = interpolate_where_mask(&frame, mask, true).unwrap();
                                    }

                                    frame.slice(s![.., .., NewAxis]).to_owned()
                                })
                                .collect();
                            merge_arrays(
                                &Mapping::with_respect_to_idx(maps.to_vec(), 0.5),
                                &frames[..],
                                Some((h, w)),
                            )
                            .unwrap()
                        };

                        let img = resize(
                            &array2_to_grayimage(
                                frame.slice(s![.., .., 0]).mapv(|v| (v * 255.0) as u8),
                            ),
                            (w as f32 / downscale as f32).round() as u32,
                            (h as f32 / downscale as f32).round() as u32,
                            FilterType::CatmullRom,
                        );

                        apply_transforms(img, &args.transform[..])
                    })
                    .collect();

                // Estimate pairwise registration
                mappings = pairwise_iclk(
                    &virtual_exposures,
                    &mappings[..],
                    pano_args.lk_args.iterations,
                    pano_args.lk_args.early_stop,
                    pano_args.lk_args.patience,
                    Some(format!("Level {}/{}:", num_lvls - lvl + 1, num_lvls).as_str()),
                )?;

                if args.viz_output.is_some() {
                    stabilized_video(
                        &Mapping::accumulate_wrt_idx(mappings.clone(), pano_args.wrt),
                        &virtual_exposures,
                        "tmp/",
                        Some(args.viz_fps),
                        Some(args.viz_step),
                        Some(format!("lvl-{}.mp4", num_lvls - lvl + 1).as_str()),
                    )?;
                }

                if lvl > 1 {
                    mappings = mappings.iter().map(|m| m.rescale(0.5)).collect();
                }
            }
            // ----------------------------------------------------------------------------------

            let canvas = merge_images(
                &Mapping::accumulate_wrt_idx(mappings.clone(), pano_args.wrt),
                &virtual_exposures,
                None,
            )?;

            canvas.save(&args.output.unwrap_or("out.png".to_string()))?;
            Ok(())
        }
    }
}
