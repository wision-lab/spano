use std::{
    env,
    fs::{create_dir_all, write},
    path::Path,
};

use anyhow::{anyhow, Result};
use image::{
    imageops::{resize, FilterType},
    io::Reader as ImageReader,
    GrayImage, Rgb,
};
use indicatif::{ParallelProgressIterator, ProgressStyle};
use ndarray::{Array1, Axis, Slice};
use photoncube2video::{
    cube::PhotonCube,
    signals::DeferredSignal,
    transforms::{
        apply_transforms, array2_to_grayimage, interpolate_where_mask, process_colorspad,
        unpack_single,
    },
};
use pyo3::prelude::*;
use rayon::{
    iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator},
    slice::ParallelSlice,
};
use tempfile::tempdir;

use crate::{
    blend::merge_images,
    cli::{Cli, Commands, LKArgs, Parser},
    lk::{hierarchical_iclk, iclk, pairwise_iclk},
    utils::{animate_hierarchical_warp, animate_warp, stabilized_video},
    warps::Mapping,
};

fn match_imgpair(global_args: Cli, lk_args: LKArgs) -> Result<()> {
    let [img1_path, img2_path, ..] = &global_args.input[..] else {
        return Err(anyhow!("Exactly two inputs are required for --input."));
    };

    // Load images and mask if needed
    let img1 = ImageReader::open(img1_path)?.decode()?.into_rgb8();
    let img2 = ImageReader::open(img2_path)?.decode()?.into_rgb8();
    let (w1, h1) = img1.dimensions();
    let (w2, h2) = img2.dimensions();

    let weights = if let Some(path) = lk_args.weights {
        let weights = ImageReader::open(path)?.decode()?.into_luma8();
        let (weights_w, weights_h) = weights.dimensions();

        if (h1 != weights_h) || (w1 != weights_w) {
            return Err(anyhow!("Mask and reference image need to be of same size."));
        }

        let weights = resize(
            &weights,
            (w1 as f32 / lk_args.downscale).round() as u32,
            (h1 as f32 / lk_args.downscale).round() as u32,
            FilterType::CatmullRom,
        );

        Some(weights)
    } else {
        None
    };

    // Resize all to maximum dimensions, as defined by `downscale`.
    let img1 = resize(
        &img1,
        (w1 as f32 / lk_args.downscale).round() as u32,
        (h1 as f32 / lk_args.downscale).round() as u32,
        FilterType::CatmullRom,
    );
    let img2 = resize(
        &img2,
        (w2 as f32 / lk_args.downscale).round() as u32,
        (h2 as f32 / lk_args.downscale).round() as u32,
        FilterType::CatmullRom,
    );

    // Perform Matching
    let (mapping, params_history_str, num_steps) = if !lk_args.multi {
        // Register images
        let (mapping, params_history) = iclk(
            &img1,
            &img2,
            Mapping::from_params(vec![0.0; 8]),
            weights.as_ref(),
            Some(lk_args.iterations),
            Some(lk_args.early_stop),
            Some(lk_args.patience),
            Some("Matching..."),
        )?;
        let num_steps = params_history.len();
        let params_history_str = serde_json::to_string_pretty(&params_history)?;
        (mapping, params_history_str, num_steps - 1)
    } else {
        // Register images
        let (mapping, params_history) = hierarchical_iclk(
            &img1,
            &img2,
            Mapping::from_params(vec![0.0; 8]),
            weights.as_ref(),
            Some(lk_args.iterations),
            (lk_args.min_size, lk_args.min_size),
            lk_args.max_lvls,
            Some(lk_args.early_stop),
            Some(lk_args.patience),
            true,
        )?;
        let num_steps = params_history.values().map(|v| v.len()).sum();
        let params_history_str = serde_json::to_string_pretty(&params_history)?;
        (mapping, params_history_str, num_steps)
    };

    println!(
        "Found following mapping in {:} steps:\n{:6.4}",
        num_steps - 1,
        &mapping.rescale(1.0 / lk_args.downscale).mat
    );
    if let Some(out_path) = global_args.output {
        let out = mapping.warp_image(
            &img2,
            (
                (h1 as f32 / lk_args.downscale).round() as usize,
                (w1 as f32 / lk_args.downscale).round() as usize,
            ),
            Some(Rgb([128, 0, 0])),
        );
        out.save(&out_path)?;
        println!("Saving warped image to {out_path}...");
    }
    if let Some(viz_path) = global_args.viz_output {
        println!("Saving animation to {viz_path}...");

        if !lk_args.multi {
            let params_history = serde_json::from_str(&params_history_str)?;
            animate_warp(
                img2_path,
                params_history,
                &global_args.img_dir.unwrap(),
                lk_args.downscale,
                Some(global_args.viz_fps),
                Some(global_args.viz_step),
                Some(&viz_path),
                Some("Making Video..."),
            )?;
        } else {
            let params_history = serde_json::from_str(&params_history_str)?;
            animate_hierarchical_warp(
                img2_path,
                params_history,
                lk_args.downscale,
                &global_args.img_dir.unwrap(),
                Some(global_args.viz_fps),
                Some(global_args.viz_step),
                Some(&viz_path),
                Some("Making Video..."),
            )?;
        }
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
    let _defer = DeferredSignal::new(py, "SIGINT")?;

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
            // Validate CLI args
            let [cube_path, ..] = &args.input[..] else {
                return Err(anyhow!(
                    "Only one input is required for --input when forming Pano."
                ));
            };
            if pano_args.burst_size <= pano_args.granularity {
                return Err(anyhow!(
                    "Argument `granularity` must be smaller than `burst-size`."
                ));
            }
            if pano_args.burst_size % pano_args.granularity != 0 {
                return Err(anyhow!(
                    "Argument `granularity` must evenly divide `burst-size`."
                ));
            }
            let granulars_per_burst = pano_args.burst_size / pano_args.granularity;

            // Load and pre-process chunks of frames from photoncube
            // We unpack the bitplanes, avergae them in groups of `burst_size`,
            // Apply color-spad corrections, and optionally downscale.
            // Any transforms (i.e: flipud) can be applied here too.
            let mut cube = PhotonCube::open(cube_path)?;
            if let Some(cfa_path) = &pano_args.cfa_path {
                cube.load_cfa(cfa_path.to_path_buf())?;
            }
            for inpaint_path in pano_args.inpaint_path.iter() {
                cube.load_mask(inpaint_path.to_path_buf())?;
            }

            let view = cube.view()?;
            let slice = view.slice_axis(
                Axis(0),
                Slice::new(
                    pano_args.start.unwrap_or(0) as isize,
                    pano_args.end.map(|v| v.min(view.len_of(Axis(0)) as isize)),
                    1,
                ),
            );

            let (_, h, w) = view.dim();
            let w = w * 8;
            let (lvls_h, lvls_w) = (
                f32::log2(
                    h as f32 / pano_args.lk_args.downscale / pano_args.lk_args.min_size as f32,
                )
                .ceil(),
                f32::log2(
                    w as f32 / pano_args.lk_args.downscale / pano_args.lk_args.min_size as f32,
                )
                .ceil(),
            );
            let num_lvls = (lvls_h as u32)
                .min(lvls_w as u32)
                .min(pano_args.lk_args.max_lvls);

            let num_ves = (slice.len_of(Axis(0)) / pano_args.burst_size) / pano_args.step;
            let num_frames_per_chunk = pano_args.burst_size / pano_args.granularity;
            let mut mappings: Vec<Mapping> = vec![Mapping::from_params(vec![0.0; 2]); num_ves - 1];
            let mut virtual_exposures;

            // Preload all data at given granularity
            let granular_frames: Vec<GrayImage> = slice.axis_chunks_iter(Axis(0), pano_args.granularity)
                .into_par_iter()
                .progress()
                .with_style(
                    ProgressStyle::with_template(
                        "{msg} ETA:{eta}, [{elapsed_precise}] {wide_bar:.cyan/blue} {pos:>6}/{len:6}",
                    )
                    .expect("Invalid progress style."),
                )
                .with_message("Preloading Data...")
                .map(|group| {
                    // Iterate over all bitplanes in group,
                    // Unpack every frame in group as a f32 array
                    // Apply any corrections and blend them together
                    let mut frame = group
                        .axis_iter(Axis(0))
                        .map(|bitplane| {
                            if pano_args.not_bitpacked {
                                bitplane.mapv(|v| v as f32)
                            } else {
                                unpack_single::<f32>(&bitplane, 1).unwrap()
                            }
                        })
                        // Sum frames together (.sum not implemented for this type)
                        .reduce(|acc, e| acc + e)
                        .unwrap();

                    // Compute mean values
                    frame.mapv_inplace(|v| v / (pano_args.granularity as f32));

                    // Apply any frame-level fixes (only for ColorSPAD at the moment)
                    if pano_args.colorspad_fix {
                        frame = process_colorspad(frame);
                    }

                    // Demosaic frame by interpolating white pixels
                    if let Some(mask) = &cube.cfa_mask {
                        frame = interpolate_where_mask(&frame, mask, false).unwrap();
                    }

                    // Inpaint any hot/dead pixels
                    if let Some(mask) = &cube.inpaint_mask {
                        frame = interpolate_where_mask(&frame, mask, false).unwrap();
                    }

                    // Convert to img and apply transforms
                    let mut img = array2_to_grayimage(
                        frame.mapv(|v| (v * 255.0) as u8),
                    );

                    if pano_args.lk_args.downscale != 1.0 {
                        img = resize(
                            &img,
                            (w as f32 / pano_args.lk_args.downscale).round() as u32,
                            (h as f32 / pano_args.lk_args.downscale).round() as u32,
                            FilterType::CatmullRom,
                        );
                    }

                    apply_transforms(img, &args.transform[..])
                })
                .collect();
            let (w, h) = granular_frames[0].dimensions();

            // -------------------- Main hierarchical matching process ----------------------------
            for lvl in (0..num_lvls).rev() {
                // Interpolate mappings to all bitplanes
                mappings = mappings.iter().map(|m| m.rescale(0.5)).collect();
                let acc_maps = Mapping::accumulate(mappings.clone());
                let interpd_maps = Mapping::interpolate_array(
                    Array1::linspace(0.0, (num_ves - 1) as f32, num_ves).to_vec(),
                    acc_maps,
                    Array1::linspace(0.0, (num_ves - 1) as f32, num_ves * num_frames_per_chunk)
                        .to_vec(),
                );
                let downscale = 1 << lvl;

                // Compute virtual exposure by merging `num_frames_per_chunk` granular frames, and downscaling result
                virtual_exposures = (
                    granular_frames.par_chunks(num_frames_per_chunk).step_by(pano_args.step),
                    interpd_maps.par_chunks(num_frames_per_chunk),
                )
                    .into_par_iter()
                    .progress()
                    .with_style(
                        ProgressStyle::with_template(
                            "{msg} ETA:{eta}, [{elapsed_precise}] {wide_bar:.cyan/blue} {pos:>6}/{len:6}",
                        )
                        .expect("Invalid progress style."),
                    )
                    .with_message(format!("({}/{}): Loading Data...", num_lvls - lvl, num_lvls))
                    .map(|(frames, maps)| {
                        let img = merge_images(
                            &Mapping::with_respect_to_idx(maps.to_vec(), 0.5),
                            frames,
                            Some((w as usize, h as usize)),
                        ).unwrap();

                        resize(
                            &img,
                            (w as f32 / downscale as f32).round() as u32,
                            (h as f32 / downscale as f32).round() as u32,
                            FilterType::CatmullRom,
                        )
                    })
                    .collect();

                // Estimate pairwise registration
                mappings = pairwise_iclk(
                    &virtual_exposures,
                    &mappings[..],
                    pano_args.lk_args.iterations,
                    pano_args.lk_args.early_stop,
                    pano_args.lk_args.patience,
                    Some(format!("({}/{}): Matching...", num_lvls - lvl, num_lvls).as_str()),
                )?;

                // If it's the first iteration, save a baseline pano using the granular frames
                if lvl == num_lvls - 1 {
                    // Accumulate wrt center frame
                    let acc_maps = Mapping::accumulate_wrt_idx(mappings.clone(), pano_args.wrt);


                    // Scale back to original size
                    let scaled_mappings: Vec<_> = acc_maps
                        .iter()
                        .map(|m| m.rescale(1.0 / (downscale as f32)))
                        .collect();

                    // Interpolate from all virtual exposures to all granular frames (if step != 1 these are not equal)
                    let interpd_maps = Mapping::interpolate_array(
                        Array1::linspace(0.0, (num_ves - 1) as f32, num_ves).to_vec(),
                        scaled_mappings,
                        Array1::linspace(0.0, (num_ves - 1) as f32, granular_frames.len()).to_vec(),
                    );
                    
                    // Repeat mapping such that it is constant for the duration of a burst frame
                    let interpd_maps: Vec<_> = interpd_maps
                        .into_iter()
                        .step_by(granulars_per_burst)
                        .flat_map(|n| std::iter::repeat(n).take(granulars_per_burst))
                        .collect();

                    // Create baseline pano and save
                    let canvas = merge_images(&interpd_maps, &granular_frames, None)?;
                    canvas.save("baseline.png")?;
                }

                // Augment mapping type every iteration
                mappings = mappings.iter().map(|m| m.upgrade()).collect();

                if let Some(viz_path) = args.viz_output.clone() {
                    let parent = Path::new(&viz_path)
                        .parent()
                        .expect("Viz output path should have a parent directory");
                    let file = format!("lvl-{}.mp4", num_lvls - lvl);
                    let path = parent.join(file);
                    stabilized_video(
                        &Mapping::accumulate_wrt_idx(mappings.clone(), pano_args.wrt),
                        &virtual_exposures,
                        None,
                        Some(args.viz_fps),
                        Some(args.viz_step),
                        Some(
                            path.to_str()
                                .expect("Cannot join output visualization paths"),
                        ),
                        Some(
                            format!("({}/{}): Creating Preview...", num_lvls - lvl, num_lvls)
                                .as_str(),
                        ),
                    )?;
                }
            }
            // ----------------------------------------------------------------------------------

            // Interpolate mapping to every granular frame
            let acc_maps = Mapping::accumulate_wrt_idx(mappings.clone(), pano_args.wrt);
            let interpd_maps = Mapping::interpolate_array(
                Array1::linspace(0.0, (num_ves - 1) as f32, num_ves).to_vec(),
                acc_maps,
                Array1::linspace(0.0, (num_ves - 1) as f32, granular_frames.len()).to_vec(),
            );

            let canvas = merge_images(&interpd_maps, &granular_frames, None)?;

            canvas.save(&args.output.unwrap_or("out.png".to_string()))?;
            Ok(())
        }
    }
}
