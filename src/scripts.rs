use std::{
    env,
    fs::{create_dir_all, write},
};

use anyhow::{anyhow, Result};
use conv::ValueInto;
use image::{
    imageops::{grayscale, resize, FilterType},
    EncodableLayout, ImageReader, Luma, Pixel, PixelWithColorType, Rgb,
};
use imageproc::definitions::{Clamp, Image};
use ndarray::{Array1, Axis, Slice};
use photoncube::{
    cube::PhotonCube,
    signals::DeferredSignal,
    transforms::{array3_to_image, image_to_array3, ref_image_to_array3},
};
use pyo3::prelude::*;

use crate::{
    blend::merge_images,
    cli::{Cli, Commands, MatchingArgs, PanoArgs, Parser},
    lk::iclk,
    pano::{panorama, visualization_callback},
    utils::animate_warp,
    warps::Mapping,
};

fn match_imgpair(global_args: &Cli, lk_args: &MatchingArgs) -> Result<()> {
    let [img1_path, img2_path, ..] = &global_args.input[..] else {
        return Err(anyhow!("Exactly two inputs are required for --input."));
    };

    // Load images and mask if needed
    let img1 = ImageReader::open(img1_path)?.decode()?.into_rgb8();
    let img2 = ImageReader::open(img2_path)?.decode()?.into_rgb8();
    let (w1, h1) = img1.dimensions();
    let (w2, h2) = img2.dimensions();

    let weights = if let Some(path) = &lk_args.weights {
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

    // Conditionally convert images to grayscale, then register images
    let (mapping, params_history) = if lk_args.grayscale {
        let img1 = grayscale(&img1);
        let img2 = grayscale(&img2);
        iclk(
            &img1,
            &img2,
            Mapping::from_params(vec![0.0; 8]),
            weights.as_ref(),
            lk_args.multi,
            Some(lk_args.iterations),
            Some(lk_args.min_size),
            Some(lk_args.max_lvls),
            Some(lk_args.early_stop),
            Some(lk_args.patience),
            true,
        )?
    } else {
        iclk(
            &img1,
            &img2,
            Mapping::from_params(vec![0.0; 8]),
            weights.as_ref(),
            lk_args.multi,
            Some(lk_args.iterations),
            Some(lk_args.min_size),
            Some(lk_args.max_lvls),
            Some(lk_args.early_stop),
            Some(lk_args.patience),
            true,
        )?
    };
    let num_steps: usize = params_history.values().map(|v| v.len()).sum();

    println!(
        "Found following mapping in {:} steps:\n{:6.4}",
        num_steps - 1,
        &mapping.rescale(1.0 / lk_args.downscale).mat
    );
    if let Some(out_path) = &global_args.output {
        let out = mapping.warp_image(
            &img2,
            (
                (h1 as f32 / lk_args.downscale).round() as usize,
                (w1 as f32 / lk_args.downscale).round() as usize,
            ),
            Some(Rgb([128, 0, 0])),
        );
        out.save(&out_path)?;
        println!("Saving warped image to {out_path:?}...");
    }
    if let Some(viz_path) = &global_args.viz_output {
        println!("Saving animation to {viz_path:?}...");
        animate_warp(
            &img2,
            params_history.clone(),
            global_args.img_dir.clone(),
            lk_args.downscale,
            Some(global_args.viz_fps),
            Some(global_args.viz_step),
            Some(viz_path.to_path_buf()),
            Some("Making Video..."),
        )?;
    }
    if let Some(params_path) = &lk_args.params_path {
        let params_history_str = serde_json::to_string_pretty(&params_history)?;
        write(params_path, params_history_str).expect("Unable to write params file.");
    }
    Ok(())
}

fn make_panorama(global_args: &Cli, pano_args: &PanoArgs) -> Result<()> {
    // Validate CLI args
    let [cube_path, ..] = &global_args.input[..] else {
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

    let mut cube = PhotonCube::open(cube_path)?;
    if let Some(cfa_path) = &pano_args.cfa_path {
        cube.load_cfa(cfa_path.to_path_buf())?;
    }
    for inpaint_path in pano_args.inpaint_path.iter() {
        cube.load_mask(inpaint_path.to_path_buf())?;
    }

    if cube.view()?.ndim() == 3 {
        _make_panorama::<Luma<u8>>(cube, &global_args, &pano_args)
    } else {
        _make_panorama::<Rgb<u8>>(cube, &global_args, &pano_args)
    }
}

fn _make_panorama<P>(cube: PhotonCube, global_args: &Cli, pano_args: &PanoArgs) -> Result<()>
where
    P: Pixel + PixelWithColorType + Send + Sync + 'static,
    <P as Pixel>::Subpixel:
        num_traits::Zero + Clone + Copy + ValueInto<f32> + Send + Sync + Clamp<f32>,
    [<P as Pixel>::Subpixel]: EncodableLayout,
    f32: From<<P as Pixel>::Subpixel>,
{
    let view = cube.view()?;
    let slice = view.slice_axis(
        Axis(0),
        Slice::new(
            pano_args.start.unwrap_or(0) as isize,
            pano_args.end.map(|v| v.min(view.len_of(Axis(0)) as isize)),
            1,
        ),
    );

    // Split up processing since we can't really invert the response or apply tonemapping
    // when only a handful of frames have been merged (eg granularity)
    let process_fn = Some(cube.process_frame(
        false,
        1.0,
        false,
        pano_args.colorspad_fix,
        pano_args.grayspad_fix,
        &cube.cfa_mask,
        &cube.inpaint_mask,
        false,
    )?);
    let post_process_fn = if pano_args.invert_response || pano_args.tonemap2srgb {
        Some(cube.process_frame(
            pano_args.invert_response,
            pano_args.factor,
            pano_args.tonemap2srgb,
            false,
            false,
            &None,
            &None,
            false,
        )?)
    } else {
        None
    };

    let post_process_im_fn = if let Some(ref post_process_fn) = post_process_fn {
        Some(|im: Image<P>| {
            array3_to_image::<P>(
                post_process_fn(
                    image_to_array3(im)
                        .into_dyn()
                        .mapv(|v| f32::from(v) / 255.0),
                )
                .unwrap()
                .into_dimensionality()
                .unwrap()
                .mapv(|v| P::Subpixel::clamp(v * 255.0)),
            )
        })
    } else {
        None
    };

    let callback_fn = if let Some(viz_path) = &global_args.viz_output {
        Some(visualization_callback(
            viz_path,
            global_args.viz_fps,
            global_args.viz_step,
            pano_args.wrt,
            &post_process_im_fn,
        )?)
    } else {
        None
    };

    let (all_mappings, virtual_exposures, granular_frames) = panorama::<P>(
        slice,
        &pano_args.matching_args,
        pano_args.burst_size,
        pano_args.step,
        &global_args.transform,
        pano_args.granularity,
        !pano_args.not_bitpacked,
        &process_fn,
        &callback_fn,
    )?;

    let (w, h) = granular_frames[0].dimensions();
    let crop = if pano_args.crop {
        Some((h as usize, w as usize))
    } else {
        None
    };
    let num_frames_per_burst = pano_args.burst_size / pano_args.granularity;
    let num_ves = virtual_exposures.len();
    let num_lvls = all_mappings.len();

    // ----------------------------------------------------------------------------------

    // Save final panorama
    if let Some(outpath) = &global_args.output {
        // Interpolate mapping to every granular frame
        let acc_maps =
            Mapping::accumulate_wrt_idx(all_mappings[num_lvls - 1].clone(), pano_args.wrt);
        let interpd_maps = Mapping::interpolate_array(
            Array1::linspace(0.0, (num_ves - 1) as f32, num_ves).to_vec(),
            acc_maps,
            Array1::linspace(0.0, (num_ves - 1) as f32, granular_frames.len()).to_vec(),
        );
        let mut canvas = merge_images(
            &interpd_maps,
            &granular_frames,
            crop,
            Some("Making Panorama..."),
        )?;
        if let Some(post_process_im_fn) = post_process_im_fn {
            canvas = post_process_im_fn(canvas);
        }
        create_dir_all(outpath.parent().unwrap()).unwrap();
        canvas.save(outpath)?;
    }

    // ----------------------------------------------------------------------------------

    // Save naivesum image as baseline
    if let Some(naivesum_path) = &pano_args.naivesum_path {
        let merged = granular_frames
            .iter()
            .map(|f| ref_image_to_array3(f).mapv(f32::from))
            .reduce(|acc, e| acc + e)
            .unwrap();
        let merged = if let Some(ref post_process_fn) = post_process_fn {
            post_process_fn(
                merged
                    .mapv(|v| v / 255.0 / granular_frames.len() as f32)
                    .into_dyn(),
            )?
            .mapv(|v| v * 255.0)
            .into_dimensionality()?
        } else {
            merged.mapv(|v| v / granular_frames.len() as f32)
        };
        let canvas = array3_to_image::<P>(merged.mapv(<P as Pixel>::Subpixel::clamp));
        create_dir_all(naivesum_path.parent().unwrap()).unwrap();
        canvas.save(naivesum_path)?;
    }

    // Save a baseline pano using the first lvl maps
    if let Some(baseline_path) = &pano_args.baseline_path {
        // Accumulate wrt center frame
        let acc_maps = Mapping::accumulate_wrt_idx(all_mappings[0].clone(), pano_args.wrt);

        // Scale back to original size
        let scaled_mappings: Vec<_> = acc_maps
            .iter()
            .map(|m| m.rescale(1.0 / ((1 << (num_lvls - 1)) as f32)))
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
            .step_by(num_frames_per_burst)
            .flat_map(|n| std::iter::repeat_n(n, num_frames_per_burst))
            .collect();

        // Create baseline pano and save
        let mut canvas = merge_images(
            &interpd_maps,
            &granular_frames,
            crop,
            Some("Making Baseline Pano..."),
        )?;
        if let Some(post_process_im_fn) = post_process_im_fn {
            canvas = post_process_im_fn(canvas);
        }
        create_dir_all(baseline_path.parent().unwrap()).unwrap();
        canvas.save(baseline_path)?;
    }

    Ok(())
}

#[pyfunction]
pub fn cli_entrypoint(py: Python) -> Result<()> {
    // Start by telling python to not intercept CTRL+C signal,
    // Otherwise we won't get it here and will not be interruptible.
    // See: https://github.com/PyO3/pyo3/pull/3560
    let _defer = DeferredSignal::new(py, "SIGINT")?;

    // Parse arguments defined in struct
    // Since we're actually calling this via python, the first argument
    // is going to be the path to the python interpreter, so we skip it.
    // See: https://www.maturin.rs/bindings#both-binary-and-library
    let args = Cli::parse_from(env::args_os().skip(1));

    match &args.command {
        Commands::LK(lk_args) => match_imgpair(&args, &lk_args),
        Commands::Pano(pano_args) => make_panorama(&args, &pano_args),
    }
}
