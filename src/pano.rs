use std::path::Path;

use anyhow::Result;
use conv::ValueInto;
use image::{
    imageops::{resize, FilterType},
    EncodableLayout, Pixel, PixelWithColorType,
};
use imageproc::definitions::{Clamp, Image};
use indicatif::{ParallelProgressIterator, ProgressStyle};
use ndarray::{Array1, ArrayD, Axis};
use photoncube::{
    cube::PhotonCubeView,
    transforms::{apply_transforms, array3_to_image, image_to_array3, unpack_single, Transform},
};
use rayon::{
    iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator},
    slice::ParallelSlice,
};

use crate::{
    blend::merge_images, cli::MatchingArgs, lk::pairwise_iclk, utils::stabilized_video,
    warps::Mapping,
};

/// Load and pre-process chunks of frames from photoncube
/// Optionally unpack the bitplanes, then average them in groups of `burst_size`.
/// Apply any SPAD fixes or corrections via the `process_fn`, and optionally downscale.
/// Any transforms (i.e: flip-ud) are applied here too.
pub fn make_granular_frames<P>(
    slice: PhotonCubeView,
    process_fn: &Option<impl Fn(ArrayD<f32>) -> Result<ArrayD<f32>> + Send + Sync>,
    transform: &[Transform],
    granularity: usize,
    is_bitpacked: bool,
    downscale: f32,
) -> Result<Vec<Image<P>>>
where
    P: Pixel + Send + Sync + 'static,
    <P as Pixel>::Subpixel: Send + Sync + Clamp<f32>,
{
    // Preload all data at given granularity
    let granular_frames = slice
        .axis_chunks_iter(Axis(0), granularity)
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
                    if is_bitpacked {
                        unpack_single::<f32>(&bitplane, 1).unwrap()
                    } else {
                        bitplane.mapv(|v| v as f32)
                    }
                })
                // Sum frames together (.sum not implemented for this type)
                .reduce(|acc, e| acc + e)
                .unwrap();

            // Compute mean values and preprocess
            frame.mapv_inplace(|v| v / (granularity as f32));
            if let Some(preprocess) = &process_fn {
                frame = preprocess(frame).unwrap();
            }

            // Ensure frame is HWC, potentially with C == 1
            if frame.ndim() == 2 {
                frame = frame.insert_axis(Axis(2))
            }

            // Convert to img and apply transforms
            let mut img = array3_to_image(
                frame
                    .mapv(|v| P::Subpixel::clamp(v * 255.0))
                    .into_dimensionality()
                    .unwrap(),
            );

            if downscale != 1.0 {
                let (w, h) = img.dimensions();
                img = resize(
                    &img,
                    (w as f32 / downscale).round() as u32,
                    (h as f32 / downscale).round() as u32,
                    FilterType::CatmullRom,
                );
            }

            apply_transforms(img, &transform[..])
        })
        .collect();

    Ok(granular_frames)
}

pub fn panorama<P>(
    slice: PhotonCubeView,
    matching_args: &MatchingArgs,
    burst_size: usize,
    step: usize,
    transform: &[Transform],
    granularity: usize,
    is_bitpacked: bool,
    process_fn: &Option<impl Fn(ArrayD<f32>) -> Result<ArrayD<f32>> + Send + Sync>,
    callback_fn: &Option<impl Fn(Vec<Mapping>, &[Image<P>], u32, u32) -> Result<()>>,
) -> Result<(Vec<Vec<Mapping>>, Vec<Image<P>>, Vec<Image<P>>)>
where
    P: Pixel + Send + Sync + 'static,
    <P as Pixel>::Subpixel: Send + Sync + Clamp<f32>,
    f32: From<<P as Pixel>::Subpixel>,
{
    let num_ves = (slice.len_of(Axis(0)) / burst_size) / step;
    let num_frames_per_burst = burst_size / granularity;
    let mut mappings: Vec<Mapping> = vec![Mapping::from_params(vec![0.0; 2]); num_ves - 1];
    let mut all_mappings: Vec<Vec<Mapping>> = vec![];
    let mut virtual_exposures: Vec<Image<P>> = vec![];

    let granular_frames = make_granular_frames::<P>(
        slice,
        process_fn,
        &transform,
        granularity,
        is_bitpacked,
        matching_args.downscale,
    )?;
    let (w, h) = granular_frames[0].dimensions();
    let (lvls_h, lvls_w) = (
        f32::log2(h as f32 / matching_args.downscale / matching_args.min_size as f32).ceil(),
        f32::log2(w as f32 / matching_args.downscale / matching_args.min_size as f32).ceil(),
    );
    let num_lvls = (lvls_h as u32)
        .min(lvls_w as u32)
        .min(matching_args.max_lvls);

    // -------------------- Main hierarchical matching process ----------------------------
    for lvl in (0..num_lvls).rev() {
        // Interpolate mappings to all bitplanes
        mappings = mappings.iter().map(|m| m.rescale(0.5)).collect();
        let acc_maps = Mapping::accumulate(mappings.clone());
        let interpd_maps = Mapping::interpolate_array(
            Array1::linspace(0.0, (num_ves - 1) as f32, num_ves).to_vec(),
            acc_maps,
            Array1::linspace(0.0, (num_ves - 1) as f32, num_ves * num_frames_per_burst).to_vec(),
        );
        let downscale = 1 << lvl;

        // Compute virtual exposure by merging `num_frames_per_chunk` granular frames, and downscaling result
        virtual_exposures = (
            granular_frames
                .par_chunks(num_frames_per_burst)
                .step_by(step),
            interpd_maps.par_chunks(num_frames_per_burst),
        )
            .into_par_iter()
            .progress()
            .with_style(
                ProgressStyle::with_template(
                    "{msg} ETA:{eta}, [{elapsed_precise}] {wide_bar:.cyan/blue} {pos:>6}/{len:6}",
                )
                .expect("Invalid progress style."),
            )
            .with_message(format!(
                "({}/{}): Loading Data...",
                num_lvls - lvl,
                num_lvls
            ))
            .map(|(frames, maps)| {
                let img = merge_images(
                    &Mapping::with_respect_to_idx(maps.to_vec(), 0.5),
                    frames,
                    Some((w as usize, h as usize)),
                    None,
                )
                .unwrap();

                resize(
                    &img,
                    (w as f32 / downscale as f32).round() as u32,
                    (h as f32 / downscale as f32).round() as u32,
                    FilterType::CatmullRom,
                )
            })
            .collect();

        // Estimate pairwise registration
        // TODO: Fix this needless copying!
        print!("({}/{}): Matching... ", num_lvls - lvl, num_lvls);
        (mappings, _) = pairwise_iclk(
            virtual_exposures
                .clone()
                .into_iter()
                .map(|ve| image_to_array3(ve).mapv(f32::from))
                .collect::<Vec<_>>()
                .as_slice(),
            &mappings[..],
            false,
            None,
            None,
            Some(matching_args.iterations),
            Some(matching_args.early_stop),
            Some(matching_args.patience),
            true,
        )?;
        all_mappings.push(mappings.clone());
        println!("Done.");

        // Augment mapping type every iteration
        mappings = mappings.iter().map(|m| m.upgrade()).collect();

        if let Some(callback) = &callback_fn {
            callback(mappings.clone(), &virtual_exposures, lvl, num_lvls)?;
        }
    }
    // ----------------------------------------------------------------------------------

    Ok((all_mappings, virtual_exposures, granular_frames))
}

pub fn visualization_callback<P, R, F>(
    viz_path: R,
    viz_fps: u64,
    viz_step: usize,
    wrt: f32,
    process_fn: &Option<F>,
) -> Result<impl Fn(Vec<Mapping>, &[Image<P>], u32, u32) -> Result<()> + use<'_, P, R, F>>
where
    P: Pixel + PixelWithColorType + Send + Sync,
    <P as Pixel>::Subpixel:
        num_traits::Zero + Clone + Copy + ValueInto<f32> + Send + Sync + Clamp<f32>,
    [<P as Pixel>::Subpixel]: EncodableLayout,
    f32: From<<P as Pixel>::Subpixel>,
    R: AsRef<Path>,
    F: Fn(Image<P>) -> Image<P> + Sync,
{
    Ok(
        move |mappings: Vec<Mapping>, frames: &[Image<P>], lvl: u32, num_lvls: u32| {
            let directory = Path::new(viz_path.as_ref());
            let file = format!("lvl-{}.mp4", num_lvls - lvl);
            let path = directory.join(file);
            stabilized_video(
                &Mapping::accumulate_wrt_idx(mappings, wrt),
                &frames,
                None,
                Some(viz_fps),
                Some(viz_step),
                Some(
                    path.to_str()
                        .expect("Cannot join output visualization paths"),
                ),
                &process_fn,
                Some(format!("({}/{}): Creating Preview...", num_lvls - lvl, num_lvls).as_str()),
            )
        },
    )
}
