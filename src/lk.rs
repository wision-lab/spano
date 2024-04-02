use std::collections::{HashMap, VecDeque};

use anyhow::{anyhow, Result};
use conv::{ValueFrom, ValueInto};
use image::{
    imageops::{colorops::grayscale, resize, FilterType},
    GrayImage, Luma, Pixel, Primitive,
};
use imageproc::{
    definitions::{Clamp, Image},
    filter::filter3x3,
    gradients::{HORIZONTAL_SCHARR, VERTICAL_SCHARR},
};
use itertools::izip;
use ndarray::{
    array, par_azip, s, stack, Array, Array1, Array2, ArrayBase, ArrayView, Axis, NewAxis,
};
use ndarray_linalg::solve::Inverse;
use nshare::ToNdarray2;
use photoncube2video::transforms::image_to_array3;
use rayon::prelude::*;

use crate::{
    utils::get_pbar,
    warps::{Mapping, TransformationType},
};

/// Compute image gradients using Scharr operator
/// Returned (dx, dy) pair as HxW arrays.
pub fn gradients<T>(img: &Image<Luma<T>>) -> (Array2<f32>, Array2<f32>)
where
    T: Primitive,
    f32: ValueFrom<T>,
{
    (
        filter3x3(img, &HORIZONTAL_SCHARR.map(|v| v as f32)).into_ndarray2(),
        filter3x3(img, &VERTICAL_SCHARR.map(|v| v as f32)).into_ndarray2(),
    )
}

/// Estimate the warp that maps `img2` to `img1` using the Inverse Compositional Lucas-Kanade algorithm.
/// In other words, img2 is the template and img1 is the static image.
pub fn iclk<P>(
    im1: &Image<P>,
    im2: &Image<P>,
    init_mapping: Mapping,
    im1_mask: Option<&GrayImage>,
    max_iters: Option<i32>,
    stop_early: Option<f32>,
    patience: Option<usize>,
    message: Option<&str>,
) -> Result<(Mapping, Vec<Vec<f32>>)>
where
    P: Pixel + Send + Sync,
    <P as Pixel>::Subpixel: Send + Sync + 'static,
    <P as Pixel>::Subpixel: ValueInto<f32> + From<u8> + Clamp<f32>,
    f32: ValueFrom<<P as Pixel>::Subpixel> + From<<P as Pixel>::Subpixel>,
{
    let im1_gray = grayscale(im1);
    let im2_gray = grayscale(im2);
    let (dx, dy) = gradients(&im2_gray);

    iclk_grayscale(
        &im1_gray,
        &im2_gray,
        (dx, dy),
        init_mapping,
        im1_mask,
        max_iters,
        stop_early,
        patience,
        message,
    )
}

/// Main iclk routine, only works for grayscale images
/// This returns the mapping that warps image 2 onto image 1's reference frame.
/// The param history however, corresponds to the inverse mappings, i.e from 1 to 2.
/// Note: No input validation is performed here, im1 and im2 can have different sizes but
///     the im2 gradients need to have the same size as im2 and im1 mask should match im1.
///
/// See: https://www.ri.cmu.edu/pub_files/pub3/baker_simon_2002_3/baker_simon_2002_3.pdf
///
/// TODO: Is anything actually enforcing this to be grayscale/single-channel?
#[allow(clippy::too_many_arguments)]
pub fn iclk_grayscale<T>(
    im1_gray: &Image<Luma<T>>,
    im2_gray: &Image<Luma<T>>,
    im2_grad: (Array2<f32>, Array2<f32>),
    init_mapping: Mapping,
    _im1_mask: Option<&GrayImage>,
    max_iters: Option<i32>,
    stop_early: Option<f32>,
    patience: Option<usize>,
    message: Option<&str>,
) -> Result<(Mapping, Vec<Vec<f32>>)>
where
    T: Primitive + Clamp<f32> + Send + Sync + 'static,
    f32: ValueFrom<T> + From<T>,
{
    // Initialize values
    let mut params = init_mapping.inverse().get_params();
    let num_params = params.len();
    let h = im2_gray.height();
    let w = im2_gray.width();
    let points = Array::from_shape_fn(((w * h) as usize, 2), |(i, j)| {
        if j == 0 {
            i % w as usize
        } else {
            i / w as usize
        }
    });
    let xs = points.column(0).mapv(|v| v as f32);
    let ys = points.column(1).mapv(|v| v as f32);
    let num_points = (w * h) as usize;

    let img1_array = image_to_array3(im1_gray.clone()).mapv(|v| f32::from(v));
    let img2_pixels = image_to_array3(im2_gray.clone())
        .mapv(|v| f32::from(v))
        .into_shape((num_points, 1))?;
    let mut warped_im1gray_pixels = Array2::<f32>::zeros((num_points, 1));
    let mut valid = Array1::<bool>::from_elem(num_points, false);
    let (dx, dy) = im2_grad;

    let mut params_history = vec![];
    params_history.push(params.clone());
    let mut dps: VecDeque<Array2<f32>> = VecDeque::with_capacity(patience.unwrap_or(10));

    // These can be cached, so we init them before the main loop
    let (steepest_descent_ic_t, hessian_inv) = match init_mapping.kind {
        TransformationType::Translational => {
            let steepest_descent_ic_t = stack![
                Axis(1),
                dx.into_shape(num_points)?,
                dy.into_shape(num_points)?
            ]; // (Nx2)

            let hessian_inv = (steepest_descent_ic_t.clone().permuted_axes([1, 0]))
                .dot(&steepest_descent_ic_t)
                .inv()
                .unwrap();

            // (Nx2x1)              (2x2)
            (
                steepest_descent_ic_t.slice(s![.., .., NewAxis]).to_owned(),
                hessian_inv,
            )
        }
        TransformationType::Affine => {
            let mut steepest_descent_ic_t = Array2::zeros((num_points, params.len()));

            let jacobian_p = {
                let ones: Array1<f32> = ArrayBase::ones(num_points);
                let zeros: Array1<f32> = ArrayBase::zeros(num_points);
                stack![
                    Axis(0),
                    stack![Axis(0), xs, zeros, ys, zeros, ones, zeros],
                    stack![Axis(0), zeros, xs, zeros, ys, zeros, ones]
                ]
                .permuted_axes([2, 0, 1])
            }; // (Nx2x6)

            let grad_im2 = stack![
                Axis(2),
                dx.into_shape((num_points, 1))?,
                dy.into_shape((num_points, 1))?
            ]; // (Nx1x2)

            // Perform the batch matrix multiply of grad_im2 @ jacobian_p
            par_azip!(
                (
                    mut v in steepest_descent_ic_t.axis_iter_mut(Axis(0)),
                    a in grad_im2.axis_iter(Axis(0)),
                    b in jacobian_p.axis_iter(Axis(0))
                )
                {v.assign(&a.dot(&b).into_shape(params.len()).unwrap())}
            );

            let hessian_inv = (steepest_descent_ic_t.clone().permuted_axes([1, 0]))
                .dot(&steepest_descent_ic_t)
                .inv()
                .unwrap();

            // (Nx6x1)              (6x6)
            (
                steepest_descent_ic_t.slice(s![.., .., NewAxis]).to_owned(),
                hessian_inv,
            )
        }
        TransformationType::Projective => {
            let mut steepest_descent_ic_t = Array2::zeros((num_points, params.len()));

            // dW_dp evaluated at (x, p) and p=0 (identity transform)
            let jacobian_p = {
                let ones: Array1<f32> = ArrayBase::ones(num_points);
                let zeros: Array1<f32> = ArrayBase::zeros(num_points);
                let minus_xx: Vec<f32> = xs.iter().map(|i1| -i1 * i1).collect();
                let minus_yy: Vec<f32> = ys.iter().map(|i1| -i1 * i1).collect();
                let minus_xy: Vec<f32> = xs.iter().zip(&ys).map(|(i1, i2)| -i1 * i2).collect();
                stack![
                    Axis(0),
                    stack![
                        Axis(0),
                        xs,
                        zeros,
                        ys,
                        zeros,
                        ones,
                        zeros,
                        minus_xx,
                        minus_xy
                    ],
                    stack![
                        Axis(0),
                        zeros,
                        xs,
                        zeros,
                        ys,
                        zeros,
                        ones,
                        minus_xy,
                        minus_yy
                    ]
                ]
                .permuted_axes([2, 0, 1])
            }; // (Nx2x8)

            let grad_im2 = stack![
                Axis(2),
                dx.into_shape((num_points, 1))?,
                dy.into_shape((num_points, 1))?
            ]; // (Nx1x2)

            // Perform the batch matrix multiply of grad_im2 @ jacobian_p
            par_azip!(
                (
                    mut v in steepest_descent_ic_t.axis_iter_mut(Axis(0)),
                    a in grad_im2.axis_iter(Axis(0)),
                    b in jacobian_p.axis_iter(Axis(0))
                )
                {v.assign(&a.dot(&b).into_shape(params.len()).unwrap())}
            );

            let hessian_inv = (steepest_descent_ic_t.clone().permuted_axes([1, 0]))
                .dot(&steepest_descent_ic_t)
                .inv()
                .unwrap();

            // (Nx8x1)              (8x8)
            (
                steepest_descent_ic_t.slice(s![.., .., NewAxis]).to_owned(),
                hessian_inv,
            )
        }
        _ => {
            return Err(anyhow!(
                "Mapping type {:?} not supported!",
                init_mapping.kind
            ))
        }
    };
    let pbar = get_pbar(max_iters.unwrap_or(250) as usize, message);

    // Main optimization loop
    for i in 0..max_iters.unwrap_or(250) {
        pbar.set_position(i as u64);

        // Create mapping from params and use it to sample points from img1
        let mapping = Mapping::from_params(params);
        mapping.warp_array3_into::<f32, _, _, _, _, _>(
            &img1_array,
            &mut warped_im1gray_pixels,
            &mut valid,
            &points,
            Some(array![0.0]),
            None,
        );

        let warped_im1gray_pixels_view =
            ArrayView::from_shape((num_points, 1), warped_im1gray_pixels.as_slice().unwrap())
                .unwrap();

        let valid_view = ArrayView::from_shape(num_points, valid.as_slice().unwrap()).unwrap();

        // Calculate parameter update dp
        let dp: Array2<f32> = hessian_inv.dot(
            &(
                steepest_descent_ic_t.axis_iter(Axis(0)),
                warped_im1gray_pixels_view.axis_iter(Axis(0)),
                img2_pixels.axis_iter(Axis(0)),
                valid_view.axis_iter(Axis(0)),
            )
                // Zip together all three iterators and valid flag
                .into_par_iter()
                // Drop them if the warped value from is out-of-bounds
                .filter(|(_, _, _, is_valid)| is_valid.iter().all(|i| *i))
                // Calculate parameter update according to formula
                .map(|(sd, p1, p2, _)| sd.to_owned() * (p1.to_owned() - p2.to_owned()).sum())
                // Sum them together, here we use reduce with a base value of zero
                .reduce(|| Array2::<f32>::zeros((num_params, 1)), |a, b| a + b),
        );
        let mapping_dp = Mapping::from_params(dp.clone().into_raw_vec());

        // Update the parameters
        params = Mapping::from_matrix(mapping.mat.dot(&mapping_dp.mat.inv()?), init_mapping.kind)
            .get_params();
        params_history.push(params.clone());

        // Push back dp update, pop old one if deque is full
        if i >= patience.unwrap_or(10) as i32 {
            dps.pop_front();
        }
        dps.push_back(dp.clone());

        // Early exit if average dp is small
        let avg_dp = &dps
            .iter()
            .fold(Array2::<f32>::zeros((num_params, 1)), |acc, e| acc + e)
            / dps.len() as f32;
        if Array2::<f32>::zeros((num_params, 1)).abs_diff_eq(&avg_dp, stop_early.unwrap_or(1e-3)) {
            break;
        }
    }
    pbar.finish_and_clear();

    Ok((Mapping::from_params(params).inverse(), params_history))
}

#[allow(clippy::type_complexity)]
#[allow(clippy::too_many_arguments)]
pub fn hierarchical_iclk<P>(
    im1: &Image<P>,
    im2: &Image<P>,
    init_mapping: Mapping,
    im1_mask: Option<&GrayImage>,
    max_iters: Option<i32>,
    min_dimensions: (u32, u32),
    max_levels: u32,
    stop_early: Option<f32>,
    patience: Option<usize>,
    message: bool,
) -> Result<(Mapping, HashMap<u32, Vec<Vec<f32>>>)>
where
    P: Pixel + Send + Sync,
    <P as Pixel>::Subpixel: Send + Sync + 'static,
    <P as Pixel>::Subpixel: ValueInto<f32> + From<u8> + Clamp<f32>,
    f32: ValueFrom<<P as Pixel>::Subpixel> + From<<P as Pixel>::Subpixel>,
{
    let im1_gray = grayscale(im1);
    let im2_gray = grayscale(im2);

    let stack_im1 = img_pyramid(&im1_gray, min_dimensions, max_levels);
    let stack_im2 = img_pyramid(&im2_gray, min_dimensions, max_levels);

    let num_lvls = stack_im1.len().min(stack_im2.len());
    let mut mapping = init_mapping;
    let mut all_params_history = HashMap::new();

    let stack_mask = im1_mask.map_or(vec![None; num_lvls], |mask| {
        img_pyramid(&mask, min_dimensions, max_levels)
            .into_iter()
            .map(Some)
            .collect()
    });

    for (i, (im1, im2, mask)) in izip!(
        stack_im1[..num_lvls].iter().rev(),
        stack_im2[..num_lvls].iter().rev(),
        stack_mask[..num_lvls].iter().rev()
    )
    .enumerate()
    {        
        // Compute mapping at lowest resolution first and double resolution it at each iteration
        let current_scale = (1 << (num_lvls - i - 1)) as f32;

        // Perform optimization at lvl
        let params_history;
        let msg = format!("Matching scale 1/{:}", &current_scale);
        let msg = if message { Some(msg) } else { None };
        (mapping, params_history) = iclk_grayscale(
            im1,
            im2,
            gradients(im2),
            mapping,
            mask.as_ref(),
            max_iters,
            stop_early,
            patience,
            msg.as_deref(),
        )?;

        // Re-normalize mapping to scale of next level of pyramid
        // But not on last iteration (since we're already at full scale)
        if i + 1 < num_lvls {
            mapping = mapping.rescale(0.5);
        }

        // Save level's param history
        all_params_history.insert(current_scale as u32, params_history);
    }

    Ok((mapping, all_params_history))
}

/// Estimate pairwise registration using single level iclk
#[allow(clippy::too_many_arguments)]
pub fn pairwise_iclk(
    frames: &Vec<GrayImage>,
    init_mappings: &[Mapping],
    iterations: i32,
    early_stop: f32,
    patience: usize,
    message: Option<&str>,
) -> Result<Vec<Mapping>> {
    let pbar = get_pbar(frames.len() - 1, message);

    // Iterate over sliding window of pairwise frames (in parallel!)
    let mappings: Vec<Mapping> = frames
        .par_windows(2)
        .zip(init_mappings)
        .map(|(window, init_mapping)| {
            pbar.inc(1);
            iclk_grayscale(
                &window[0],
                &window[1],
                gradients(&window[1]),
                init_mapping.clone(),
                None,
                Some(iterations),
                Some(early_stop),
                Some(patience),
                None,
            )
            // Drop param_history
            .map(|(mapping, _)| mapping)
        })
        // Collect to force reorder
        .collect::<Result<Vec<_>>>()?;

    // Return raw pairwise warps (N-1 in total)
    pbar.finish_and_clear();
    Ok(mappings)
}

/// Given an image, return an image pyramid with the largest size first and halving the size 
/// every time until either the max-levels are reached or the minimum size is reached.
pub fn img_pyramid<P>(im: &Image<P>, min_dimensions: (u32, u32), max_levels: u32) -> Vec<Image<P>>
where
    P: Pixel + 'static,
{
    let (min_width, min_height) = min_dimensions;
    let (mut w, mut h) = im.dimensions();
    let mut stack = vec![im.clone()];

    for _ in 0..max_levels {
        if w >= min_width * 2 && h >= min_height * 2 {
            (w, h) = (
                (w as f32 / 2.0).round() as u32,
                (h as f32 / 2.0).round() as u32,
            );
            let resized = resize(im, w, h, FilterType::CatmullRom);
            stack.push(resized);
        }
    }
    stack
}
