use std::collections::HashMap;

use anyhow::{anyhow, Result};
use conv::{ValueFrom, ValueInto};
use image::imageops::colorops::grayscale;
use image::imageops::{resize, FilterType};
use image::{Luma, Pixel, Primitive};
use imageproc::{
    definitions::{Clamp, Image},
    filter::filter3x3,
    gradients::{HORIZONTAL_SOBEL, VERTICAL_SOBEL},
};

use indicatif::{ProgressBar, ProgressStyle};
use itertools::izip;
use ndarray::{
    array, par_azip, s, stack, Array, Array1, Array2, Array3, ArrayBase, ArrayView, Axis, NewAxis,
};
use ndarray_linalg::solve::Inverse;
use nshare::ToNdarray2;
use rayon::prelude::*;

use crate::transforms::image_to_array3;
use crate::warps::{warp_array3_into, Mapping, TransformationType};

/// Compute image gradients using Sobel operator
/// Returned (dx, dy) pair as HxW arrays.
pub fn gradients<T>(img: &Image<Luma<T>>) -> (Array2<f32>, Array2<f32>)
where
    T: Primitive,
    f32: ValueFrom<T>,
{
    (
        filter3x3(img, &HORIZONTAL_SOBEL.map(|v| v as f32)).into_ndarray2(),
        filter3x3(img, &VERTICAL_SOBEL.map(|v| v as f32)).into_ndarray2(),
    )
}

/// Estimate the warp that maps `img2` to `img1` using the Inverse Compositional Lucas-Kanade algorithm.
/// In other words, img2 is the template and img1 is the static image.
pub fn iclk<P>(
    im1: &Image<P>,
    im2: &Image<P>,
    init_mapping: Mapping,
    max_iters: Option<i32>,
    stop_early: Option<f32>,
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
        max_iters,
        stop_early,
        Some(""),
    )
}

/// Main iclk routine, only works for grayscale images
pub fn iclk_grayscale<T>(
    im1_gray: &Image<Luma<T>>,
    im2_gray: &Image<Luma<T>>,
    im2_grad: (Array2<f32>, Array2<f32>),
    init_mapping: Mapping,
    max_iters: Option<i32>,
    stop_early: Option<f32>,
    message: Option<&str>,
) -> Result<(Mapping, Vec<Vec<f32>>)>
where
    T: Primitive + Clamp<f32> + Send + Sync + 'static,
    f32: ValueFrom<T> + From<T>,
{
    // Initialize values
    let mut params = init_mapping.inverse().get_params();
    let h = im2_gray.height().min(im1_gray.height());
    let w = im2_gray.width().min(im1_gray.width());
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
    let mut warped_im1gray_pixels = Array3::<f32>::zeros((h as usize, w as usize, 1));
    let mut valid = Array2::<bool>::from_elem((h as usize, w as usize), false);
    let (dx, dy) = im2_grad;

    let mut params_history = vec![];
    params_history.push(params.clone());

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

    // Conditionally setup a pbar
    let pbar = if let Some(msg) = message {
        ProgressBar::new(max_iters.unwrap_or(250) as u64)
            .with_style(ProgressStyle::with_template(
                "{msg} ETA:{eta}, [{elapsed_precise}] {wide_bar:.cyan/blue} {pos:>6}/{len:6}",
            )?)
            .with_message(msg.to_owned())
    } else {
        ProgressBar::hidden()
    };

    // Main optimization loop
    for i in 0..max_iters.unwrap_or(250) {
        pbar.set_position(i as u64);

        // Create mapping from params and use it to sample points from img1
        let mapping = Mapping::from_params(&params);
        warp_array3_into::<_, f32>(
            &mapping,
            &img1_array,
            &mut warped_im1gray_pixels,
            &mut valid,
            Some(&points),
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
                .reduce(|| Array2::<f32>::zeros((params.len(), 1)), |a, b| a + b),
        );
        let mapping_dp = Mapping::from_params(&dp.clone().into_raw_vec());

        // Update the parameters
        params = Mapping::from_matrix(mapping.mat.dot(&mapping_dp.mat.inv()?), init_mapping.kind)
            .get_params();
        params_history.push(params.clone());

        // Early exit if dp is small
        if Array2::<f32>::zeros((params.len(), 1)).abs_diff_eq(&dp, stop_early.unwrap_or(1e-3)) {
            break;
        }
    }

    Ok((Mapping::from_params(&params).inverse(), params_history))
}

#[allow(clippy::type_complexity)]
pub fn hierarchical_iclk<P>(
    im1: &Image<P>,
    im2: &Image<P>,
    init_mapping: Mapping,
    max_iters: Option<i32>,
    min_dimensions: (u32, u32),
    max_levels: u32,
    stop_early: Option<f32>,
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

    for (i, (im1, im2)) in izip!(
        stack_im1[..num_lvls].iter().rev(),
        stack_im2[..num_lvls].iter().rev()
    )
    .enumerate()
    {
        // Compute mapping at lowest resolution first and double resolution it at each iteration
        let current_scale = (1 << (num_lvls - i - 1)) as f32;

        // Perform optimization at lvl
        let params_history;
        (mapping, params_history) = iclk_grayscale(
            im1,
            im2,
            gradients(im2),
            mapping,
            max_iters,
            stop_early,
            Some(&format!("Using scale 1/{:}", &current_scale)),
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

pub fn img_pyramid<P>(im: &Image<P>, min_dimensions: (u32, u32), max_levels: u32) -> Vec<Image<P>>
where
    P: Pixel + 'static,
{
    let (min_width, min_height) = min_dimensions;
    let (mut w, mut h) = im.dimensions();
    let mut stack = vec![im.clone()];

    for _ in 0..max_levels {
        if w > min_width && h > min_height {
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
