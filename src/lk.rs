use std::collections::HashMap;

use anyhow::{anyhow, Result};
use conv::{ValueFrom, ValueInto};
use image::imageops::colorops::grayscale;
use image::imageops::{resize, FilterType};
use image::{GenericImageView, Luma, Pixel, Primitive};
use imageproc::{
    definitions::{Clamp, Image},
    filter::filter3x3,
    gradients::{HORIZONTAL_SOBEL, VERTICAL_SOBEL},
};

use indicatif::{ProgressIterator, ProgressStyle};
use itertools::{izip, Itertools};
use ndarray::{par_azip, s, stack, Array1, Array2, ArrayBase, Axis, NewAxis};
use ndarray_linalg::solve::Inverse;
use nshare::ToNdarray2;
use rayon::prelude::*;

use crate::{
    blend::interpolate_bilinear,
    warps::{Mapping, TransformationType},
};

type Subpixel<I> = <<I as GenericImageView>::Pixel as Pixel>::Subpixel;

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
    <P as Pixel>::Subpixel: Send + Sync,
    <P as Pixel>::Subpixel: ValueInto<f32> + From<u8> + Clamp<f32>,
    f32: ValueFrom<<P as Pixel>::Subpixel> + From<<P as Pixel>::Subpixel>,
{
    let im1_gray = grayscale(im1);
    let im2_gray = grayscale(im2);
    let (dx, dy) = gradients(&im2_gray);

    iclk_grayscale(&im1_gray, &im2_gray, (dx, dy), init_mapping, max_iters, stop_early)
}

/// Main iclk routine, only works for grayscale images
pub fn iclk_grayscale<T>(
    im1_gray: &Image<Luma<T>>,
    im2_gray: &Image<Luma<T>>,
    im2_grad: (Array2<f32>, Array2<f32>),
    init_mapping: Mapping,
    max_iters: Option<i32>,
    stop_early: Option<f32>,
) -> Result<(Mapping, Vec<Vec<f32>>)>
where
    T: Primitive + Clamp<f32> + Send + Sync,
    f32: ValueFrom<T> + From<T>,
{
    // Initialize values
    let mut params = init_mapping.inverse().get_params();
    let h = im2_gray.height().min(im1_gray.height());
    let w = im2_gray.width().min(im1_gray.width());
    let points: Vec<(f32, f32)> = (0..h)
        .cartesian_product(0..w)
        .map(|(y, x)| (x as f32, y as f32))
        .collect();

    let sampler = |(x, y)| {
        interpolate_bilinear(im1_gray, x, y)
            .map(|p| p.channels().iter().map(|v| f32::from(*v)).sum())
    };
    let im2gray_pixels: Vec<f32> = im2_gray
        .pixels()
        .map(|p| p.channels().iter().map(|v| f32::from(*v)).sum())
        .collect();
    let (dx, dy) = im2_grad;

    let mut params_history = vec![];
    params_history.push(params.clone());

    // These can be cached, so we init them before the main loop
    let (steepest_descent_ic_t, hessian_inv) = match init_mapping.kind {
        TransformationType::Translational => {
            let steepest_descent_ic_t = stack![
                Axis(1),
                dx.into_shape(points.len())?,
                dy.into_shape(points.len())?
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
            let mut steepest_descent_ic_t = Array2::zeros((points.len(), params.len()));

            let jacobian_p = {
                let (x, y): (Vec<_>, Vec<_>) = points.iter().cloned().unzip();
                let ones: Array1<f32> = ArrayBase::ones(points.len());
                let zeros: Array1<f32> = ArrayBase::zeros(points.len());
                stack![
                    Axis(0),
                    stack![Axis(0), x, zeros, y, zeros, ones, zeros],
                    stack![Axis(0), zeros, x, zeros, y, zeros, ones]
                ]
                .permuted_axes([2, 0, 1])
            }; // (Nx2x6)

            let grad_im2 = stack![
                Axis(2),
                dx.into_shape((points.len(), 1))?,
                dy.into_shape((points.len(), 1))?
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
            let mut steepest_descent_ic_t = Array2::zeros((points.len(), params.len()));

            // dW_dp evaluated at (x, p) and p=0 (identity transform)
            let jacobian_p = {
                let (x, y): (Vec<_>, Vec<_>) = points.iter().cloned().unzip();
                let ones: Array1<f32> = ArrayBase::ones(points.len());
                let zeros: Array1<f32> = ArrayBase::zeros(points.len());
                let minus_xx: Vec<f32> = x.iter().map(|i1| -i1 * i1).collect();
                let minus_yy: Vec<f32> = y.iter().map(|i1| -i1 * i1).collect();
                let minus_xy: Vec<f32> = x.iter().zip(&y).map(|(i1, i2)| -i1 * i2).collect();
                stack![
                    Axis(0),
                    stack![Axis(0), x, zeros, y, zeros, ones, zeros, minus_xx, minus_xy],
                    stack![Axis(0), zeros, x, zeros, y, zeros, ones, minus_xy, minus_yy]
                ]
                .permuted_axes([2, 0, 1])
            }; // (Nx2x8)

            let grad_im2 = stack![
                Axis(2),
                dx.into_shape((points.len(), 1))?,
                dy.into_shape((points.len(), 1))?
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

    // Main optimization loop
    for _ in (0..max_iters.unwrap_or(250))
        .progress()
        .with_style(ProgressStyle::with_template(
            "{msg} ETA:{eta}, [{elapsed_precise}] {wide_bar:.cyan/blue} {pos:>6}/{len:6}",
        )?)
    {
        // Create mapping from params and use it to warp all points
        let mapping = Mapping::from_params(&params);
        let warped_points = points.par_iter().map(mapping.warpfn());
        let warped_im1gray_pixels: Vec<Option<f32>> = warped_points.map(sampler).collect();

        // Calculate parameter update dp
        let dp: Array2<f32> = hessian_inv.dot(
            &(
                steepest_descent_ic_t.axis_iter(Axis(0)),
                &warped_im1gray_pixels,
                &im2gray_pixels,
            )
                // Zip together all three iterators
                .into_par_iter()
                // Drop them if the warped value from is None (i.e: out-of-bounds)
                .filter_map(|(sd, p1_opt, p2)| p1_opt.map(|p1| sd.to_owned() * (p1 - p2)))
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

    Ok((
        Mapping::from_params(&params).inverse(),
        params_history
    ))
}

pub fn hierarchical_iclk<P>(
    im1: &Image<P>,
    im2: &Image<P>,
    init_mapping: Mapping,
    max_iters: Option<i32>,
    min_dimensions: (u32, u32),
    max_levels: i32,
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
        println!("Using scale {:}", &current_scale);

        // Perform optimization at lvl
        let params_history;
        (mapping, params_history) = iclk_grayscale(&im1, &im2, gradients(&im2), mapping, max_iters, stop_early)?;

        // Re-normalize mapping to scale of next level of pyramid
        mapping = mapping.transform(
            Some(Mapping::scale(2.0, 2.0)),
            Some(Mapping::scale(0.5, 0.5)), 
        );

        // Save level's param history
        all_params_history.insert(current_scale as u32, params_history);
    }

    Ok((mapping, all_params_history))
}


pub fn img_pyramid<P>(im: &Image<P>, min_dimensions: (u32, u32), max_levels: i32) -> Vec<Image<P>>
where
    P: Pixel + 'static,
{
    let (min_width, min_height) = min_dimensions;
    let (mut w, mut h) = im.dimensions();
    let mut stack = vec![im.clone()];

    for _ in 0..max_levels {
        if w > min_width && h > min_height {
            (w, h) = ((w as f32 / 2.0).round() as u32, (h as f32 / 2.0).round() as u32);
            let resized = resize(im, w, h, FilterType::CatmullRom);
            stack.push(resized);
        }
    }
    stack
}
