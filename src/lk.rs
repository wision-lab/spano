use std::fs;

use anyhow::{anyhow, Result};
use conv::{ValueFrom, ValueInto};
use image::imageops::colorops::grayscale;
use image::{GenericImageView, Luma, Pixel, Primitive};

use imageproc::{
    definitions::{Clamp, Image},
    filter::filter3x3,
    gradients::{HORIZONTAL_SOBEL, VERTICAL_SOBEL},
};
use indicatif::{ProgressIterator, ProgressStyle};
use itertools::Itertools;
use ndarray::{par_azip, stack, Array1, Array2, Array3, ArrayBase, Axis};
use ndarray_linalg::norm::Norm;
use ndarray_linalg::solve::Inverse;
use nshare::ToNdarray2;
use rayon::prelude::*;

use crate::{
    blend::interpolate_bilinear,
    warps::{Mapping, TransformationType},
};

type Subpixel<I> = <<I as GenericImageView>::Pixel as Pixel>::Subpixel;

/// Compute image gradients using Sobel operator
/// Returned array is 2xHxW, dx then dy.
pub fn gradients<T>(img: &Image<Luma<T>>) -> Array3<f32>
where
    T: Primitive,
    f32: ValueFrom<T>,
{
    stack![
        Axis(0),
        filter3x3(img, &VERTICAL_SOBEL.map(|v| v as f32)).into_ndarray2(),
        filter3x3(img, &HORIZONTAL_SOBEL.map(|v| v as f32)).into_ndarray2(),
    ]
}

//              T               I
pub fn iclk<P>(
    im1: &Image<P>,
    im2: &Image<P>,
    kind: TransformationType,
    max_iters: Option<i32>,
) -> Result<Mapping>
where
    P: Pixel + Send + Sync,
    <P as Pixel>::Subpixel: Send + Sync,
    <P as Pixel>::Subpixel: ValueInto<f32> + From<u8> + Clamp<f32>,
    f32: ValueFrom<<P as Pixel>::Subpixel> + From<<P as Pixel>::Subpixel>,
{
    // Initialize values
    let mut params = vec![0.0, 0.0];
    let h = im2.height().min(im1.height());
    let w = im2.width().min(im1.width());
    let points: Vec<(f32, f32)> = (0..h)
        .cartesian_product(0..w)
        .map(|(x, y)| (x as f32, y as f32))
        .collect();
    let im1_gray = grayscale(im1);
    let im2_gray = grayscale(im2);

    let sampler = |(x, y)| {
        interpolate_bilinear(&im1_gray, x, y)
            .map(|p| p.channels().iter().map(|v| f32::from(*v)).sum())
    };
    let im2gray_pixels: Vec<f32> = im2_gray
        .pixels()
        .map(|p| p.channels().iter().map(|v| f32::from(*v)).sum())
        .collect();

    let mut history: Vec<f32> = Vec::new();

    // These can be cached, so we init them before the main loop
    let (steepest_descent_ic, hessian_inv) = match kind {
        TransformationType::Translational => {
            let steepest_descent_ic = gradients(&im2_gray).permuted_axes([2, 0, 1])
                .into_shape((kind.num_params(), points.len()))?;

            let hessian_inv = {
                let hessian: f32 = steepest_descent_ic
                    .axis_iter(Axis(0))
                    .into_par_iter()
                    .map(|a| a.dot(&a))
                    .sum();
                1.0 / hessian
            };
            // (Nx2)              f32
            (steepest_descent_ic, hessian_inv)
        }
        TransformationType::Affine => {
            let mut steepest_descent_ic = Array2::zeros((points.len(), kind.num_params()));

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

            let grad_im2 = gradients(&im2_gray)
                .into_shape((2, points.len()))?
                .permuted_axes([1, 0]);

            // Perform the batch matrix multiply of grad_im2 @ jacobian_p
            par_azip!(
                (
                    mut v in steepest_descent_ic.axis_iter_mut(Axis(0)),
                    a in grad_im2.axis_iter(Axis(0)),
                    b in jacobian_p.axis_iter(Axis(0))
                )
                {v.assign(&a.dot(&b))}
            );

            let hessian_inv = {
                let hessian: f32 = steepest_descent_ic
                    .axis_iter(Axis(0))
                    .into_par_iter()
                    .map(|a| a.dot(&a))
                    .sum();
                1.0 / hessian
            };
            // (Nx6)              f32
            (steepest_descent_ic, hessian_inv)
        }
        _ => return Err(anyhow!("Mapping type {:?} not yet supported!", kind)),
    };

    // Main optimization loop
    for _ in (0..max_iters.unwrap_or(100))
        .progress()
        .with_style(ProgressStyle::with_template(
            "{msg} ETA:{eta}, [{elapsed_precise}] {wide_bar:.cyan/blue} {pos:>6}/{len:6}",
        )?)
    {
        // Create mapping from params and use it to warp all points
        let mapping = Mapping::from_params(&params);
        // let warped_points = points.par_iter().map(mapping.warpfn_centered(im1.dimensions()));
        let warped_points = points.par_iter().map(mapping.warpfn());
        let warped_im1gray_pixels: Vec<Option<f32>> = warped_points.map(sampler).collect();

        // Calculate parameter update dp
        let dp: Array1<f32> = hessian_inv
            * (
                steepest_descent_ic.axis_iter(Axis(0)),
                &warped_im1gray_pixels,
                &im2gray_pixels,
            )
                // Zip together all three iterators
                .into_par_iter()
                // Drop them if the warped value from is None (i.e: out-of-bounds)
                .filter_map(|(sd, p1_opt, p2)| p1_opt.map(|p1| sd.to_owned() * (p1 - p2)))
                // Sum them together, here we use reduce with a base value of zero
                .reduce(|| Array1::<f32>::zeros(kind.num_params()), |a, b| a + b);
        let mapping_dp = Mapping::from_params(&dp.clone().into_raw_vec());

        // Update the parameters
        params = Mapping::from_matrix(
            mapping.mat.dot(&mapping_dp.mat.inv()?),
            TransformationType::Affine,
        )
        .get_params();
        history.push(dp.norm());

        // Early exit if dp is small
        if Array1::<f32>::zeros(kind.num_params()).abs_diff_eq(&dp, 1e-7) {
            break;
        }
    }

    let serialized = serde_json::to_string(&history)?;
    fs::write("hist.json", serialized)?;

    Ok(Mapping::from_params(&params))
}
