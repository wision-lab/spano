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
use ndarray::{par_azip, stack, Array1, Array2, ArrayBase, Axis, Array3, s, NewAxis};
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
    let mut params = vec![0.0; kind.num_params()];
    let h = im2.height().min(im1.height());
    let w = im2.width().min(im1.width());
    let points: Vec<(f32, f32)> = (0..h)
        .cartesian_product(0..w)
        // .map(|(y, x)| (
        //     2.0 * (x as f32 / w as f32) - 1.0, 
        //     2.0 * (y as f32 / h as f32) - 1.0
        // ))
        .map(|(y, x)| (x as f32, y as f32))
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

    let mut dp_history: Vec<f32> = Vec::new();
    let mut params_history: Vec<Vec<f32>> = Vec::new();

    // These can be cached, so we init them before the main loop
    let (steepest_descent_ic_t, hessian_inv) = match kind {
        TransformationType::Translational => {
            let (dx, dy) = gradients(&im2_gray);
            let steepest_descent_ic_t = stack![
                Axis(1),
                dx.clone().into_shape(points.len())?,
                dy.clone().into_shape(points.len())?
            ]; // (Nx2)

            let hessian_inv = (steepest_descent_ic_t.clone().permuted_axes([1, 0]))
                .dot(&steepest_descent_ic_t).inv().unwrap();

            // (Nx2x1)              (2x2)
            (steepest_descent_ic_t.slice(s![.., .., NewAxis]).to_owned(), hessian_inv)
        }
        TransformationType::Affine => {
            let mut steepest_descent_ic_t = Array2::zeros((points.len(), kind.num_params()));

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
            println!("jacobian_p {:?}", jacobian_p.shape());
            println!("jacobian_p {:?}", jacobian_p.slice(s![500, .., ..]));

            let (dx, dy) = gradients(&im2_gray);
            println!("dx {:?}", dx.shape());
            let grad_im2 = stack![
                Axis(2),
                dx.into_shape((points.len(), 1))?,
                dy.into_shape((points.len(), 1))?
            ]; // (Nx1x2)
            println!("grad_im2 {:?}", grad_im2.shape());

            // Perform the batch matrix multiply of grad_im2 @ jacobian_p
            par_azip!(
                (
                    mut v in steepest_descent_ic_t.axis_iter_mut(Axis(0)),
                    a in grad_im2.axis_iter(Axis(0)),
                    b in jacobian_p.axis_iter(Axis(0))
                )
                {v.assign(&a.dot(&b).into_shape(kind.num_params()).unwrap())}
            );

            let hessian_inv = (steepest_descent_ic_t.clone().permuted_axes([1, 0]))
                .dot(&steepest_descent_ic_t).inv().unwrap();
            println!("{hessian_inv:?}");

            // (Nx6x1)              (6x6)
            (steepest_descent_ic_t.slice(s![.., .., NewAxis]).to_owned(), hessian_inv)
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
        // let warped_points = points
        //     .par_iter()
        //     .map(mapping.warpfn_centered(im1.dimensions()));
        let warped_points = points.par_iter().map(mapping.warpfn());
        let warped_im1gray_pixels: Vec<Option<f32>> = warped_points.map(sampler).collect();

        // Calculate parameter update dp
        let dp: Array2<f32> = hessian_inv
            .dot(
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
                .reduce(|| Array2::<f32>::zeros((kind.num_params(), 1)), |a, b| a + b)
        );
        let mapping_dp = Mapping::from_params(&dp.clone().into_raw_vec());

        // Update the parameters
        params = Mapping::from_matrix(mapping.mat.dot(&mapping_dp.mat.inv()?), kind).get_params();
        dp_history.push(dp.norm());
        params_history.push(params.clone());

        // Early exit if dp is small
        if Array2::<f32>::zeros((kind.num_params(), 1)).abs_diff_eq(&dp, 1e-7) {
            break;
        }
    }

    let serialized = serde_json::to_string(&dp_history)?;
    fs::write("dp_hist.json", serialized)?;

    let serialized = serde_json::to_string(&params_history)?;
    fs::write("params_hist.json", serialized)?;

    Ok(Mapping::from_params(&params).inverse())
}
