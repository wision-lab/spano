use std::collections::{HashMap, VecDeque};

use anyhow::{anyhow, Result};
use burn::{
    module::{Param, ParamId},
    nn::{conv::Conv2dConfig, Initializer, PaddingConfig2d},
};
use burn_tensor::{ElementConversion, Int, Shape, Tensor};
use conv::{ValueFrom, ValueInto};
use image::{
    imageops::{colorops::grayscale, resize, FilterType},
    GrayImage, Luma, Pixel,
};
use imageproc::{
    definitions::{Clamp, Image},
    filter::filter3x3,
    gradients::{HORIZONTAL_SCHARR, VERTICAL_SCHARR},
};
use itertools::izip;
use ndarray::Ix2;
use ndarray_linalg::solve::Inverse;
use num_traits::ToPrimitive;
use rayon::prelude::*;

use crate::{
    kernels::Backend,
    transforms::{array_to_tensor, bmm, image_to_tensor3, tensor3_to_image, tensor_to_array},
    utils::get_pbar,
    warps::{Mapping, TransformationType},
};

/// Compute image gradients using Scharr operator.
/// Input is expected to be grayscale, HW1 format.
/// Returned (dx, dy) pair as HxWx1 tensors.
/// WARNING: These gradients might be the negative grad due to convolutions... TODO?
pub fn tensor_gradients_<B: Backend>(img: Tensor<B, 3>) -> (Tensor<B, 3>, Tensor<B, 3>) {
    // Input is HWC, needs to be 1CHW for conv to work.
    let [h, w, _] = img.dims();
    let img = img.permute([2, 0, 1]).unsqueeze_dim(0);

    // Weights need to be of shape `[channels_out, channels_in / groups, kernel_size_1, kernel_size_2]`
    let config = Conv2dConfig::new([2, 1], [3, 3])
        .with_padding(PaddingConfig2d::Explicit(1, 1))
        .with_initializer(Initializer::Zeros);
    let device = Default::default();
    let mut conv = config.init::<B>(&device);
    let horizontal = Tensor::<B, 1, Int>::from_ints(HORIZONTAL_SCHARR, &device)
        .reshape([1, 3, 3])
        .float();
    let vertical = Tensor::<B, 1, Int>::from_ints(VERTICAL_SCHARR, &device)
        .reshape([1, 3, 3])
        .float();

    let weights = Tensor::stack(vec![horizontal, vertical], 0);
    conv.weight = Param::initialized(ParamId::new(), weights);

    let grad = conv.forward(img).squeeze(0).permute([1, 2, 0]);
    let grad_x = grad.clone().slice([0..(h as usize), 0..(w as usize), 0..1]);
    let grad_y = grad.clone().slice([0..(h as usize), 0..(w as usize), 1..2]);
    (grad_x, grad_y)
}

/// Compute image gradients using Scharr operator
/// Returned (dx, dy) pair as HxW arrays.
pub fn tensor_gradients<B: Backend>(img: Tensor<B, 3>) -> (Tensor<B, 3>, Tensor<B, 3>) {
    let img_ = tensor3_to_image::<Luma<f32>, B>(img.clone());
    (
        image_to_tensor3::<Luma<f32>, B>(
            filter3x3(&img_, &HORIZONTAL_SCHARR.map(|v| v as f32)),
            &img.device(),
        ),
        image_to_tensor3::<Luma<f32>, B>(
            filter3x3(&img_, &VERTICAL_SCHARR.map(|v| v as f32)),
            &img.device(),
        ),
    )
}

/// Estimate the warp that maps `img2` to `img1` using the Inverse Compositional Lucas-Kanade algorithm.
/// In other words, img2 is the template and img1 is the static image.
#[allow(clippy::too_many_arguments)]
pub fn iclk<P, B>(
    im1: &Image<P>,
    im2: &Image<P>,
    init_mapping: Mapping<B>,
    im1_weights: Option<&GrayImage>,
    max_iters: Option<i32>,
    stop_early: Option<f32>,
    patience: Option<usize>,
    message: Option<&str>,
) -> Result<(Mapping<B>, Vec<Vec<f32>>)>
where
    P: Pixel + Send + Sync,
    <P as Pixel>::Subpixel: Send + Sync + 'static,
    <P as Pixel>::Subpixel: ValueInto<f32> + From<u8> + Clamp<f32>,
    f32: ValueFrom<<P as Pixel>::Subpixel> + From<<P as Pixel>::Subpixel>,
    B: Backend,
{
    let device = &init_mapping.device();
    let im1_gray = image_to_tensor3(grayscale(im1), device);
    let im2_gray = image_to_tensor3(grayscale(im2), device);
    let (dx, dy) = tensor_gradients(im2_gray.clone());
    let im1_weights =
        im1_weights.map(|w| image_to_tensor3::<Luma<u8>, B>(w.clone(), device).squeeze(2));

    iclk_grayscale(
        im1_gray,
        im2_gray,
        (dx, dy),
        init_mapping,
        im1_weights,
        max_iters,
        stop_early,
        patience,
        message,
    )
}

/// Main iclk routine, only works for grayscale images
/// This returns the mapping that warps image 2 onto image 1's reference frame.
/// The param history however, corresponds to the inverse mappings, i.e from 1 to 2.
///
/// Weights can be specified for image 1. They are concatenated to the reference image and
/// warped together. The warped weights then affect that pixel's loss and is effectively
/// discarded from the optimization step if it's zero.
///
/// Note: No input validation is performed here, im1 and im2 can have different sizes but
///     the im2 gradients need to have the same size as im2 and im1 weights should match im1.
///
/// See: https://www.ri.cmu.edu/pub_files/pub3/baker_simon_2002_3/baker_simon_2002_3.pdf
///
/// TODO: Is anything actually enforcing this to be grayscale/single-channel?
#[allow(clippy::too_many_arguments)]
pub fn iclk_grayscale<B: Backend>(
    im1_gray: Tensor<B, 3>,
    im2_gray: Tensor<B, 3>,
    im2_grad: (Tensor<B, 3>, Tensor<B, 3>),
    init_mapping: Mapping<B>,
    im1_weights: Option<Tensor<B, 2>>,
    max_iters: Option<i32>,
    stop_early: Option<f32>,
    patience: Option<usize>,
    message: Option<&str>,
) -> Result<(Mapping<B>, Vec<Vec<f32>>)> {
    // Initialize values
    let mut params = init_mapping.inverse().get_params();
    let num_params = params.len();
    let [h, w, c] = im2_gray.dims();

    let device = &init_mapping.device();
    // TODO: Use modulo here once it's available. See: https://github.com/tracel-ai/burn/pull/1726
    let xs: Tensor<B, 2> = Tensor::arange(0..(w as i64), device)
        .unsqueeze_dim::<2>(0)
        .repeat(0, h)
        .float();
    let ys: Tensor<B, 2> = Tensor::arange(0..(h as i64), device)
        .unsqueeze_dim::<2>(1)
        .repeat(1, w)
        .float();

    // Augment img1 with any weights
    let (im1_gray, has_weights) = if let Some(weights) = im1_weights {
        (
            Tensor::cat(vec![im1_gray, weights.unsqueeze_dim(2)], 2),
            true,
        )
    } else {
        (im1_gray, false)
    };
    // Initialize buffers that we'll warp data into
    // Note: These can be initialized as empty because all values are overridden anyways
    let mut warped_im1gray = B::float_empty(Shape::new([h, w, c + (has_weights as usize)]), device);
    let mut valid = B::bool_empty(Shape::new([h, w]), device);

    // These can be cached, so we init them before the main loop
    let (dx, dy) = im2_grad;
    let (steepest_descent_ic_t, hessian) = match init_mapping.kind {
        TransformationType::Translational => {
            let steepest_descent_ic_t: Tensor<B, 3> = Tensor::cat(vec![dx.clone(), dy.clone()], 2); // (HxWx2)

            let flat_steepest_descent_ic_t: Tensor<B, 2> =
                steepest_descent_ic_t.clone().flatten(0, 1); // (Nx2)

            let hessian = flat_steepest_descent_ic_t
                .clone()
                .transpose()
                .matmul(flat_steepest_descent_ic_t); // (2x2)

            // (HxWx2), (2x2)
            (steepest_descent_ic_t, hessian)
        }
        TransformationType::Affine => {
            let jacobian_p = {
                let ones = Tensor::ones_like(&xs);
                let zeros = Tensor::zeros_like(&ys);
                Tensor::stack(
                    // 2x6xHxW
                    vec![
                        Tensor::stack::<3>(
                            vec![
                                xs.clone(),
                                zeros.clone(),
                                ys.clone(),
                                zeros.clone(),
                                ones.clone(),
                                zeros.clone(),
                            ],
                            0,
                        ), // 6xHxW
                        Tensor::stack::<3>(
                            vec![
                                zeros.clone(),
                                xs.clone(),
                                zeros.clone(),
                                ys.clone(),
                                zeros.clone(),
                                ones.clone(),
                            ],
                            0,
                        ), // 6xHxW
                    ],
                    0,
                )
                .permute([2, 3, 0, 1])
            }; // (HxWx2x6)

            let grad_im2: Tensor<B, 4> = Tensor::stack(vec![dx.clone(), dy.clone()], 3); // (HxWx1x2)
            let steepest_descent_ic_t = bmm::<B>(grad_im2.flatten(0, 1), jacobian_p.flatten(0, 1))
                .reshape(Shape::new([h, w, 6]));

            let flat_steepest_descent_ic_t: Tensor<B, 2> =
                steepest_descent_ic_t.clone().flatten(0, 1); // (Nx6)

            let hessian = flat_steepest_descent_ic_t
                .clone()
                .transpose()
                .matmul(flat_steepest_descent_ic_t); // (6x6)

            // (HxWx6), (6x6)
            (steepest_descent_ic_t, hessian)
        }
        TransformationType::Projective => {
            // dW_dp evaluated at (x, p) and p=0 (identity transform)
            let jacobian_p = {
                let ones = Tensor::ones_like(&xs);
                let zeros = Tensor::zeros_like(&ys);

                let minus_xx = -xs.clone() * xs.clone();
                let minus_yy = -ys.clone() * ys.clone();
                let minus_xy = -xs.clone() * ys.clone();

                Tensor::stack(
                    // 2x8xHxW
                    vec![
                        Tensor::stack::<3>(
                            vec![
                                xs.clone(),
                                zeros.clone(),
                                ys.clone(),
                                zeros.clone(),
                                ones.clone(),
                                zeros.clone(),
                                minus_xx.clone(),
                                minus_xy.clone(),
                            ],
                            0,
                        ), // 8xHxW
                        Tensor::stack::<3>(
                            vec![
                                zeros.clone(),
                                xs.clone(),
                                zeros.clone(),
                                ys.clone(),
                                zeros.clone(),
                                ones.clone(),
                                minus_xy.clone(),
                                minus_yy.clone(),
                            ],
                            0,
                        ), // 8xHxW
                    ],
                    0,
                )
                .permute([2, 3, 0, 1])
            }; // (HxWx2x8)

            let grad_im2: Tensor<B, 4> = Tensor::stack(vec![dx.clone(), dy.clone()], 3); // (HxWx1x2)
            let steepest_descent_ic_t = bmm::<B>(grad_im2.flatten(0, 1), jacobian_p.flatten(0, 1))
                .reshape(Shape::new([h, w, 8]));

            let flat_steepest_descent_ic_t: Tensor<B, 2> =
                steepest_descent_ic_t.clone().flatten(0, 1); // (Nx8)

            let hessian = flat_steepest_descent_ic_t
                .clone()
                .transpose()
                .matmul(flat_steepest_descent_ic_t); // (8x8)

            // (HxWx8), (8x8)
            (steepest_descent_ic_t, hessian)
        }
        _ => {
            return Err(anyhow!(
                "Mapping type {:?} not supported!",
                init_mapping.kind
            ))
        }
    };
    // Ewwww, I know. See: https://github.com/tracel-ai/burn/issues/1538
    let hessian_inv = array_to_tensor::<B, 2>(
        tensor_to_array(hessian)
            .into_dimensionality::<Ix2>()
            .expect("Matrix should be 2-dimensional")
            .inv()?
            .into_dyn(),
        device,
    );

    // Tracking variables
    let pbar = get_pbar(max_iters.unwrap_or(250) as usize, message);
    let mut params_history = vec![];
    params_history.push(params.clone());
    let mut dps: VecDeque<_> = VecDeque::with_capacity(patience.unwrap_or(10));

    // Main optimization loop
    for i in 0..max_iters.unwrap_or(250) {
        pbar.set_position(i as u64);

        // Create mapping from params and use it to warp img1 into img2's coordinate frame
        let mapping = Mapping::<B>::from_params(params).to_device(device);
        mapping.warp_tensor3_into(im1_gray.clone(), &mut warped_im1gray, &mut valid, None);

        let (warped_im1_vals, weights) = if has_weights {
            // Extract warped img1 and warped weights, and zero out weights where not valid
            let warped_im1_vals = B::float_slice(warped_im1gray.clone(), [0..h, 0..w, 0..c]);
            let weights = B::float_slice(warped_im1gray.clone(), [0..h, 0..w, c..c + 1]);
            let weights = B::float_reshape(weights, Shape::new([h, w]));
            let weights = B::float_mask_fill(weights, valid.clone(), (0.0).elem());
            (warped_im1_vals, weights)
        } else {
            (warped_im1gray.clone(), B::bool_into_float(valid.clone()))
        };

        // Calculate parameter update dp
        // HWP = HWP * (HW1 - HW1) * HW1
        let errs = steepest_descent_ic_t.clone()
            * (Tensor::from_primitive(warped_im1_vals) - im2_gray.clone())
            * Tensor::from_primitive(weights).unsqueeze_dim(2);
        let dp: Tensor<B, 2> = hessian_inv.clone().matmul(
            errs.sum_dim(0)
                .sum_dim(1)
                .reshape(Shape::new([num_params, 1])),
        );
        let mapping_dp = Mapping::<B>::from_params(dp.clone().into_data().convert().value);

        // Update the parameters
        params = Mapping::<B>::from_tensor(
            mapping.mat.matmul(mapping_dp.inverse().mat),
            init_mapping.kind,
        )
        .get_params();
        params_history.push(params.clone());

        // Push back dp update, pop old one if deque is full
        if i >= patience.unwrap_or(10) as i32 {
            dps.pop_front();
        }
        dps.push_back(dp.clone());

        // Early exit if average dp is small
        let avg_dp: Tensor<B, 2> = Tensor::stack::<3>(dps.clone().into(), 2)
            .mean_dim(2)
            .squeeze(2);

        if avg_dp.abs().max().into_scalar().to_f32().unwrap() <= stop_early.unwrap_or(1e-3) {
            break;
        }
    }
    pbar.finish_and_clear();
    Ok((Mapping::from_params(params).inverse(), params_history))
}

#[allow(clippy::type_complexity)]
#[allow(clippy::too_many_arguments)]
pub fn hierarchical_iclk<P, B>(
    im1: &Image<P>,
    im2: &Image<P>,
    init_mapping: Mapping<B>,
    im1_weights: Option<&GrayImage>,
    max_iters: Option<i32>,
    min_dimensions: (u32, u32),
    max_levels: u32,
    stop_early: Option<f32>,
    patience: Option<usize>,
    message: bool,
) -> Result<(Mapping<B>, HashMap<u32, Vec<Vec<f32>>>)>
where
    P: Pixel + Send + Sync,
    <P as Pixel>::Subpixel: Send + Sync + 'static,
    <P as Pixel>::Subpixel: ValueInto<f32> + From<u8> + Clamp<f32>,
    f32: ValueFrom<<P as Pixel>::Subpixel> + From<<P as Pixel>::Subpixel>,
    B: Backend,
{
    let device = &init_mapping.device();
    let im1_gray = grayscale(im1);
    let im2_gray = grayscale(im2);

    let stack_im1 = img_pyramid(&im1_gray, min_dimensions, max_levels);
    let stack_im2 = img_pyramid(&im2_gray, min_dimensions, max_levels);

    let num_lvls = stack_im1.len().min(stack_im2.len());
    let mut mapping = init_mapping;
    let mut all_params_history = HashMap::new();

    let stack_weights = im1_weights.map_or(vec![None; num_lvls], |weights| {
        img_pyramid(weights, min_dimensions, max_levels)
            .into_iter()
            .map(Some)
            .collect()
    });

    for (i, (im1, im2, weights)) in izip!(
        stack_im1[..num_lvls].iter().rev(),
        stack_im2[..num_lvls].iter().rev(),
        stack_weights[..num_lvls].iter().rev()
    )
    .enumerate()
    {
        // Compute mapping at lowest resolution first and double resolution it at each iteration
        let current_scale = (1 << (num_lvls - i - 1)) as f32;

        // Perform optimization at lvl
        let params_history;
        let msg = format!("Matching scale 1/{:}", &current_scale);
        let msg = if message { Some(msg) } else { None };
        let im1 = image_to_tensor3(im1.clone(), device);
        let im2 = image_to_tensor3(im2.clone(), device);
        let im2_grad = tensor_gradients(im2.clone());
        let weights = weights
            .as_ref()
            .map(|w| image_to_tensor3::<Luma<u8>, B>(w.clone(), device).squeeze(2));

        (mapping, params_history) = iclk_grayscale(
            im1,
            im2,
            im2_grad,
            mapping,
            weights,
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
pub fn pairwise_iclk<B: Backend>(
    frames: &Vec<GrayImage>,
    init_mappings: &[Mapping<B>],
    iterations: i32,
    early_stop: f32,
    patience: usize,
    message: Option<&str>,
) -> Result<Vec<Mapping<B>>> {
    let device = &init_mappings[0].device();
    let pbar = get_pbar(frames.len() - 1, message);

    // Iterate over sliding window of pairwise frames (in parallel!)
    let mappings: Vec<Mapping<B>> = frames
        .par_windows(2)
        .zip(init_mappings)
        .map(|(window, init_mapping)| {
            pbar.inc(1);

            let im1 = image_to_tensor3(window[0].clone(), device);
            let im2 = image_to_tensor3(window[1].clone(), device);
            let im2_grad = tensor_gradients(im2.clone());

            iclk_grayscale(
                im1,
                im2,
                im2_grad,
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
