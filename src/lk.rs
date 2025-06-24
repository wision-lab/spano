use std::collections::{HashMap, VecDeque};

use anyhow::{anyhow, Result};
use conv::{ValueFrom, ValueInto};
use image::{GrayImage, Luma, Pixel};
use imageproc::{
    definitions::{Clamp, Image},
    gradients::{HORIZONTAL_PREWITT, VERTICAL_PREWITT},
};
use itertools::izip;
use ndarray::{
    concatenate, par_azip, s, stack, Array, Array1, Array2, Array3, ArrayBase, Axis, NewAxis,
    RawData,
};
use ndarray_linalg::solve::Inverse;
use ndarray_ndimage::{correlate, BorderMode};
use numpy::{
    Element, Ix3, PyArrayDyn, PyArrayMethods, PyUntypedArray, PyUntypedArrayMethods, ToPyArray,
};
use photoncube2video::{signals::DeferredSignal, transforms::ref_image_to_array3};
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::{
    utils::get_pbar,
    warps::{Mapping, TransformationType},
};

/// Compute image gradients using Prewitt operator
/// Returned (dx, dy) pair as HxWxC arrays.
pub fn gradients<S>(arr: &ArrayBase<S, Ix3>) -> (Array3<f32>, Array3<f32>)
where
    S: RawData<Elem = f32> + ndarray::Data,
{
    let dx = correlate(
        &arr.view(),
        &Array3::from_shape_vec((3, 3, 1), HORIZONTAL_PREWITT.to_vec())
            .expect("Filter should contain 9 elements.")
            .mapv(|v| v as f32)
            .view(),
        BorderMode::Reflect,
        0,
    );
    let dy = correlate(
        &arr.view(),
        &Array3::from_shape_vec((3, 3, 1), VERTICAL_PREWITT.to_vec())
            .expect("Filter should contain 9 elements.")
            .mapv(|v| v as f32)
            .view(),
        BorderMode::Reflect,
        0,
    );
    (dx, dy)
}

/// Estimate the warp that maps `img2` to `img1` using the Pyramidal (or multi-level)
/// Inverse Compositional Lucas-Kanade algorithm. This is akin to calling `iclk` on each level
/// of `img_pyramid`.
/// See `iclk_py` for more details.
#[allow(clippy::type_complexity)]
#[allow(clippy::too_many_arguments)]
pub fn iclk<P>(
    im1: &Image<P>,
    im2: &Image<P>,
    init_mapping: Mapping,
    im1_weights: Option<&GrayImage>,
    multi: bool,
    max_iters: Option<u32>,
    min_dimension: Option<usize>,
    max_levels: Option<u32>,
    stop_early: Option<f32>,
    patience: Option<u32>,
    message: bool,
) -> Result<(Mapping, HashMap<u32, Vec<Vec<f32>>>)>
where
    P: Pixel + Send + Sync,
    <P as Pixel>::Subpixel: Send + Sync + 'static,
    <P as Pixel>::Subpixel: ValueInto<f32> + From<u8> + Clamp<f32>,
    f32: ValueFrom<<P as Pixel>::Subpixel> + From<<P as Pixel>::Subpixel>,
{
    let im1 = ref_image_to_array3(im1).mapv(f32::from).to_owned();
    let im2 = ref_image_to_array3(im2).mapv(f32::from).to_owned();
    let weights = im1_weights.as_ref().map(|w| {
        ref_image_to_array3::<Luma<u8>>(w)
            .mapv(|v| v as f32 / 255.0)
            .to_owned()
    });

    iclk_array(
        &im1,
        &im2,
        init_mapping,
        weights.as_ref(),
        multi,
        max_iters,
        min_dimension,
        max_levels,
        stop_early,
        patience,
        message,
    )
}

/// See `iclk_py` for more details.
#[allow(clippy::type_complexity)]
#[allow(clippy::too_many_arguments)]
pub fn iclk_array<S>(
    im1: &ArrayBase<S, Ix3>,
    im2: &ArrayBase<S, Ix3>,
    init_mapping: Mapping,
    im1_weights: Option<&ArrayBase<S, Ix3>>,
    multi: bool,
    max_iters: Option<u32>,
    min_dimension: Option<usize>,
    max_levels: Option<u32>,
    stop_early: Option<f32>,
    patience: Option<u32>,
    message: bool,
) -> Result<(Mapping, HashMap<u32, Vec<Vec<f32>>>)>
where
    S: RawData<Elem = f32> + ndarray::Data + Sync,
{
    let mut all_params_history = HashMap::new();

    // Early out with single scale matching
    if !multi {
        let msg = if message { Some("Matching") } else { None };
        let (mapping, params_history) = _iclk_single(
            im1,
            im2,
            init_mapping,
            im1_weights,
            max_iters,
            stop_early,
            patience,
            msg,
        )?;
        all_params_history.insert(1, params_history);
        return Ok((mapping, all_params_history));
    }

    // Perform multi-scale matching
    let min_dimension = min_dimension.unwrap_or(16);
    let min_dimensions = (min_dimension, min_dimension);
    let max_levels = max_levels.unwrap_or(8);
    let stack_im1 = img_pyramid(im1, min_dimensions, max_levels);
    let stack_im2 = img_pyramid(im2, min_dimensions, max_levels);

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

        (mapping, params_history) = _iclk_single(
            im1,
            im2,
            mapping,
            weights.as_ref(),
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

#[allow(clippy::too_many_arguments)]
fn _iclk_single<S>(
    im1: &ArrayBase<S, Ix3>,
    im2: &ArrayBase<S, Ix3>,
    init_mapping: Mapping,
    im1_weights: Option<&ArrayBase<S, Ix3>>,
    max_iters: Option<u32>,
    stop_early: Option<f32>,
    patience: Option<u32>,
    message: Option<&str>,
) -> Result<(Mapping, Vec<Vec<f32>>)>
where
    S: RawData<Elem = f32> + ndarray::Data + Sync,
{
    // Initialize values
    let mut params = init_mapping.inverse().get_params();
    let (h, w, c) = im2.dim();
    let num_points = w * h;
    let num_params = params.len();

    let points = Array::from_shape_fn((num_points, 2), |(i, j)| if j == 0 { i % w } else { i / w });
    let xs = points.column(0).mapv(|v| v as f32);
    let ys = points.column(1).mapv(|v| v as f32);
    let mut img1_array_with_weights;

    let (img1_array, has_weights) = if let Some(weights) = im1_weights {
        // Concatenate weights as last channel of img1_array if has_weights
        img1_array_with_weights = concatenate(Axis(2), &[im1.view(), weights.view()])?;
        img1_array_with_weights = img1_array_with_weights.as_standard_layout().to_owned();
        (img1_array_with_weights.view(), true)
    } else {
        (im1.view(), false)
    };
    let mut warped_im1gray_pixels = Array2::<f32>::zeros((num_points, c + has_weights as usize));
    let (dx, dy) = gradients(im2);
    let img2_pixels = im2.view().into_shape((num_points, c))?;
    let mut valid = Array1::<bool>::from_elem(num_points, false);

    // These can be cached, so we init them before the main loop
    let grad_im2 = stack![
        Axis(2),
        dx.view().into_shape((num_points, c))?,
        dy.view().into_shape((num_points, c))?
    ]; // (HWxCx2)
    let ones: Array1<f32> = ArrayBase::ones(num_points);
    let zeros: Array1<f32> = ArrayBase::zeros(num_points);

    let steepest_descent_ic = match init_mapping.kind {
        TransformationType::Identity => return Ok((Mapping::identity(None), vec![vec![]])),
        TransformationType::Translational => {
            // Jacobian is the identity, so just return the gradients
            grad_im2
        }
        TransformationType::Homothety => {
            let mut steepest_descent_ic = Array3::zeros((num_points, c, 3));

            let jacobian_p = stack![
                Axis(1),
                stack![Axis(1), ones, zeros, xs],
                stack![Axis(1), zeros, ones, ys]
            ]; // (HWx2xN)

            // Perform the batch matrix multiply of grad_im2 @ jacobian_p
            par_azip!(
                (
                    mut v in steepest_descent_ic.axis_iter_mut(Axis(0)),
                    a in grad_im2.axis_iter(Axis(0)),
                    b in jacobian_p.axis_iter(Axis(0))
                )
                {v.assign(&a.dot(&b))}
            );
            steepest_descent_ic
        }
        TransformationType::Similarity => {
            let mut steepest_descent_ic = Array3::zeros((num_points, c, 4));

            let jacobian_p = stack![
                Axis(1),
                stack![Axis(1), ones, zeros, xs, -ys.clone()],
                stack![Axis(1), zeros, ones, ys, xs]
            ]; // (HWx2xN)

            // Perform the batch matrix multiply of grad_im2 @ jacobian_p
            par_azip!(
                (
                    mut v in steepest_descent_ic.axis_iter_mut(Axis(0)),
                    a in grad_im2.axis_iter(Axis(0)),
                    b in jacobian_p.axis_iter(Axis(0))
                )
                {v.assign(&a.dot(&b))}
            );
            steepest_descent_ic
        }
        TransformationType::Affine => {
            let mut steepest_descent_ic = Array3::zeros((num_points, c, 6));

            let jacobian_p = stack![
                Axis(1),
                stack![Axis(1), xs, zeros, ys, zeros, ones, zeros],
                stack![Axis(1), zeros, xs, zeros, ys, zeros, ones]
            ]; // (HWx2xN)

            // Perform the batch matrix multiply of grad_im2 @ jacobian_p
            par_azip!(
                (
                    mut v in steepest_descent_ic.axis_iter_mut(Axis(0)),
                    a in grad_im2.axis_iter(Axis(0)),
                    b in jacobian_p.axis_iter(Axis(0))
                )
                {v.assign(&a.dot(&b))}
            );
            steepest_descent_ic
        }
        TransformationType::Projective => {
            let mut steepest_descent_ic = Array3::zeros((num_points, c, 8));

            // dW_dp evaluated at (x, p) and p=0 (identity transform)
            let jacobian_p = {
                let minus_xx: Vec<f32> = xs.iter().map(|i1| -i1 * i1).collect();
                let minus_yy: Vec<f32> = ys.iter().map(|i1| -i1 * i1).collect();
                let minus_xy: Vec<f32> = xs.iter().zip(&ys).map(|(i1, i2)| -i1 * i2).collect();
                stack![
                    Axis(1),
                    stack![
                        Axis(1),
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
                        Axis(1),
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
            }; // (HWx2xN)

            // Perform the batch matrix multiply of grad_im2 @ jacobian_p
            par_azip!(
                (
                    mut v in steepest_descent_ic.axis_iter_mut(Axis(0)),
                    a in grad_im2.axis_iter(Axis(0)),
                    b in jacobian_p.axis_iter(Axis(0))
                )
                {v.assign(&a.dot(&b))}
            );
            steepest_descent_ic
        }
        TransformationType::Unknown => {
            return Err(anyhow!(
                "Mapping type {:?} not supported!",
                init_mapping.kind
            ))
        }
    };
    let hessian: Array2<f32> = steepest_descent_ic
        .axis_iter(Axis(0))
        .map(|cn| cn.t().dot(&cn))
        .fold(Array2::<f32>::zeros((num_params, num_params)), |a, b| a + b);
    let hessian_inv = hessian.inv().unwrap();
    let steepest_descent_ic_t = steepest_descent_ic.permuted_axes([0, 2, 1]);

    // Tracking variables
    let pbar = get_pbar(max_iters.unwrap_or(250) as usize, message);
    let mut params_history = vec![];
    params_history.push(params.clone());
    let mut dps: VecDeque<Array2<f32>> = VecDeque::with_capacity(patience.unwrap_or(10) as usize);

    // Main optimization loop
    for i in 0..max_iters.unwrap_or(250) {
        pbar.set_position(i as u64);

        // Create mapping from params and use it to sample points from img1
        // TODO: Warp with background or without?
        let mapping = Mapping::from_params(params);
        mapping.warp_array3_into::<f32, _, _, _, _, _>(
            &img1_array,
            &mut warped_im1gray_pixels,
            &mut valid,
            &points,
            None,
            None,
        );

        // Calculate parameter update dp
        let sd_param_updates = (
            steepest_descent_ic_t.axis_iter(Axis(0)),
            warped_im1gray_pixels.axis_iter(Axis(0)),
            img2_pixels.axis_iter(Axis(0)),
            valid.as_slice().unwrap().par_iter(),
        )
            // Zip together all three iterators and valid flag
            .into_par_iter()
            // Drop them if the warped value from is out-of-bounds
            // Calculate parameter update according to formula
            .filter_map(|(sd, p1, p2, &is_valid)| {
                if is_valid {
                    let weight = if has_weights { p1[c] } else { 1.0 };
                    let diff = p1.slice(s![..c]).to_owned() - p2.to_owned();
                    Some(sd.dot(&diff.slice(s![.., NewAxis])) * weight)
                } else {
                    None
                }
            })
            // Sum them together, here we use reduce with a base value of zero
            .reduce(|| Array2::<f32>::zeros((num_params, 1)), |a, b| a + b);
        let dp: Array2<f32> = hessian_inv.dot(&sd_param_updates);
        let mapping_dp = Mapping::from_params(dp.clone().into_raw_vec());

        // Update the parameters
        params = Mapping::from_matrix(mapping.mat.dot(&mapping_dp.mat.inv()?), init_mapping.kind)
            .get_params();
        params_history.push(params.clone());

        // Push back dp update, pop old one if deque is full
        if i >= patience.unwrap_or(10) {
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

/// Estimate pairwise registration using iclk
#[allow(clippy::type_complexity)]
#[allow(clippy::too_many_arguments)]
pub fn pairwise_iclk<S>(
    frames: &[ArrayBase<S, Ix3>],
    init_mappings: &[Mapping],
    multi: bool,
    max_iters: Option<u32>,
    min_dimension: Option<usize>,
    max_levels: Option<u32>,
    stop_early: Option<f32>,
    patience: Option<u32>,
    message: bool,
) -> Result<(Vec<Mapping>, Vec<HashMap<u32, Vec<Vec<f32>>>>)>
where
    S: RawData<Elem = f32> + ndarray::Data + Sync,
{
    let msg = if message { Some("Matching") } else { None };
    let pbar = get_pbar(frames.len() - 1, msg);

    // Iterate over sliding window of pairwise frames (in parallel!)
    let (mappings, hists) = frames
        .par_windows(2)
        .zip(init_mappings)
        .map(|(window, init_mapping)| {
            pbar.inc(1);
            iclk_array(
                &window[0],
                &window[1],
                init_mapping.clone(),
                None,
                multi,
                max_iters,
                min_dimension,
                max_levels,
                stop_early,
                patience,
                false,
            )
        })
        // Collect to force reorder
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .unzip();

    // Return raw pairwise warps (N-1 in total)
    pbar.finish_and_clear();
    Ok((mappings, hists))
}

/// Given an image, return an image pyramid with the largest size first and halving the size
/// every time until either the max-levels are reached or the minimum size is reached.
pub fn img_pyramid<S>(
    im: &ArrayBase<S, Ix3>,
    min_dimensions: (usize, usize),
    max_levels: u32,
) -> Vec<Array3<f32>>
where
    S: RawData<Elem = f32> + ndarray::Data,
{
    let mut stack = vec![im.to_owned()];
    let (min_width, min_height) = min_dimensions;
    let (mut h, mut w, _c) = im.dim();

    for _ in 0..max_levels {
        if w >= min_width * 2 && h >= min_height * 2 {
            (h, w) = (h / 2, w / 2);

            let resized = (stack[stack.len() - 1]
                .slice(s![0..h*2;2, 0..w*2;2, ..])
                .to_owned()
                + stack[stack.len() - 1].slice(s![0..h*2;2, 1..w*2;2, ..])
                + stack[stack.len() - 1].slice(s![1..h*2;2, 0..w*2;2, ..])
                + stack[stack.len() - 1].slice(s![1..h*2;2, 1..w*2;2, ..]))
                / 4.0;
            stack.push(resized);
        }
    }
    stack
}

// --------------------------------------------------------------- Python Interface ---------------------------------------------------------------
pub fn pyarray_cast<'py, T: Element>(
    im: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyArrayDyn<T>>> {
    // See <https://github.com/PyO3/rust-numpy/issues/246>
    let im = if let Ok(im) = im.downcast::<PyArrayDyn<f64>>() {
        im.cast::<T>(im.is_fortran_contiguous())?
    } else if let Ok(im) = im.downcast::<PyArrayDyn<f32>>() {
        im.cast::<T>(im.is_fortran_contiguous())?
    } else if let Ok(im) = im.downcast::<PyArrayDyn<i64>>() {
        im.cast::<T>(im.is_fortran_contiguous())?
    } else if let Ok(im) = im.downcast::<PyArrayDyn<i32>>() {
        im.cast::<T>(im.is_fortran_contiguous())?
    } else if let Ok(im) = im.downcast::<PyArrayDyn<i16>>() {
        im.cast::<T>(im.is_fortran_contiguous())?
    } else if let Ok(im) = im.downcast::<PyArrayDyn<i8>>() {
        im.cast::<T>(im.is_fortran_contiguous())?
    } else if let Ok(im) = im.downcast::<PyArrayDyn<u64>>() {
        im.cast::<T>(im.is_fortran_contiguous())?
    } else if let Ok(im) = im.downcast::<PyArrayDyn<u32>>() {
        im.cast::<T>(im.is_fortran_contiguous())?
    } else if let Ok(im) = im.downcast::<PyArrayDyn<u16>>() {
        im.cast::<T>(im.is_fortran_contiguous())?
    } else if let Ok(im) = im.downcast::<PyArrayDyn<u8>>() {
        im.cast::<T>(im.is_fortran_contiguous())?
    } else if let Ok(im) = im.downcast::<PyArrayDyn<bool>>() {
        im.cast::<T>(im.is_fortran_contiguous())?
    } else if let Ok(im) = im.downcast::<PyUntypedArray>() {
        return Err(anyhow!("Array dtype not understood, expected one of [f64, f32, i64, i32, i16, i8, u64, u32, u16, u8, bool], got {}", im.dtype()).into());
    } else {
        return Err(anyhow!("Input cannot be understood as valid array.").into());
    };
    Ok(im)
}

pub fn pyarray_to_im_bridge<T: Element>(im: &Bound<'_, PyAny>) -> PyResult<Array3<T>> {
    let im = pyarray_cast(im)?.to_owned_array();
    let im = match im.ndim() {
        2 => im.insert_axis(Axis(2)),
        3 => im,
        _ => {
            return Err(anyhow!(
                "Expected image with 2 or 3 dimensions, stored as HWC. Got {:} dims.",
                im.ndim(),
            )
            .into())
        }
    }
    .into_dimensionality()
    .map_err(|e| anyhow!(e))?;
    Ok(im)
}

/// Main iclk routine, which works for an arbitrary number of channels.
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
/// See: <https://www.ri.cmu.edu/pub_files/pub3/baker_simon_2002_3/baker_simon_2002_3.pdf>
#[pyfunction]
#[pyo3(
    name = "iclk",
    signature = (im1, im2, init_mapping=None, im1_weights=None, multi=true, max_iters=250, min_dimension=16, max_levels=8, stop_early=1e-3, patience=10, message=false)
)]
#[allow(clippy::type_complexity)]
#[allow(clippy::too_many_arguments)]
pub fn iclk_py<'py>(
    py: Python<'py>,
    im1: &Bound<'py, PyAny>,
    im2: &Bound<'py, PyAny>,
    init_mapping: Option<Mapping>,
    im1_weights: Option<&Bound<'py, PyAny>>,
    multi: bool,
    max_iters: u32,
    min_dimension: usize,
    max_levels: u32,
    stop_early: f32,
    patience: u32,
    message: bool,
) -> Result<(Mapping, HashMap<u32, Vec<Vec<f32>>>)> {
    let _defer = DeferredSignal::new(py, "SIGINT")?;

    let im1 = pyarray_to_im_bridge(im1)?;
    let im2 = pyarray_to_im_bridge(im2)?;
    let weights = im1_weights.map(|a| pyarray_to_im_bridge(a)).transpose()?;

    iclk_array(
        &im1,
        &im2,
        init_mapping.unwrap_or(Mapping::from_params(vec![0.0; 8])),
        weights.as_ref(),
        multi,
        Some(max_iters),
        Some(min_dimension),
        Some(max_levels),
        Some(stop_early),
        Some(patience),
        message,
    )
}

/// Estimate pairwise registration using iclk
#[pyfunction]
#[pyo3(
    name = "pairwise_iclk",
    signature = (frames, init_mappings=None, multi=true, max_iters=250, min_dimension=16, max_levels=8, stop_early=1e-3, patience=10, message=false)
)]
#[allow(clippy::type_complexity)]
#[allow(clippy::too_many_arguments)]
pub fn pairwise_iclk_py<'py>(
    py: Python<'py>,
    frames: Vec<Bound<'py, PyAny>>,
    init_mappings: Option<Vec<Mapping>>,
    multi: bool,
    max_iters: u32,
    min_dimension: usize,
    max_levels: u32,
    stop_early: f32,
    patience: u32,
    message: bool,
) -> Result<(Vec<Mapping>, Vec<HashMap<u32, Vec<Vec<f32>>>>)> {
    let _defer = DeferredSignal::new(py, "SIGINT")?;

    let frames: Vec<Array3<f32>> = frames
        .iter()
        .map(pyarray_to_im_bridge::<f32>)
        .collect::<Result<Vec<_>, _>>()?;

    pairwise_iclk(
        &frames,
        &init_mappings.unwrap_or(vec![Mapping::from_params(vec![0.0; 8]); frames.len() - 1])[..],
        multi,
        Some(max_iters),
        Some(min_dimension),
        Some(max_levels),
        Some(stop_early),
        Some(patience),
        message,
    )
}

/// Given an image, return an image pyramid with the largest size first and halving the size
/// every time until either the max-levels are reached or the minimum size is reached.
#[pyfunction]
#[pyo3(
    name = "img_pyramid",
    signature = (im, min_dimension=16, max_levels=8),
)]
pub fn img_pyramid_py<'py>(
    py: Python<'py>,
    im: &Bound<'py, PyAny>,
    min_dimension: usize,
    max_levels: u32,
) -> PyResult<Vec<Py<PyAny>>> {
    let _defer = DeferredSignal::new(py, "SIGINT")?;

    Ok(img_pyramid(
        &pyarray_to_im_bridge(im)?,
        (min_dimension, min_dimension),
        max_levels,
    )
    .iter()
    .map(|a| a.to_pyarray_bound(py).to_owned().into_py(py))
    .collect())
}
