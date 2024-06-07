use std::hash::Hash;

use anyhow::{anyhow, Result};
use burn_tensor::{Shape, Tensor};
use image::Pixel;
use imageproc::definitions::{Clamp, Image};
use itertools::Itertools;
use num_traits::ToPrimitive;

use crate::{
    kernels::Backend,
    transforms::{image_to_tensor3, tensor3_to_image},
    warps::Mapping,
};

/// Computes normalized and clipped distance transform (bwdist) for rectangle that fills image.
// #[cached(sync_writes = true)]
pub fn distance_transform<B: Backend, const D: usize>(
    shape: Shape<D>,
    device: &B::Device,
) -> Tensor<B, 3> {
    // Assume shape is HW*
    let [h, w] = shape.dims[..2] else {
        panic!("Shape should have at least two dimensions.")
    };
    let corners = [
        (0.0, 0.0),
        (w as f32, 0.0),
        (w as f32, h as f32),
        (0.0, h as f32),
    ];
    let num_points = (w as i64) * (h as i64);
    // TODO: Use modulo here once it's available. See: https://github.com/tracel-ai/burn/pull/1726
    let points: Tensor<B, 2> = Tensor::stack(
        vec![
            // Tensor::cat(vec![Tensor::arange(0..(w as i64), device); h], 0), // ~80ms
            // Tensor::from_floats(&(0..num_points).map(|i| (i as f32) % (w as f32)).collect::<Vec<_>>()[..], device), // ~ 18ms
            Tensor::arange(0..(w as i64), device)
                .unsqueeze_dim::<2>(0)
                .repeat(0, h)
                .flatten(0, 1)
                .float(), // ~10ms
            Tensor::arange(0..num_points, device)
                .div_scalar(w as u32)
                .float(),
        ],
        1,
    );
    let dist = |q: Tensor<B, 2>| {
        corners
            .iter()
            .circular_tuple_windows()
            .map(|(p1, p2)| distance_to_line(*p1, *p2, q.clone()))
            .fold(Tensor::full([h * w, 1], f32::INFINITY, device), |a, b| {
                a.min_pair(b)
            })
    };
    dist(points).reshape([h, w, 1])
}

/// Find distance to line defined by two points
/// See: https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line
pub fn distance_to_line<B: Backend>(
    p1: (f32, f32),
    p2: (f32, f32),
    query: Tensor<B, 2>,
) -> Tensor<B, 2> {
    let (x1, y1) = p1;
    let (x2, y2) = p2;

    let num_points = query.dims()[0];
    let x0 = query.clone().slice([0..num_points, 0..1]);
    let y0 = query.clone().slice([0..num_points, 1..2]);

    let out = ((-y0 + y1) * (x2 - x1) - (-x0 + x1) * (y2 - y1)).abs();
    out / ((x2 - x1).powf(2.0) + (y2 - y1).powf(2.0)).sqrt()
}

/// Merge frames using simple linear blending
/// If size (height, width) is specified, that will be used as the canvas size,
/// otherwise, find smallest canvas size that fits all warps.
pub fn blend_tensors3<B: Backend>(
    mappings: &[Mapping<B>],
    inputs: &[Tensor<B, 3>],
    size: Option<(usize, usize)>,
) -> Result<Tensor<B, 3>>
where
    B::Device: Eq + Hash,
{
    // Validate input shapes
    let [frame_size] = inputs
        .iter()
        .map(|f| f.dims())
        .unique()
        .collect::<Vec<[usize; 3]>>()[..]
    else {
        return Err(anyhow!("All frames must have same size."));
    };
    let [h, w, c] = frame_size;

    // Validate devices
    let unique_devices: Vec<B::Device> = mappings
        .iter()
        .map(|m| m.device())
        .chain(inputs.iter().map(|i| i.device()))
        .unique()
        .collect();
    if unique_devices.len() != 1 {
        return Err(anyhow!("All tensors should be on the same device"));
    }
    let device = &unique_devices[0];

    let ((canvas_h, canvas_w), offset) = if let Some(val) = size {
        (val, Mapping::identity().to_device(device))
    } else {
        let (extent, offset) = Mapping::maximum_extent(mappings, &[(w, h)]);
        let (canvas_w, canvas_h) = extent
            .to_data()
            .value
            .iter()
            .map(|i| i.to_f32().unwrap())
            .collect_tuple()
            .expect("Canvas should have width and height");
        ((canvas_h.ceil() as usize, canvas_w.ceil() as usize), offset)
    };
    let shifted_mappings: Vec<_> = mappings
        .iter()
        .map(|m| m.transform(None, Some(offset.clone())))
        .collect();

    let mut output = B::float_zeros(Shape::new([canvas_h, canvas_w, c + 1]), device);
    let weights = distance_transform::<B, 3>(Shape::new(frame_size), device);
    B::blend_into_tensor3(&shifted_mappings[..], inputs, weights, &mut output);
    let canvas = Tensor::from_primitive(output);
    let merged = canvas.clone().slice([0..canvas_h, 0..canvas_w, 0..c])
        / canvas.slice([0..canvas_h, 0..canvas_w, c..c + 1]);
    Ok(merged)
}

/// Wrapper for `blend_tensors3` that converts to/from images.
pub fn blend_images<P, B>(
    mappings: &[Mapping<B>],
    frames: &[Image<P>],
    size: Option<(usize, usize)>,
) -> Result<Image<P>>
where
    P: Pixel + Send + Sync,
    f32: From<<P as Pixel>::Subpixel>,
    <P as Pixel>::Subpixel: Clamp<f32>,
    B::Device: Eq + Hash,
    B: Backend,
{
    let frames: Vec<_> = frames
        .iter()
        .map(|f| image_to_tensor3(f.clone(), &mappings[0].device()))
        .collect();
    let merged = blend_tensors3(mappings, &frames[..], size.map(|(w, h)| (h, w)))?;
    Ok(tensor3_to_image(merged))
}
