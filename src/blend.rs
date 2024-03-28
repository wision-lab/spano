use anyhow::{anyhow, Result};
use cached::proc_macro::cached;
use image::Pixel;
use imageproc::definitions::{Clamp, Image};
use itertools::Itertools;
use ndarray::{
    azip, concatenate, s, stack, Array, Array1, Array2, Array3, ArrayBase, Axis, Ix3, NewAxis,
    RawData,
};
use photoncube2video::transforms::{array3_to_image, ref_image_to_array3};
use rayon::iter::{IntoParallelIterator, ParallelIterator};

use crate::warps::Mapping;

/// Computes normalized and clipped distance transform (bwdist) for rectangle that fills image
/// The MATLAB version of this uses a different algorithm, which is much faster but less general.
/// Here, we simply cache the result instead.
///
/// References:
///     Breu, Heinz, Joseph Gil, David Kirkpatrick, and Michael Werman, "Linear Time Euclidean
///     Distance Transform Algorithms," IEEE Transactions on Pattern Analysis and Machine
///     Intelligence, Vol. 17, No. 5, May 1995, pp. 529-533.
#[cached(sync_writes = true)]
pub fn distance_transform(size: (usize, usize)) -> Array2<f32> {
    polygon_distance_transform(&Mapping::identity().corners(size), size)

    // let (w, h) = size;
    // let corners = vec![(0.0, 0.0), (w as f32, 0.0), (w as f32, h as f32), (0.0, h as f32)];
    // let dist = |q| corners.iter().circular_tuple_windows().map(|(p1, p2)| distance_to_line(*p1, *p2, q)).fold(f32::INFINITY, |a, b| a.min(b));
    // let max_dist = dist((w as f32 / 2.0, h as f32 / 2.0));

    // Array::from_shape_fn((h, w), |(i, j)| {
    //     dist((j as f32, i as f32)) / max_dist
    // })
}

/// Find distance to line defined by two points
/// See: https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line
pub fn distance_to_line(p1: (f32, f32), p2: (f32, f32), query: (f32, f32)) -> f32 {
    let (x1, y1) = p1;
    let (x2, y2) = p2;
    let (x0, y0) = query;

    ((x2 - x1) * (y1 - y0) - (x1 - x0) * (y2 - y1)).abs()
        / ((x2 - x1).powf(2.0) + (y2 - y1).powf(2.0)).sqrt()
}

/// Computes normalized and clipped distance transform (bwdist) for arbitray polygon
pub fn polygon_distance_transform(corners: &Array2<f32>, size: (usize, usize)) -> Array2<f32> {
    // Points is a Nx2 array of xy pairs
    let (width, height) = size;
    let points = Array::from_shape_fn((width * height, 2), |(i, j)| {
        if j == 0 {
            (i % width) as f32
        } else {
            (i / width) as f32
        }
    });
    let mut weights = -polygon_sdf(&points, corners);
    let max = weights.fold(-f32::INFINITY, |a, b| a.max(*b));
    weights.mapv_inplace(|v| v / max);
    weights.into_shape((height, width)).unwrap()
}

/// Akin to the distance transform used by opencv or bwdist in MATLB but much more general.
pub fn polygon_sdf(points: &Array2<f32>, vertices: &Array2<f32>) -> Array1<f32> {
    // Adapted from: https://www.shadertoy.com/view/wdBXRW

    let num_points = points.shape()[0];
    let num_vertices = vertices.shape()[0];

    let mut d: Array1<f32> = (points - vertices.slice(s![0, ..]).to_owned())
        .mapv(|v| v * v)
        .sum_axis(Axis(1));
    let mut s = Array1::<f32>::ones(num_points);

    for i in 0..num_vertices {
        // distances
        let j = if i == 0 { num_vertices - 1 } else { i - 1 };
        let e = vertices.slice(s![j, ..]).to_owned() - vertices.slice(s![i, ..]).to_owned();
        let w = points - vertices.slice(s![i, ..]).to_owned();

        let mut weights = (&w * &e).sum_axis(Axis(1)) / (e.dot(&e));
        weights.mapv_inplace(|v| v.clamp(0.0, 1.0));

        let ew = stack![Axis(0), &weights * e[0], &weights * e[1]];
        let b = &w - &ew.t();
        azip!((di in &mut d, bi in b.rows()) *di = di.min(bi.dot(&bi)));

        // winding number from http://geomalgorithms.com/a03-_inclusion.html
        let cond = Array1::from_vec(
            (
                points
                    .slice(s![.., 1])
                    .mapv(|v| v >= vertices[(i, 1)])
                    .to_vec(),
                points
                    .slice(s![.., 1])
                    .mapv(|v| v < vertices[(j, 1)])
                    .to_vec(),
                (
                    (&e.slice(s![0]).to_owned() * &w.slice(s![.., 1]).to_owned()).to_vec(),
                    (&e.slice(s![1]).to_owned() * &w.slice(s![.., 0]).to_owned()).to_vec(),
                )
                    .into_par_iter()
                    .map(|(a, b)| a > b)
                    .collect::<Vec<_>>(),
            )
                .into_par_iter()
                .map(|(c1, c2, c3)| (!c1 & !c2 & !c3) | (c1 & c2 & c3))
                .collect(),
        );
        azip!((si in &mut s, ci in &cond) *si = if *ci {-*si} else {*si});
    }

    s * d.mapv(|v| v.sqrt())
}

/// Merge frames using simple linear blending
/// If size (height, width) is specified, that will be used as the canvas size,
/// otherwise, find smallest canvas size that fits all warps.
pub fn merge_arrays<S>(
    mappings: &[Mapping],
    frames: &[ArrayBase<S, Ix3>],
    size: Option<(usize, usize)>,
) -> Result<Array3<f32>>
where
    S: RawData<Elem = f32> + ndarray::Data,
{
    let [frame_size] = frames
        .iter()
        .map(|f| f.dim())
        .unique()
        .collect::<Vec<(usize, usize, usize)>>()[..]
    else {
        return Err(anyhow!("All frames must have same size."));
    };
    let (h, w, c) = frame_size;

    let ((canvas_h, canvas_w), offset) = if let Some(val) = size {
        (val, Mapping::identity())
    } else {
        let (extent, offset) = Mapping::maximum_extent(mappings, &[(w, h)]);
        let (canvas_w, canvas_h) = extent
            .iter()
            .collect_tuple()
            .expect("Canvas should have width and height");
        ((canvas_h.ceil() as usize, canvas_w.ceil() as usize), offset)
    };

    let mut canvas: Array3<f32> = Array3::zeros((canvas_h, canvas_w, (c + 1)));
    let mut valid: Array2<bool> = Array2::from_elem((canvas_h, canvas_w), false);
    let weights = distance_transform((w, h));
    let weights = weights.slice(s![.., .., NewAxis]);
    let merge = |dst: &mut [f32], src: &[f32]| {
        dst[0] += src[0] * src[1];
        dst[1] += src[1];
    };

    for (frame, map) in frames.iter().zip(mappings) {
        let frame = concatenate(Axis(2), &[frame.view(), weights.view()])?;
        map.transform(None, Some(offset.clone()))
            .warp_array3_into::<_, f32>(
                &frame.as_standard_layout(),
                &mut canvas,
                &mut valid,
                None,
                None,
                Some(merge),
            );
    }

    let canvas = canvas.slice(s![.., .., ..c]).to_owned() / canvas.slice(s![.., .., -1..]);
    Ok(canvas)
}

/// Wrapper for `merge_arrays` that converts to/from images.
pub fn merge_images<P>(
    mappings: &[Mapping],
    frames: &[Image<P>],
    size: Option<(usize, usize)>,
) -> Result<Image<P>>
where
    P: Pixel + Send + Sync,
    f32: From<<P as Pixel>::Subpixel>,
    <P as Pixel>::Subpixel: Clamp<f32>,
{
    let frames: Vec<_> = frames
        .iter()
        .map(|f| ref_image_to_array3(f).mapv(f32::from))
        .collect();
    let merged = merge_arrays(mappings, &frames[..], size.map(|(w, h)| (h, w)))?;
    Ok(array3_to_image(merged.mapv(<P as Pixel>::Subpixel::clamp)))
}
