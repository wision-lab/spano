use ndarray::{azip, s, stack, Array, Array1, Array2, Axis};
use rayon::iter::{IntoParallelIterator, ParallelIterator};

use crate::warps::Mapping;

/// Computes normalized and clipped distance transform (bwdist) for rectangle that fills image
pub fn distance_transform(size: (usize, usize)) -> Array2<f32> {
    polygon_distance_transform(&Mapping::identity().corners(size), size)
}

/// Computes normalized and clipped distance transform (bwdist) for arbitray polygon
pub fn polygon_distance_transform(corners: &Array2<f32>, size: (usize, usize)) -> Array2<f32> {
    // Points is a Nx2 array of xy pairs
    let (height, width) = size;
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
    weights.into_shape(size).unwrap()
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
