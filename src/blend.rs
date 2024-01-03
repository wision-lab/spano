use conv::ValueInto;
use image::{GenericImageView, Pixel};
use imageproc::{
    definitions::{Clamp, Image},
    math::cast,
};
use ndarray::{azip, s, stack, Array, Array1, Array2, Axis};
use rayon::iter::{IntoParallelIterator, ParallelIterator};

use crate::Mapping;

/// Again, this is almost lifted verbatum from:
///     https://docs.rs/imageproc/0.23.0/src/imageproc/geometric_transformations.rs.html#681
/// But alas, this function is not declared as public so we can't just import it...
pub fn interpolate_bilinear<P>(image: &Image<P>, x: f32, y: f32) -> Option<P>
where
    P: Pixel,
    <P as Pixel>::Subpixel: ValueInto<f32> + Clamp<f32>,
{
    let left = x.floor();
    let right = left + 1f32;
    let top = y.floor();
    let bottom = top + 1f32;

    let right_weight = x - left;
    let bottom_weight = y - top;

    let (width, height) = image.dimensions();

    if right_weight.abs() < 1e-8 && bottom_weight.abs() < 1e-8 {
        // If it's integer, return that pixel
        image
            .get_pixel_checked(x as u32, y as u32)
            .map(|p| p.to_owned())
    } else if left < 0f32 || right >= width as f32 || top < 0f32 || bottom >= height as f32 {
        // None if out of bound
        None
    } else {
        // Do the interpolation
        let (tl, tr, bl, br) = unsafe {
            (
                image.unsafe_get_pixel(left as u32, top as u32),
                image.unsafe_get_pixel(right as u32, top as u32),
                image.unsafe_get_pixel(left as u32, bottom as u32),
                image.unsafe_get_pixel(right as u32, bottom as u32),
            )
        };
        Some(blend_bilinear(tl, tr, bl, br, right_weight, bottom_weight))
    }
}

/// Again, this is almost lifted verbatum from:
///     https://docs.rs/imageproc/0.23.0/src/imageproc/geometric_transformations.rs.html#681
/// But alas, this function is not declared as public so we can't just import it...
pub fn interpolate_bilinear_with_bkg<P>(image: &Image<P>, x: f32, y: f32, background: P) -> P
where
    P: Pixel,
    <P as Pixel>::Subpixel: ValueInto<f32> + Clamp<f32>,
{
    let (width, height) = image.dimensions();

    let get_pix_or_bkg = |x: f32, y: f32| {
        if x < 0f32 || x >= width as f32 || y < 0f32 || y >= height as f32 {
            background
        } else {
            unsafe { image.unsafe_get_pixel(x as u32, y as u32) }
        }
    };

    let left = x.floor();
    let right = left + 1f32;
    let top = y.floor();
    let bottom = top + 1f32;
    let right_weight = x - left;
    let bottom_weight = y - top;

    // Do the interpolation
    let (tl, tr, bl, br) = (
        get_pix_or_bkg(left, top),
        get_pix_or_bkg(right, top),
        get_pix_or_bkg(left, bottom),
        get_pix_or_bkg(right, bottom),
    );
    blend_bilinear(tl, tr, bl, br, right_weight, bottom_weight)
}

/// Again, this is lifted almost verbatum from the imageproc crate...
pub fn blend_bilinear<P>(
    top_left: P,
    top_right: P,
    bottom_left: P,
    bottom_right: P,
    right_weight: f32,
    bottom_weight: f32,
) -> P
where
    P: Pixel,
    P::Subpixel: ValueInto<f32> + Clamp<f32>,
{
    let top = top_left.map2(&top_right, |u, v| {
        P::Subpixel::clamp((1f32 - right_weight) * cast(u) + right_weight * cast(v))
    });

    let bottom = bottom_left.map2(&bottom_right, |u, v| {
        P::Subpixel::clamp((1f32 - right_weight) * cast(u) + right_weight * cast(v))
    });

    top.map2(&bottom, |u, v| {
        P::Subpixel::clamp((1f32 - bottom_weight) * cast(u) + bottom_weight * cast(v))
    })
}

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
