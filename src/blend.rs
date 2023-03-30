use conv::ValueInto;
use image::{GenericImageView, Pixel};
use imageproc::{
    definitions::{Clamp, Image},
    math::cast,
};
use ndarray::{s, Array1, Array2};

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
        image.get_pixel_checked(x as u32, y as u32).map(|p| p.to_owned())
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

/// Akin to the distance transform used by opencv or bwdist in MATLB but much more general.
#[allow(dead_code)]
pub fn polygon_sdf_vec(vertices: Array2<f32>) -> impl Fn(f32, f32) -> f32 {
    // Adapted from: https://www.shadertoy.com/view/wdBXRW

    move |x: f32, y: f32| {
        let num = vertices.shape()[0];
        let point = Array1::from_vec(vec![x, y]);
        let mut d = (vertices.slice(s![0, ..]).to_owned() - &point)
            .mapv(|v| v * v)
            .sum();
        let mut s = 1.0;

        for i in 0..num {
            let j = if i == 0 { num - 1 } else { i - 1 };
            let e = vertices.slice(s![j, ..]).to_owned() - vertices.slice(s![i, ..]).to_owned();
            let w = &point - vertices.slice(s![i, ..]).to_owned();
            let b = &w - &e * (w.dot(&e) / e.dot(&e)).clamp(0.0, 1.0);
            d = d.min(b.dot(&b));

            let cond: u8 = [
                y >= vertices[(i, 1)],
                y < vertices[(j, 1)],
                e[0] * w[1] > e[1] * w[0],
            ]
            .map(|i| i as u8)
            .iter()
            .sum();

            if (cond == 0) || (cond == 3) {
                s = -s;
            }
        }
        s * d.sqrt()
    }
}
