use std::ops::DivAssign;

use conv::ValueInto;
use image::{ImageBuffer, Pixel};
use imageproc::definitions::{Clamp, Image};
use ndarray::{array, concatenate, s, Array1, Array2, Axis};
use ndarray::{stack, Array};
use ndarray_interp::{
    interp1d::{CubicSpline, Interp1DBuilder},
    interp2d::{Bilinear, Interp2DBuilder},
};
use ndarray_linalg::solve::Inverse;
use num_traits::AsPrimitive;

#[derive(Copy, Clone, Debug)]
pub enum TransformationType {
    Translational,
    Affine,
    Projective,
    Unknown,
}

impl TransformationType {
    pub fn num_params(&self) -> usize {
        match &self {
            TransformationType::Translational => 2,
            TransformationType::Affine => 6,
            TransformationType::Projective => 8,
            TransformationType::Unknown => 0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Mapping {
    pub mat: Array2<f32>,
    pub is_identity: bool,
    pub kind: TransformationType,
}

impl Mapping {
    /// Return the mapping that trasforms a point using a 3x3 matrix.
    pub fn from_matrix(mat: Array2<f32>, kind: TransformationType) -> Self {
        let is_identity = Array2::<f32>::eye(3).abs_diff_eq(&mat, 1e-8);
        Self {
            mat,
            is_identity,
            kind,
        }
    }

    /// Given a list of transform parameters, return a function that maps a
    /// source point to its destination. The type of mapping depends on the number of params (DoF).
    pub fn from_params(params: &[f32]) -> Self {
        let (full_params, kind) = match &params {
            // Translations
            [dx, dy] => (
                vec![1.0, 0.0, *dx, 0.0, 1.0, *dy, 0.0, 0.0, 1.0],
                TransformationType::Translational,
            ),

            // Affine Transforms
            [p1, p2, p3, p4, p5, p6] => (
                vec![*p1 + 1.0, *p3, *p5, *p2, *p4 + 1.0, *p6, 0.0, 0.0, 1.0],
                TransformationType::Affine,
            ),

            // Projective Transforms
            [p1, p2, p3, p4, p5, p6, p7, p8] => (
                vec![*p1 + 1.0, *p3, *p5, *p2, *p4 + 1.0, *p6, *p7, *p8, 1.0],
                TransformationType::Projective,
            ),
            _ => panic!(),
        };

        let mat = Array2::from_shape_vec((3, 3), full_params).unwrap();
        Self::from_matrix(mat, kind)
    }

    pub fn scale(x: f32, y: f32) -> Self {
        Self::from_params(&[x - 1.0, 0.0, 0.0, y - 1.0, 0.0, 0.0])
    }

    pub fn shift(x: f32, y: f32) -> Self {
        Self::from_params(&[x, y])
    }

    pub fn identity() -> Self {
        Self::from_params(&[0.0, 0.0])
    }

    #[inline]
    pub fn warp_points<T>(&self, points: &Array2<T>) -> Array2<f32>
    where
        T: AsPrimitive<f32> + Copy + 'static,
    {
        let points = points.mapv(|v| v.as_());

        if self.is_identity {
            return points;
        }

        let num_points = points.shape()[0];
        let points = concatenate![Axis(1), points, Array2::ones((num_points, 1))];

        let mut warped_points: Array2<f32> = self.mat.dot(&points.t());
        let d = warped_points.index_axis(Axis(0), 2).mapv(|v| v.max(1e-8));
        warped_points.div_assign(&d);

        warped_points.t().slice(s![.., ..2]).to_owned()
    }

    pub fn get_params(&self) -> Vec<f32> {
        let p = (&self.mat.clone() / self.mat[(2, 2)]).into_raw_vec();
        match &self.kind {
            TransformationType::Translational => vec![p[2], p[5]],
            TransformationType::Affine => vec![p[0] - 1.0, p[3], p[1], p[4] - 1.0, p[2], p[5]],
            TransformationType::Projective => {
                vec![p[0] - 1.0, p[3], p[1], p[4] - 1.0, p[2], p[5], p[6], p[7]]
            }
            _ => panic!("Transformation cannot be unknown!"),
        }
    }

    pub fn get_params_full(&self) -> Vec<f32> {
        let p = (&self.mat.clone() / self.mat[(2, 2)]).into_raw_vec();
        vec![p[0] - 1.0, p[3], p[1], p[4] - 1.0, p[2], p[5], p[6], p[7]]
    }

    pub fn inverse(&self) -> Self {
        Self {
            mat: self.mat.inv().expect("Cannot invert mapping"),
            is_identity: self.is_identity,
            kind: self.kind,
        }
    }

    pub fn transform(&self, lhs: Option<Self>, rhs: Option<Self>) -> Self {
        let (lhs_mat, lhs_id, lhs_kind) = lhs
            .map_or((Array2::eye(3), false, TransformationType::Unknown), |m| {
                (m.mat, m.is_identity, m.kind)
            });

        let (rhs_mat, rhs_id, rhs_kind) = rhs
            .map_or((Array2::eye(3), false, TransformationType::Unknown), |m| {
                (m.mat, m.is_identity, m.kind)
            });

        Mapping {
            mat: lhs_mat.dot(&self.mat).dot(&rhs_mat).to_owned(),
            is_identity: lhs_id & self.is_identity & rhs_id,
            kind: *[lhs_kind, self.kind, rhs_kind]
                .iter()
                .max_by_key(|k| k.num_params())
                .unwrap(),
        }
    }

    pub fn rescale(&self, scale: f32) -> Self {
        self.transform(
            Some(Mapping::scale(1.0 / scale, 1.0 / scale)),
            Some(Mapping::scale(scale, scale)),
        )
    }

    pub fn corners(&self, size: (u32, u32)) -> Array2<f32> {
        let (w, h) = size;
        let corners = array![[0, 0], [w, 0], [w, h], [0, h]];
        self.inverse().warp_points(&corners)
    }

    pub fn extent(&self, size: (u32, u32)) -> (Array1<f32>, Array1<f32>) {
        let corners = self.corners(size);
        let min_coords = corners.map_axis(Axis(0), |view| {
            view.iter().fold(f32::INFINITY, |a, b| a.min(*b))
        });
        let max_coords = corners.map_axis(Axis(0), |view| {
            view.iter().fold(-f32::INFINITY, |a, b| a.max(*b))
        });
        (min_coords, max_coords)
    }

    pub fn maximum_extent(maps: &[Self], size: (u32, u32)) -> (Array1<f32>, Self) {
        let (min_coords, max_coords): (Vec<Array1<f32>>, Vec<Array1<f32>>) =
            maps.iter().map(|m| m.extent(size)).unzip();

        let min_coords: Vec<_> = min_coords.iter().map(|arr| arr.view()).collect();
        let min_coords = stack(Axis(0), &min_coords[..])
            .unwrap()
            .map_axis(Axis(0), |view| {
                view.iter().fold(f32::INFINITY, |a, b| a.min(*b))
            });

        let max_coords: Vec<_> = max_coords.iter().map(|arr| arr.view()).collect();
        let max_coords = stack(Axis(0), &max_coords[..])
            .unwrap()
            .map_axis(Axis(0), |view| {
                view.iter().fold(-f32::INFINITY, |a, b| a.max(*b))
            });

        let extent = max_coords - &min_coords;
        let offset = Mapping::from_params(&min_coords.to_vec()[..]);
        (extent, offset)
    }

    pub fn interpolate_array(ts: Array1<f32>, maps: &Vec<Self>, query: Array1<f32>) -> Vec<Self> {
        let params = Array2::from_shape_vec(
            (maps.len(), 8),
            maps.iter().flat_map(|m| m.get_params_full()).collect(),
        )
        .unwrap();

        let interpolator = Interp1DBuilder::new(params)
            .x(ts)
            .strategy(CubicSpline::new())
            .build()
            .unwrap();

        let interp_params = interpolator.interp_array(&query).unwrap();
        interp_params
            .axis_iter(Axis(0))
            .map(|p| Self::from_params(&p.to_vec()))
            .collect()
    }

    pub fn interpolate_scalar(ts: Array1<f32>, maps: &Vec<Self>, query: f32) -> Self {
        Self::interpolate_array(ts, maps, array![query])
            .into_iter()
            .nth(0)
            .unwrap()
    }
}

pub fn warp_image<P, Fi>(mapping: &Mapping, get_pixel: Fi, width: u32, height: u32) -> Image<P>
where
    P: Pixel,
    <P as Pixel>::Subpixel: Send + Sync,
    <P as Pixel>::Subpixel: ValueInto<f32> + Clamp<f32>,
    Fi: Fn(f32, f32) -> P + Send + Sync,
{
    // Points is a Nx2 array of xy pairs
    let points = Array::from_shape_fn(((width * height) as usize, 2), |(i, j)| {
        if j == 0 {
            i as u32 % width
        } else {
            i as u32 / width
        }
    });

    // Warp all points and determine indices of in-bound ones
    let warpd = mapping.warp_points(&points);
    let xs = warpd.column(0);
    let ys = warpd.column(1);

    // Create and return image buffer
    let pixels = xs.iter().zip(ys).map(|(&x, &y)| get_pixel(x, y));
    let mut out = ImageBuffer::new(width, height);

    // TODO: Maybe vectorize get_pixel?
    for ((_, _, out_p), pix) in out.enumerate_pixels_mut().zip(pixels) {
        *out_p = pix
    }

    out
}

pub fn sample_warped_grid(
    mapping: &Mapping,
    data: Array2<f32>,
    width: usize,
    height: usize,
    ret_valid: bool,
) -> (Array1<f32>, Option<Array1<bool>>) {
    // Points is a Nx2 array of xy pairs
    let points = Array::from_shape_fn((width * height, 2), |(i, j)| {
        if j == 0 {
            (i % width) as u32
        } else {
            (i / width) as u32
        }
    });

    // Warp all points and determine indices of in-bound ones
    let warpd = mapping.warp_points(&points);
    let qx = warpd.column(0);
    let qy = warpd.column(1);

    let valid = if ret_valid {
        let [data_h, data_w] = data.shape() else {
            unreachable!("Data is Array2, should have 2 dims!")
        };
        let in_range_x = |x| 0.0 <= x && x <= (*data_w as f32) - 1.0;
        let in_range_y = |y| 0.0 <= y && y <= (*data_h as f32) - 1.0;
        Some(Array::from_iter(
            qx.iter()
                .zip(qy)
                .map(|(&x, &y)| in_range_x(x) && in_range_y(y)),
        ))
    } else {
        None
    };

    let interpolator = Interp2DBuilder::new(data)
        .strategy(Bilinear::new().allow_default(true))
        .build()
        .unwrap();
    let samples = interpolator.interp_array(&qy, &qx).unwrap();
    (samples, valid)
}

// ----------------------------------------------------------------------------
#[cfg(test)]
mod test_warps {
    use approx::assert_relative_eq;
    use ndarray::array;

    use crate::warps::{Mapping, TransformationType};

    #[test]
    fn test_warp_points() {
        let map = Mapping::from_matrix(
            array![
                [1.13411823, 4.38092511, 9.315785],
                [1.37351153, 5.27648111, 1.60252762],
                [7.76114426, 9.66312177, 2.61286966]
            ],
            TransformationType::Projective,
        );
        let point = array![[0, 0]];
        let warpd = map.warp_points(&point);
        assert_relative_eq!(warpd, array![[3.56534624, 0.61332092]]);
    }
}
