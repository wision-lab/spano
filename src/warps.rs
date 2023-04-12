use conv::ValueInto;
use image::Pixel;
use imageproc::definitions::{Clamp, Image};
use ndarray::{Array1, Array2};
use ndarray_linalg::solve::Inverse;
use num_traits::AsPrimitive;
use rayon::prelude::*;

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

#[derive(Debug)]
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
        Self::from_params(&[x, 0.0, 0.0, y, 0.0, 0.0])
    }

    pub fn shift(x: f32, y: f32) -> Self {
        Self::from_params(&[x, y])
    }

    #[inline]
    pub fn warpfn<T>(&self) -> impl Fn(&(T, T)) -> (f32, f32) + '_
    where
        T: AsPrimitive<f32> + Copy + 'static,
    {
        move |&(x, y)| self.warp_point((x, y))
    }

    #[inline]
    pub fn warpfn_centered<T>(&self, dim: (u32, u32)) -> impl Fn(&(T, T)) -> (f32, f32) + '_
    where
        T: AsPrimitive<f32> + Copy + 'static,
    {
        let (h, w) = dim;
        move |&(x, y)| self.warp_points_centered((x, y), h as f32, w as f32)
    }

    #[inline]
    pub fn warp_point<T>(&self, p: (T, T)) -> (f32, f32)
    where
        T: AsPrimitive<f32> + Copy + 'static,
    {
        let (x, y) = p;
        let x = x.as_();
        let y = y.as_();

        if self.is_identity {
            return (x, y);
        }

        let point = Array1::from_vec(vec![x, y, 1.0]);
        if let [x_, y_, d] = &self.mat.dot(&point).to_vec()[..] {
            let d = d.max(1e-8);
            (x_ / d, y_ / d)
        } else {
            // This should never occur as long as mat is a 3x3 matrix
            // Maybe we should panic here as a better default?
            (f32::NAN, f32::NAN)
        }
    }

    #[inline]
    pub fn warp_points_centered<T>(&self, p: (T, T), h: f32, w: f32) -> (f32, f32)
    where
        T: AsPrimitive<f32> + Copy + 'static,
    {
        let (x, y) = p;
        let x = x.as_();
        let y = y.as_();

        let x_centered = 2.0 * (x / w) - 1.0;
        let y_centered = 2.0 * (y / h) - 1.0;
        
        // let x_centered = (x + 1.0) * w / 2.0;
        // let y_centered = (y + 1.0) * h / 2.0;

        let (xp, yp) = self.warp_point((x_centered, y_centered));

        ((xp + 1.0) * w / 2.0, (yp + 1.0) * h / 2.0)

        // (2.0 * (xp / w) - 1.0, 2.0 * (yp / h) - 1.0)
    }

    pub fn get_params(&self) -> Vec<f32> {
        let p = (&self.mat.clone() / self.mat[(2, 2)]).into_raw_vec();
        match &self.kind {
            TransformationType::Translational => vec![p[2], p[5]],
            TransformationType::Affine => vec![p[0] - 1.0, p[3], p[1], p[4] - 1.0, p[2], p[5]],
            TransformationType::Projective => {
                vec![p[0] - 1.0, p[3], p[1], p[4] - 1.0, p[2], p[5], p[6], p[7]]
            },
            _ => panic!("Transformation cannot be unknown!")
        }
    }

    pub fn inverse(&self) -> Mapping {
        Mapping {
            mat: self.mat.inv().expect("Cannot invert mapping"),
            is_identity: self.is_identity,
            kind: self.kind,
        }
    }

    pub fn transform(&self, lhs: Option<Self>, rhs: Option<Self>) -> Self {
        let (lhs_mat, lhs_id, lhs_kind) = lhs.map_or(
            (Array2::eye(3), false, TransformationType::Unknown),
            |m| (m.mat, m.is_identity, m.kind)
        );

        let (rhs_mat, rhs_id, rhs_kind) = rhs.map_or(
            (Array2::eye(3), false, TransformationType::Unknown),
            |m| (m.mat, m.is_identity, m.kind)
        );

        Mapping {
            mat: lhs_mat.dot(&self.mat).dot(&rhs_mat).to_owned(),
            is_identity: lhs_id & self.is_identity & rhs_id,
            kind: *[lhs_kind, self.kind, rhs_kind].iter().max_by_key(|k| k.num_params()).unwrap()
        }
    }
}

/// Warp an image into a pre-allocated buffer
/// Heavily modeled from
///     https://docs.rs/imageproc/0.23.0/src/imageproc/geometric_transformations.rs.html#496
pub fn warp<P, Fi, Fc>(out: &mut Image<P>, mapping: Fc, get_pixel: Fi)
where
    P: Pixel,
    <P as Pixel>::Subpixel: Send + Sync,
    <P as Pixel>::Subpixel: ValueInto<f32> + Clamp<f32>,
    Fc: Fn(&(f32, f32)) -> (f32, f32) + Send + Sync,
    Fi: Fn(f32, f32) -> P + Send + Sync,
{
    let width = out.width();
    let raw_out = out.as_mut();
    let pitch = P::CHANNEL_COUNT as usize * width as usize;
    let chunks = raw_out.par_chunks_mut(pitch);

    chunks.enumerate().for_each(|(y, row)| {
        for (x, slice) in row.chunks_mut(P::CHANNEL_COUNT as usize).enumerate() {
            let (px, py) = mapping(&(x as f32, y as f32));
            *P::from_slice_mut(slice) = get_pixel(px, py);
        }
    });
}
