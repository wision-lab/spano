use burn_tensor::{
    ops::{BoolTensor, FloatTensor}, Bool, Shape, Tensor
};
use image::Pixel;
use imageproc::definitions::{Clamp, Image};
use itertools::chain;
use ndarray::{array, Array, Array1, Array2, Axis, Ix2};
use ndarray_interp::interp1d::{CubicSpline, Interp1DBuilder, Linear};
use ndarray_linalg::solve::Inverse;
use strum::{EnumCount, VariantArray};
use strum_macros::{Display, EnumString};

use crate::{
    kernels::Backend,
    transforms::{array_to_tensor, image_to_tensor3, tensor3_to_image, tensor_to_array},
};

// Note: We cannot use #[pyclass] her as we're stuck in pyo3@0.15.2 to support py36, so
// we use `EnumString` to convert strings into their enum values.
// TODO: Use pyclass and remove strum dependency when we drop py36 support.
#[derive(Copy, Clone, Debug, EnumString, Display, PartialEq, EnumCount, VariantArray)]
pub enum TransformationType {
    Unknown,
    Identity,
    Translational,
    Affine,
    Projective,
}

impl TransformationType {
    pub fn num_params(&self) -> usize {
        match &self {
            TransformationType::Identity => 0,
            TransformationType::Translational => 2,
            TransformationType::Affine => 6,
            TransformationType::Projective => 8,
            TransformationType::Unknown => 0,
        }
    }
}

// #[pyclass]
#[derive(Debug, Clone)]
pub struct Mapping<B: Backend> {
    pub mat: Tensor<B, 2>,
    pub kind: TransformationType,
}

// These are synchronization primitives used when multithreading.
// We have to define them for our type as they are not automatically derived.
// Without this, we'd have unnecessary trait bounds that specify that
// Mapping::mat is Send+Sync such as:
//     <B as burn::prelude::Backend>::FloatTensorPrimitive<2>: Sync
unsafe impl<B: Backend> Sync for Mapping<B> {}
unsafe impl<B: Backend> Send for Mapping<B> {}

// Note: Methods in this `impl` block are _not_ exposed to python
impl<B: Backend> Mapping<B> {
    pub fn from_tensor(mat: Tensor<B, 2>, kind: TransformationType) -> Self {
        Self { mat, kind }
    }

    pub fn from_matrix(mat: Array2<f32>, kind: TransformationType) -> Self {
        let device = Default::default();
        Self {
            mat: array_to_tensor::<B, 2>(mat.into_dyn(), &device),
            kind,
        }
    }

    pub fn device(&self) -> B::Device {
        self.mat.device()
    }

    pub fn to_device(&self, device: &B::Device) -> Self {
        Self {
            mat: self.mat.clone().to_device(device),
            kind: self.kind,
        }
    }

    /// Warp a set of Nx2 points using the mapping.
    pub fn warp_points(&self, points: Tensor<B, 2>) -> Tensor<B, 2> {
        if self.kind == TransformationType::Identity {
            return points;
        }

        let num_points = points.dims()[0];
        let ones = Tensor::ones([num_points, 1], &self.device());
        let points = Tensor::cat(vec![points, ones], 1);

        let warped_points = self.mat.clone().matmul(points.transpose()).transpose();
        let d = warped_points.clone().slice([0..num_points, 2..3]);
        let warped_points = warped_points.slice([0..num_points, 0..2]);

        warped_points.div(d)
    }

    /// Get location of corners of an image of shape `size` once warped with `self`.
    pub fn corners(&self, size: (usize, usize)) -> Tensor<B, 2> {
        let (w, h) = size;
        let corners = array_to_tensor(
            array![
                [0.0, 0.0],
                [w as f32, 0.0],
                [w as f32, h as f32],
                [0.0, h as f32]
            ]
            .into_dyn(),
            &self.device(),
        );
        self.inverse().warp_points(corners)
    }

    /// Equivalent to getting minimum and maximum x/y coordinates of `corners`.
    pub fn extent(&self, size: (usize, usize)) -> (Tensor<B, 1>, Tensor<B, 1>) {
        let corners = self.corners(size);
        let min_coords = corners.clone().min_dim(0).squeeze(0);
        let max_coords = corners.max_dim(0).squeeze(0);
        (min_coords, max_coords)
    }

    /// Get maximum extent of a collection of warps and theirs sizes.
    /// Sizes are expected to be (x, y) pairs, _not_ (h, w)/(y, x). Similarly extent will be (x, y).
    /// Maps and Sizes might be different lengths:
    ///     - Maybe all warps operate on a single size
    ///     - If warps are the same, this is just max size
    /// Returns an extent (max width, max height) and offset warp.
    pub fn maximum_extent(maps: &[Self], sizes: &[(usize, usize)]) -> (Tensor<B, 1>, Self) {
        // We detect which is longer and cycle the other one.
        let (min_coords, max_coords): (Vec<_>, Vec<_>) = if maps.len() >= sizes.len() {
            maps.iter()
                .zip(sizes.iter().cycle())
                .map(|(m, s)| m.extent(*s))
                .unzip()
        } else {
            sizes
                .iter()
                .zip(maps.iter().cycle())
                .map(|(s, m)| m.extent(*s))
                .unzip()
        };

        let min_coords: Tensor<B, 1> = Tensor::stack::<2>(min_coords, 0).min_dim(0).squeeze(0);
        let max_coords: Tensor<B, 1> = Tensor::stack::<2>(max_coords, 0).min_dim(0).squeeze(0);

        let extent = max_coords - min_coords.clone();
        let offset = Mapping::from_params(min_coords.to_data().convert().value);
        (extent, offset)
    }

    pub fn warp_image<P>(
        &self,
        data: &Image<P>,
        out_size: (usize, usize),
        background: Option<P>,
    ) -> Image<P>
    where
        P: Pixel,
        f32: From<<P as Pixel>::Subpixel>,
        <P as Pixel>::Subpixel: Clamp<f32>,
    {
        let background =
            background.map(|v| v.channels().into_iter().map(|i| f32::from(*i)).collect());
        let img_src = image_to_tensor3::<P, B>(data.clone(), &self.device());
        let (img_warped, _) = self.warp_tensor3(img_src, out_size, background);
        tensor3_to_image::<P, B>(img_warped)
    }

    /// Warp array using mapping into a new buffer of shape `out_size`.
    /// This returns the new buffer along with a mask of which pixels were warped.
    pub fn warp_tensor3(
        &self,
        data: Tensor<B, 3>,
        out_size: (usize, usize),
        background: Option<Vec<f32>>,
    ) -> (Tensor<B, 3>, Tensor<B, 2, Bool>) {
        let (h, w) = out_size;
        let [_, _, c] = data.dims();
        let device = &data.device();
        let mut out = B::float_zeros(Shape::new([h, w, c]), &device);
        let mut valid = B::bool_empty(Shape::new([h, w]), &device);

        self.warp_tensor3_into(data, &mut out, &mut valid, background);
        (Tensor::from_primitive(out), Tensor::from_primitive(valid))
    }

    /// Main interface for warping tensors, use directly if output/valid buffers can be reused.
    /// It uses a custom WSGL compute shader to accelerate warping. Both the warping and bilinear
    /// interpolation are done in the shader. See `kernel.wsgl` and `kernel.rs` for more.
    ///
    /// Args:
    ///     data:
    ///         Data to warp from, this can be any 3 dimensional tensor.
    ///     out:
    ///         Pre-allocated buffer in which to put interpolated data.
    ///     valid:
    ///         Pre-allocated buffer which will hold which pixels have been warped. This tracks which
    ///         pixels are out of bounds. The two first dimensions here need to match that of `out`.
    ///     background: If provided, interpolate between this color and data when sample is near border.
    pub fn warp_tensor3_into(
        &self,
        data: Tensor<B, 3>,
        out: &mut FloatTensor<B, 3>,
        valid: &mut BoolTensor<B, 2>,
        background: Option<Vec<f32>>,
    ) {
        B::warp_into_tensor3(self.clone(), data, out, valid, background);
    }

    /// Given a list of transform parameters, return the Mapping that would transform a
    /// source point to its destination. The type of mapping depends on the number of params (DoF).
    // #[staticmethod]
    // #[pyo3(text_signature = "(cls, params: List[float]) -> Self")]
    pub fn from_params(params: Vec<f32>) -> Self {
        let (full_params, kind) = match &params[..] {
            // Identity
            [] => (Array2::eye(3).into_raw_vec(), TransformationType::Identity),

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

    /// Return a purely scaling (affine) Mapping.
    // #[staticmethod]
    // #[pyo3(text_signature = "(cls, x: float, y: float) -> Self")]
    pub fn scale(x: f32, y: f32) -> Self {
        Self::from_params(vec![x - 1.0, 0.0, 0.0, y - 1.0, 0.0, 0.0])
    }

    /// Return a purely translational Mapping.
    // #[staticmethod]
    // #[pyo3(text_signature = "(cls, x: float, y: float) -> Self")]
    pub fn shift(x: f32, y: f32) -> Self {
        Self::from_params(vec![x, y])
    }

    /// Return an identity Mapping.
    // #[staticmethod]
    // #[pyo3(text_signature = "(cls) -> Self")]
    pub fn identity() -> Self {
        Self::from_params(vec![])
    }

    /// Interpolate a list of Mappings and query a single point.
    /// Ex: Mapping:<B>:::interpolate_scalar(
    ///         [0, 1],
    ///         [Mapping.identity(), Mapping.shift(10, 20)],
    ///         0.5
    ///     ) == Mapping::<B>::shift(5, 10)
    /// See `interpolate_array` for more.
    // #[staticmethod]
    // #[pyo3(text_signature = "(ts: List[float], maps: List[Self], query: float) -> Self:")]
    pub fn interpolate_scalar(ts: Vec<f32>, maps: Vec<Self>, query: f32) -> Self {
        Self::interpolate_array(ts, maps, vec![query])
            .into_iter()
            .nth(0)
            .unwrap()
    }

    /// Interpolate a list of Mappings and query multiple points.
    /// This defaults to performing a cubic spline iterpolation of the warp params, and
    /// falls back to linear interpolation if not enough data points are known (<2).
    // #[staticmethod]
    // #[pyo3(
    //     text_signature = "(ts: List[float], maps: List[Self], query: List[float]) -> List[Self]:"
    // )]
    pub fn interpolate_array(ts: Vec<f32>, maps: Vec<Self>, query: Vec<f32>) -> Vec<Self> {
        let params = Array2::from_shape_vec(
            (maps.len(), 8),
            maps.iter().flat_map(|m| m.get_params_full()).collect(),
        )
        .unwrap();

        let interp_params = if maps.len() > 2 {
            let interpolator = Interp1DBuilder::new(params)
                .x(Array1::from_vec(ts))
                .strategy(CubicSpline::new())
                .build()
                .unwrap();
            interpolator.interp_array(&Array1::from_vec(query)).unwrap()
        } else {
            let interpolator = Interp1DBuilder::new(params)
                .x(Array1::from_vec(ts))
                .strategy(Linear::new())
                .build()
                .unwrap();
            interpolator.interp_array(&Array1::from_vec(query)).unwrap()
        };

        interp_params
            .axis_iter(Axis(0))
            .map(|p| Self::from_params(p.to_vec()))
            .collect()
    }

    /// Compose/accumulate all pairwise mappings together.
    /// This adds in an identity warp to the start to have one warp per frame.
    // #[staticmethod]
    // #[pyo3(text_signature = "(mappings: List[Self]) -> List[Self]")]
    pub fn accumulate(mappings: Vec<Self>) -> Vec<Self> {
        // TODO: maybe impl Copy to minimize the clones here...
        // TODO: Can we avoid the above collect and cumulatively compose in parallel?
        chain([Mapping::identity()], mappings)
            .scan(Mapping::identity(), |acc, x| {
                *acc = acc.transform(None, Some(x.clone()));
                Some(acc.clone())
            })
            .collect()
    }

    /// Apply wrt correction such that the wrt warp becomes the identity.
    // #[staticmethod]
    // #[pyo3(text_signature = "(mappings: List[Self], wrt_map: Self) -> List[Self]")]
    pub fn with_respect_to(mappings: Vec<Self>, wrt_map: Self) -> Vec<Self> {
        mappings
            .iter()
            .map(|m| m.transform(Some(wrt_map.inverse()), None))
            .collect()
    }

    /// Apply wrt correction such that the interpolated warp at the
    /// normalized [0, 1] wrt_idx becomes the identity.
    // #[staticmethod]
    // #[pyo3(text_signature = "(mappings: List[Self], wrt_idx: float) -> List[Self]")]
    pub fn with_respect_to_idx(mappings: Vec<Self>, wrt_idx: f32) -> Vec<Self> {
        let wrt_map = Mapping::interpolate_scalar(
            Array::linspace(0.0, 1.0, mappings.len()).to_vec(),
            mappings.to_owned(),
            wrt_idx,
        );
        Self::with_respect_to(mappings, wrt_map)
    }

    /// Compose/accumulate all pairwise mappings together and apply wrt_idx correction
    /// such that the warp of the frame at the normalized [0, 1] wrt index is the identity.
    /// This effectively accumulates the warps, interpolates them to find the
    /// wrp mapping and then uses `with_respect_to_idx` to undo wrt mapping.
    // #[staticmethod]
    // #[pyo3(text_signature = "(mappings: List[Self], wrt_idx: float) -> List[Self]")]
    pub fn accumulate_wrt_idx(mappings: Vec<Self>, wrt_idx: f32) -> Vec<Self> {
        let mappings = Self::accumulate(mappings);
        let wrt_map = Mapping::interpolate_scalar(
            Array::linspace(0.0, 1.0, mappings.len()).to_vec(),
            mappings.to_owned(),
            wrt_idx,
        );
        Self::with_respect_to(mappings, wrt_map)
    }

    /// Get minimum number of parameters that describe the Mapping.
    // #[pyo3(text_signature = "(self) -> List[float]")]
    pub fn get_params(&self) -> Vec<f32> {
        let mat = tensor_to_array(self.mat.clone())
            .into_dimensionality::<Ix2>()
            .unwrap();
        let p = (&mat.clone() / mat[(2, 2)]).into_raw_vec();
        match &self.kind {
            TransformationType::Identity => vec![],
            TransformationType::Translational => vec![p[2], p[5]],
            TransformationType::Affine => vec![p[0] - 1.0, p[3], p[1], p[4] - 1.0, p[2], p[5]],
            TransformationType::Projective => {
                vec![p[0] - 1.0, p[3], p[1], p[4] - 1.0, p[2], p[5], p[6], p[7]]
            }
            TransformationType::Unknown => panic!("Transformation cannot be unknown!"),
        }
    }

    /// Get all parameters of the Mapping (overparameterized for everything but projective warp).
    // #[pyo3(text_signature = "(self) -> List[float]")]
    pub fn get_params_full(&self) -> Vec<f32> {
        let mat = tensor_to_array(self.mat.clone())
            .into_dimensionality::<Ix2>()
            .unwrap();
        let p = (&mat.clone() / mat[(2, 2)]).into_raw_vec();
        vec![p[0] - 1.0, p[3], p[1], p[4] - 1.0, p[2], p[5], p[6], p[7]]
    }

    /// Invert the mapping by creating new mapping with inverse matrix.
    // #[pyo3(text_signature = "(self) -> Self")]
    pub fn inverse(&self) -> Self {
        // TODO: Is this really necessary?!
        let mat = array_to_tensor(
            tensor_to_array(self.mat.clone())
                .into_dimensionality::<Ix2>()
                .unwrap()
                .inv()
                .expect("Cannot invert mapping")
                .into_dyn(),
            &self.device(),
        );
        Self {
            mat,
            kind: self.kind,
        }
    }

    /// Upgrade Type of warp if it's not unknown, i.e: Identity -> Translational -> Affine -> Projective
    // #[pyo3(text_signature = "(self) -> Self")]
    pub fn upgrade(&self) -> Self {
        // Warning: This relies on the UNKNOWN type being first in the enum!
        if self.kind == TransformationType::Unknown {
            return self.clone();
        }
        let idx = TransformationType::VARIANTS
            .iter()
            .position(|k| *k == self.kind)
            .unwrap();

        Self::from_tensor(
            self.mat.clone(),
            TransformationType::VARIANTS[(idx + 1).min(TransformationType::COUNT - 1)],
        )
    }

    /// Downgrade Type of warp if it's not unknown, i.e: Projective -> Affine -> Translational -> Identity
    // #[pyo3(text_signature = "(self) -> Self")]
    pub fn downgrade(&self) -> Self {
        // Warning: This relies on the UNKNOWN type being first in the enum!
        if self.kind == TransformationType::Unknown {
            return self.clone();
        }
        let idx = TransformationType::VARIANTS
            .iter()
            .position(|k| *k == self.kind)
            .unwrap();

        Self::from_tensor(
            self.mat.clone(),
            TransformationType::VARIANTS[(idx - 1).max(1)],
        )
    }

    /// Compose with other mappings from left or right. Useful for scaling, offsetting, etc...
    /// Resulting mapping will have be cast to the most general mapping kind of all inputs.
    // #[pyo3(text_signature = "(self, *, lhs: Optional[Self], rhs: Optional[Self]) -> Self")]
    pub fn transform(&self, lhs: Option<Self>, rhs: Option<Self>) -> Self {
        let eye = Tensor::<B, 2>::eye(3, &self.device());
        let (lhs_mat, lhs_kind) = lhs.map_or((eye.clone(), TransformationType::Unknown), |m| {
            (m.mat, m.kind)
        });

        let (rhs_mat, rhs_kind) = rhs.map_or((eye.clone(), TransformationType::Unknown), |m| {
            (m.mat, m.kind)
        });

        Mapping {
            mat: lhs_mat.matmul(self.mat.clone()).matmul(rhs_mat),
            kind: *[lhs_kind, self.kind, rhs_kind]
                .iter()
                .max_by_key(|k| k.num_params())
                .unwrap(),
        }
    }

    /// Rescale mapping and keep it's kind intact. This enables a mapping to work
    /// for a rescaled image (since the pixel coordinates get changed too).
    // #[pyo3(text_signature = "(self, scale: float) -> Self")]
    pub fn rescale(&self, scale: f32) -> Self {
        let mut map = self.transform(
            Some(Mapping::scale(1.0 / scale, 1.0 / scale)),
            Some(Mapping::scale(scale, scale)),
        );
        map.kind = self.kind;
        map
    }
}

// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
#[cfg(test)]
mod test_warps {
    use burn::backend::wgpu::{AutoGraphicsApi, WgpuRuntime};
    use burn_tensor::Tensor;
    use ndarray::{array, Array2};

    use crate::warps::{Mapping, TransformationType};

    #[test]
    fn test_warp_points() {
        type MyBackend = burn::backend::wgpu::JitBackend<WgpuRuntime<AutoGraphicsApi, f32, i32>>;
        let device = Default::default();

        let map = Mapping::<MyBackend>::from_matrix(
            array![
                [1.13411823, 4.38092511, 9.315785],
                [1.37351153, 5.27648111, 1.60252762],
                [7.76114426, 9.66312177, 2.61286966]
            ],
            TransformationType::Projective,
        );
        let point = Tensor::from_floats([[0.0, 0.0]], &device);
        let warpd = map.warp_points(point);
        Tensor::<MyBackend, 2>::from_floats([[3.56534624, 0.61332092]], &device)
            .into_data()
            .assert_approx_eq(&warpd.into_data(), 3);
    }

    #[test]
    fn test_updowngrade() {
        type MyBackend = burn::backend::wgpu::JitBackend<WgpuRuntime<AutoGraphicsApi, f32, i32>>;

        let map = Mapping::<MyBackend>::from_matrix(Array2::eye(3), TransformationType::Unknown);
        assert!(map.upgrade().kind == TransformationType::Unknown);
        assert!(map.downgrade().kind == TransformationType::Unknown);

        let mut map = Mapping::<MyBackend>::identity();
        assert!(map.kind == TransformationType::Identity);

        map = map.upgrade();
        assert!(map.kind == TransformationType::Translational);
        map = map.upgrade();
        assert!(map.kind == TransformationType::Affine);
        map = map.upgrade();
        assert!(map.kind == TransformationType::Projective);
        map = map.upgrade();
        assert!(map.kind == TransformationType::Projective);

        map = map.downgrade();
        assert!(map.kind == TransformationType::Affine);
        map = map.downgrade();
        assert!(map.kind == TransformationType::Translational);
        map = map.downgrade();
        assert!(map.kind == TransformationType::Identity);
        map = map.downgrade();
        assert!(map.kind == TransformationType::Identity);
    }
}
