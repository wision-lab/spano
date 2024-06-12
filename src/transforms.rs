use burn_ndarray::{NdArray, NdArrayTensor};
use burn_tensor::{backend::Backend, Shape, Tensor};
use image::{ImageBuffer, Pixel};
use imageproc::definitions::Clamp;
use ndarray::{ArrayBase, Dim, IxDynImpl, OwnedRepr};
use num_traits::ToPrimitive;
use photoncube2video::transforms::Transform;

/// Normalize tensor between 0-1
pub fn normalize<B: Backend, const D: usize>(tensor: Tensor<B, D>) -> Tensor<B, D> {
    let min = tensor.clone().min();
    let max = tensor.clone().max();
    (tensor - min.clone().into_scalar()).div_scalar((max - min).into_scalar())
}

pub fn tensor3_to_image<P, B>(tensor: Tensor<B, 3>) -> ImageBuffer<P, Vec<P::Subpixel>>
where
    B: Backend,
    P: Pixel,
    <P as Pixel>::Subpixel: Clamp<f32>,
{
    let [height, width, _] = tensor.dims();
    let values = tensor
        .flatten::<1>(0, 2)
        .to_data()
        .value
        .into_iter()
        .map(|a| {
            <P as Pixel>::Subpixel::clamp(a.to_f32().expect("FloatElement should cast to f32"))
        })
        .collect();

    ImageBuffer::<P, Vec<P::Subpixel>>::from_raw(width as u32, height as u32, values)
        .expect("container should have the right size for the image dimensions")
}

pub fn image_to_tensor3<P, B>(
    im: ImageBuffer<P, Vec<P::Subpixel>>,
    device: &B::Device,
) -> Tensor<B, 3>
where
    B: Backend,
    P: Pixel,
    f32: From<<P as Pixel>::Subpixel>,
{
    let shape = Shape::new([
        im.height() as usize,
        im.width() as usize,
        P::CHANNEL_COUNT as usize,
    ]);
    let raw = im.into_raw();
    Tensor::<B, 1>::from_floats(
        &raw.into_iter().map(|i| i.into()).collect::<Vec<_>>()[..],
        device,
    )
    .reshape(shape)
}

pub fn tensor_to_array<B: Backend, const D: usize>(
    tensor: Tensor<B, D>,
) -> ArrayBase<OwnedRepr<f32>, Dim<IxDynImpl>> {
    let arr = Tensor::<NdArray, D>::from_data(tensor.into_data().convert(), &Default::default());
    let primitive: NdArrayTensor<f32, D> = arr.into_primitive();
    primitive.array.to_owned()
}

pub fn array_to_tensor<B: Backend, const D: usize>(
    array: ArrayBase<OwnedRepr<f32>, Dim<IxDynImpl>>,
    device: &B::Device,
) -> Tensor<B, D> {
    let primitive: NdArrayTensor<f32, D> = NdArrayTensor::new(array.into());
    let arr = Tensor::<NdArray, D>::from_primitive(primitive);
    Tensor::<B, D>::from_data(arr.into_data().convert(), device)
}

pub fn apply_tensor_transforms<B: Backend>(
    frame: Tensor<B, 2>,
    transform: &[Transform],
) -> Tensor<B, 2> {
    // Note: if we don't shadow `frame` as a mut, we cannot override it in the loop
    let mut frame = frame;

    // TODO: File a bug report, flip doesn't seem to work after a transpose?
    // See: https://github.com/tracel-ai/burn/issues/1099
    for t in transform.iter() {
        frame = match t {
            Transform::Identity => continue,
            Transform::Rot90 => frame.flip([0]).transpose(),
            Transform::Rot180 => frame.flip([0, 1]),
            Transform::Rot270 => frame.flip([1]).transpose(),
            Transform::FlipUD => frame.flip([0]),
            Transform::FlipLR => frame.flip([1]),
        };
    }
    frame
}

/// Perform batch matrix multiplication by splitting first dimension
/// if larger than 65535 and recombining results
/// See: https://github.com/tracel-ai/burn/issues/1865
pub fn bmm<B: Backend>(a: Tensor<B, 3>, b: Tensor<B, 3>) -> Tensor<B, 3> {
    let batch_size = 65535;
    let [n1, i, j1] = a.dims();
    let [n2, j2, k] = b.dims();
    assert_eq!(n1, n2);
    assert_eq!(j1, j2);

    if n1 <= batch_size {
        return a.matmul(b);
    }

    let ranges: Vec<_> = (0..(n1 as u32).div_ceil(batch_size as u32))
        .map(|i| (batch_size * i as usize)..(batch_size * (i + 1) as usize).min(n1))
        .collect();
    let result_parts: Vec<_> = ranges
        .into_iter()
        .map(|r| {
            let a_part = a.clone().slice([r.clone(), 0..i, 0..j1]);
            let b_part = b.clone().slice([r.clone(), 0..j2, 0..k]);
            a_part.matmul(b_part)
        })
        .collect();
    Tensor::cat(result_parts, 0)
}

// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
#[cfg(test)]
mod test_transforms {
    use approx::assert_relative_eq;
    use burn::backend::wgpu::{AutoGraphicsApi, WgpuRuntime};
    use burn_tensor::{backend::Backend, Tensor};
    use ndarray::{array, Ix2};
    use photoncube2video::transforms::Transform;

    use super::{apply_tensor_transforms, array_to_tensor};
    use crate::transforms::tensor_to_array;
    type B = burn::backend::wgpu::JitBackend<WgpuRuntime<AutoGraphicsApi, f32, i32>>;

    // Enables better debug info to be printed
    fn assert_relative_eq_tensor<B: Backend, const D: usize>(a: Tensor<B, D>, b: Tensor<B, D>) {
        assert_relative_eq!(tensor_to_array(a), tensor_to_array(b))
    }

    #[test]
    fn test_array_tensor_conversions() {
        let a = array![
            [7.941392e-11, 2.2846538e-12],
            [2.2846538e-12, 4.1901697e-11]
        ];
        let t = array_to_tensor::<B, 2>(a.clone().into_dyn(), &Default::default());
        let a_ = tensor_to_array(t).into_dimensionality::<Ix2>().unwrap();
        assert_relative_eq!(a, a_);
    }

    #[test]
    fn test_tensor_transforms_identity() {
        let device = &Default::default();
        let arr: Tensor<B, 2> = Tensor::from_ints([[1, 2], [3, 4], [5, 6]], device).float();
        assert_relative_eq_tensor(
            apply_tensor_transforms(arr.clone(), &[Transform::Identity]),
            arr.clone(),
        );
    }

    #[test]
    fn test_tensor_transforms_rot90() {
        let device = &Default::default();
        let arr: Tensor<B, 2> = Tensor::from_ints([[1, 2], [3, 4], [5, 6]], device).float();
        let arr_90: Tensor<B, 2> = Tensor::from_ints([[5, 3, 1], [6, 4, 2]], device).float();
        assert_relative_eq_tensor(
            apply_tensor_transforms(arr.clone(), &[Transform::Rot90]),
            arr_90.clone(),
        );
    }

    #[test]
    fn test_tensor_transforms_rot180() {
        let device = &Default::default();
        let arr: Tensor<B, 2> = Tensor::from_ints([[1, 2], [3, 4], [5, 6]], device).float();
        let arr_180: Tensor<B, 2> = Tensor::from_ints([[6, 5], [4, 3], [2, 1]], device).float();
        assert_relative_eq_tensor(
            apply_tensor_transforms(arr.clone(), &[Transform::Rot180]),
            arr_180.clone(),
        );
    }

    #[test]
    fn test_tensor_transforms_rot270() {
        let device = &Default::default();
        let arr: Tensor<B, 2> = Tensor::from_ints([[1, 2], [3, 4], [5, 6]], device).float();
        let arr_270: Tensor<B, 2> = Tensor::from_ints([[2, 4, 6], [1, 3, 5]], device).float();
        assert_relative_eq_tensor(
            apply_tensor_transforms(arr.clone(), &[Transform::Rot270]),
            arr_270.clone(),
        );
    }

    #[test]
    fn test_tensor_transforms_flipud() {
        let device = &Default::default();
        let arr: Tensor<B, 2> = Tensor::from_ints([[1, 2], [3, 4], [5, 6]], device).float();
        let arr_ud: Tensor<B, 2> = Tensor::from_ints([[5, 6], [3, 4], [1, 2]], device).float();
        assert_relative_eq_tensor(
            apply_tensor_transforms(arr.clone(), &[Transform::FlipUD]),
            arr_ud.clone(),
        );
    }

    #[test]
    fn test_tensor_transforms_fliplr() {
        let device = &Default::default();
        let arr: Tensor<B, 2> = Tensor::from_ints([[1, 2], [3, 4], [5, 6]], device).float();
        let arr_lr: Tensor<B, 2> = Tensor::from_ints([[2, 1], [4, 3], [6, 5]], device).float();
        assert_relative_eq_tensor(
            apply_tensor_transforms(arr.clone(), &[Transform::FlipLR]),
            arr_lr.clone(),
        );
    }
}
