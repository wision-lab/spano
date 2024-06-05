use burn_tensor::{Shape, Tensor};
use image::{ImageBuffer, Pixel};
use imageproc::definitions::Clamp;
use ndarray::Array2;
use num_traits::ToPrimitive;

use crate::kernels::Backend;

pub fn tensor3_to_image<P, B>(tensor: Tensor<B, 3>) -> ImageBuffer<P, Vec<P::Subpixel>>
where
    B: Backend,
    P: Pixel,
    <P as Pixel>::Subpixel: Clamp<f32>,
{
    let [height, width, _] = tensor.shape().dims;
    let values = tensor
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
        &device,
    )
    .reshape(shape)
}

pub fn array2_to_tensor2<T, B: Backend>(arr: Array2<T>, device: &B::Device) -> Tensor<B, 2>
where
    f32: From<T>,
{
    // TODO: This is likely suboptimal
    assert!(arr.is_standard_layout());
    let shape = Shape::new(arr.dim().into());
    Tensor::<B, 1>::from_floats(
        &arr.into_raw_vec()
            .into_iter()
            .map(|i| i.into())
            .collect::<Vec<_>>()[..],
        &device,
    )
    .reshape(shape)
}

pub fn tensor2_to_array2<B: Backend>(tensor: Tensor<B, 2>) -> Array2<f32> {
    let vals: Vec<f32> = tensor.to_data().convert().value;

    Array2::from_shape_vec(tensor.shape().dims, vals)
        .expect("Tensor should be converted to Array of same shape.")
}
