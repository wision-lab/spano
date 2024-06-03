use burn_tensor::{Shape, Tensor};
use image::{ImageBuffer, Pixel};
use imageproc::definitions::Clamp;

use crate::kernel::Backend;

pub fn tensor3_to_image<P, B>(tensor: Tensor<B, 3>) -> ImageBuffer<P, Vec<P::Subpixel>>
where
    B: Backend,
    P: Pixel,
    <P as Pixel>::Subpixel: Clamp<<B as burn::prelude::Backend>::FloatElem>,
{
    let [height, width, _] = tensor.shape().dims;
    let values = tensor
        .to_data()
        .value
        .into_iter()
        .map(|a| <P as Pixel>::Subpixel::clamp(a))
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
