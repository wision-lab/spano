use burn::backend::wgpu::{AutoGraphicsApi, WgpuRuntime};
use image::{imageops::grayscale, io::Reader as ImageReader, Luma};
use spano::{
    lk::tensor_gradients,
    transforms::{image_to_tensor3, normalize, tensor3_to_image},
};

fn main() {
    type MyBackend = burn::backend::wgpu::JitBackend<WgpuRuntime<AutoGraphicsApi, f32, i32>>;
    let device = Default::default();

    let img_src = ImageReader::open("tests/source.png")
        .unwrap()
        .decode()
        .unwrap()
        .into_rgb8();

    let img = image_to_tensor3::<Luma<u8>, MyBackend>(grayscale(&img_src), &device);
    let (grad_x, grad_y) = tensor_gradients(img);
    let img_x = tensor3_to_image::<Luma<u8>, MyBackend>(normalize(grad_x) * 255.0);
    let img_y = tensor3_to_image::<Luma<u8>, MyBackend>(normalize(grad_y) * 255.0);

    img_x.save("examples/grads_x.png".to_string()).unwrap();
    img_y.save("examples/grads_y.png".to_string()).unwrap();
}
