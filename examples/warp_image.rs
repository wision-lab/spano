use burn::backend::wgpu::{AutoGraphicsApi, WgpuRuntime};
use image::{io::Reader as ImageReader, Rgb};
use ndarray::array;
use spano::warps::{Mapping, TransformationType};

fn main() {
    type MyBackend = burn::backend::wgpu::JitBackend<WgpuRuntime<AutoGraphicsApi, f32, i32>>;

    let map = Mapping::<MyBackend>::from_matrix(
        array![
            [1.9068071, 0.09958228, -171.64162],
            [0.3666181, 1.5628628, -92.86306],
            [0.0013926513, 0.00030605582, 1.0]
        ],
        TransformationType::Projective,
    );

    let img_src = ImageReader::open("tests/source.png")
        .unwrap()
        .decode()
        .unwrap()
        .into_rgb8();
    let (w, h) = img_src.dimensions();

    let img_warped = map.warp_image::<Rgb<u8>>(&img_src, (h as usize, w as usize), None);
    img_warped.save("out.png".to_string()).unwrap();
}
