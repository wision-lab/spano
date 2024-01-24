use approx::assert_relative_eq;
use image::{io::Reader as ImageReader, Rgb};
use ndarray::array;

use spano::{
    transforms::image_to_array3,
    warps::{warp_image, Mapping, TransformationType},
};

#[test]
fn test_warp_img() {
    let map = Mapping::from_matrix(
        array![
            [0.47654548, -0.045553986, 4.847797],
            [-0.14852144, 0.6426208, 2.1364543],
            [-0.009891294, -0.0021317923, 0.88151735]
        ],
        TransformationType::Projective,
    )
    .rescale(1.0 / 16.0);

    let img_src = ImageReader::open("tests/source.png")
        .unwrap()
        .decode()
        .unwrap()
        .into_rgb8();
    let img_dst = ImageReader::open("tests/target.png")
        .unwrap()
        .decode()
        .unwrap()
        .into_rgb8();
    let (w, h) = img_src.dimensions();

    let img_warped = warp_image(
        &map,
        &img_src,
        (h as usize, w as usize),
        Some(Rgb([128, 0, 0])),
    );
    let arr_dst = image_to_array3(img_dst).mapv(|v| v as f32);
    let arr_warped = image_to_array3(img_warped).mapv(|v| v as f32);
    assert_relative_eq!(arr_dst, arr_warped);
}
