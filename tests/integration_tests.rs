use approx::assert_relative_eq;
use burn::backend::wgpu::{AutoGraphicsApi, WgpuRuntime};
use image::{io::Reader as ImageReader, Rgb};
use ndarray::array;
use photoncube2video::transforms::image_to_array3;
use spano::{
    lk::hierarchical_iclk,
    warps::{Mapping, TransformationType},
};
type B = burn::backend::wgpu::JitBackend<WgpuRuntime<AutoGraphicsApi, f32, i32>>;

#[test]
fn test_warp_img() {
    let map = Mapping::<B>::from_matrix(
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
    let img_dst = ImageReader::open("tests/target.png")
        .unwrap()
        .decode()
        .unwrap()
        .into_rgb8();
    let (w, h) = img_src.dimensions();

    let img_warped = map.warp_image(&img_src, (h as usize, w as usize), Some(Rgb([128, 0, 0])));
    // img_warped.save("tests/warp_img_result.png").unwrap();

    let arr_dst = image_to_array3(img_dst).mapv(|v| v as f32);
    let arr_warped = image_to_array3(img_warped).mapv(|v| v as f32);
    assert_relative_eq!(arr_dst, arr_warped);
}

#[test]
fn test_lk() {
    let map = Mapping::<B>::from_matrix(
        array![
            [0.4479, -0.0426, 79.3745],
            [-0.1567, 0.6156, 39.4790],
            [-0.0006, -0.0001, 0.8669]
        ],
        TransformationType::Projective,
    );

    let img_src = ImageReader::open("tests/source.png")
        .unwrap()
        .decode()
        .unwrap()
        .into_rgb8();
    let img_dst = ImageReader::open("tests/warped.png")
        .unwrap()
        .decode()
        .unwrap()
        .into_rgb8();

    let (estimated_map, _) = hierarchical_iclk(
        &img_src,
        &img_dst,
        Mapping::<B>::from_params(vec![0.0; 8]),
        None,
        Some(250),
        (25, 25),
        5,
        Some(1e-3),
        None,
        true,
    )
    .unwrap();

    // Allow 5% error in corner coordinates
    let expected_corners = map.corners((480, 640));
    estimated_map
        .corners((480, 640))
        .into_data()
        .assert_approx_eq_diff(
            &expected_corners.clone().into_data(),
            (expected_corners * 0.05).max().into_scalar().into(),
        );
}
