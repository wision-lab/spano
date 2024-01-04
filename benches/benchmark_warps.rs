use criterion::{criterion_group, criterion_main, Criterion};
use image::{io::Reader as ImageReader, Rgb};

use ndarray::{array, Array3, Array2};
use nshare::ToNdarray3;
use spano::{
    distance_transform, interpolate_bilinear_with_bkg, warp_array3_into, warp_image, Mapping,
    TransformationType,
};

pub fn benchmark_warp_image(c: &mut Criterion) {
    let img = ImageReader::open("madison1.png")
        .unwrap()
        .decode()
        .unwrap()
        .into_rgb8();
    let (w, h) = img.dimensions();

    let get_pixel = |x, y| interpolate_bilinear_with_bkg(&img, x, y, Rgb([128, 0, 0]));
    let map = Mapping::from_matrix(
        array![
            [0.47654548, -0.045553986, 4.847797],
            [-0.14852144, 0.6426208, 2.1364543],
            [-0.009891294, -0.0021317923, 0.88151735]
        ],
        TransformationType::Projective,
    )
    .rescale(1.0 / 16.0);

    c.bench_function("warp_image", |b| {
        b.iter(|| {
            warp_image(&map, get_pixel, w as usize, h as usize);
        })
    });
}

pub fn benchmark_warp_array3_into(c: &mut Criterion) {
    let img = ImageReader::open("madison1.png")
        .unwrap()
        .decode()
        .unwrap()
        .into_rgb8();
    let (w, h) = img.dimensions();
    let arr = img
        .into_ndarray3()
        .mapv(|v| v as f32)
        .permuted_axes([1, 2, 0]);

    let map = Mapping::from_matrix(
        array![
            [0.47654548, -0.045553986, 4.847797],
            [-0.14852144, 0.6426208, 2.1364543],
            [-0.009891294, -0.0021317923, 0.88151735]
        ],
        TransformationType::Projective,
    )
    .rescale(1.0 / 16.0);

    let mut out = Array3::zeros((3, h as usize, w as usize));
    let mut valid = Array2::from_elem((h as usize, w as usize), false);

    c.bench_function("warp_array3_into", |b| {
        b.iter(|| {
            warp_array3_into(&map, &arr, &mut out, &mut valid, None, Some(array![0.0, 0.0, 0.0]));
        })
    });
}

pub fn benchmark_distance_transform(c: &mut Criterion) {
    c.bench_function("distance_transform", |b| {
        b.iter(|| {
            let _ = distance_transform((300, 300));
        })
    });
}

criterion_group!(
    benches,
    // benchmark_warp_image,
    benchmark_warp_array3_into,
    // benchmark_distance_transform
);
criterion_main!(benches);
