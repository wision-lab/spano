use criterion::{black_box, criterion_group, criterion_main, Criterion};
use image::{io::Reader as ImageReader, Rgb};

use ndarray::array;
use spano::{interpolate_bilinear_with_bkg, warp_image, Mapping, TransformationType};

pub fn benchmark_warp(c: &mut Criterion) {
    let img = ImageReader::open("madison1.png")
        .unwrap()
        .decode()
        .unwrap()
        .into_rgb8();
    let (w, h) = img.dimensions();

    let get_pixel = |x, y| interpolate_bilinear_with_bkg(&img, x, y, Rgb([128, 0, 0]));
    let map = Mapping::from_matrix(
        array![
            [1.13411823, 4.38092511, 9.315785],
            [1.37351153, 5.27648111, 1.60252762],
            [7.76114426, 9.66312177, 2.61286966]
        ],
        TransformationType::Projective,
    );

    c.bench_function("warp", |b| {
        b.iter(|| {
            warp_image(&map, get_pixel, w, h);
        })
    });
}

criterion_group!(benches, benchmark_warp);
criterion_main!(benches);
