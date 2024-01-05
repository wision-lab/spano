use criterion::{criterion_group, criterion_main, Criterion};

#[cfg(target_os = "linux")]
use pprof::criterion::{PProfProfiler, Output};

use image::{io::Reader as ImageReader, Rgb};
use ndarray::{array, Array2, Array3};
use nshare::ToNdarray3;

use spano::{
    distance_transform, interpolate_bilinear_with_bkg, warp_array3_into, warp_image, Mapping,
    TransformationType, warp_array3, 
    // array_to_image,
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
            let _out = warp_image(&map, get_pixel, w as usize, h as usize);
            // out.save("out1.png").unwrap();
        })
    });
}

pub fn benchmark_warp_array3(c: &mut Criterion) {
    let img = ImageReader::open("madison1.png")
        .unwrap()
        .decode()
        .unwrap()
        .into_rgb8();
    let (w, h) = img.dimensions();
    let data = img.into_ndarray3().mapv(|v| v as f32);

    let map = Mapping::from_matrix(
        array![
            [0.47654548, -0.045553986, 4.847797],
            [-0.14852144, 0.6426208, 2.1364543],
            [-0.009891294, -0.0021317923, 0.88151735]
        ],
        TransformationType::Projective,
    )
    .rescale(1.0 / 16.0);

    c.bench_function("warp_array3", |b| {
        b.iter(|| {
            let _out = warp_array3(&map, &data, (3, h as usize, w as usize), Some(array![128.0, 0.0, 0.0]));
            // let out = array_to_image(out.mapv(|v| v as u8));
            // out.save("out2.png").unwrap();
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
    let arr = img.into_ndarray3().mapv(|v| v as f32);

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
            warp_array3_into(
                &map,
                &arr,
                &mut out,
                &mut valid,
                None,
                Some(array![0.0, 0.0, 0.0]),
                None,
            );
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

#[cfg(target_os = "linux")]
criterion_group!{
    name = benches;
    config = Criterion::default()
        .with_profiler(
            PProfProfiler::new(100, Output::Flamegraph(None))
        );
    targets =  
        benchmark_warp_image,
        benchmark_warp_array3,
        // benchmark_warp_array3_into,
        // benchmark_distance_transform
}

#[cfg(not(target_os = "linux"))]
criterion_group!{
    benches,
    benchmark_warp_image,
    benchmark_warp_array3,
    // benchmark_warp_array3_into,
    // benchmark_distance_transform
}

criterion_main!(benches);
