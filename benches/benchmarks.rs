use criterion::{criterion_group, criterion_main, Criterion};
use image::{
    imageops::{resize, FilterType::CatmullRom},
    io::Reader as ImageReader,
};
use ndarray::{array, Array3};
#[cfg(target_os = "linux")]
use pprof::criterion::{Output, PProfProfiler};
use spano::{
    blend::distance_transform,
    lk::iclk,
    warps::{Mapping, TransformationType},
};

pub fn benchmark_warp_array3(c: &mut Criterion) {
    let img = ImageReader::open("tests/source.png")
        .unwrap()
        .decode()
        .unwrap()
        .into_rgb8();
    let (w, h) = img.dimensions();
    let data = Array3::from_shape_vec((h as usize, w as usize, 3), img.into_raw())
        .unwrap()
        .mapv(|v| v as f32);

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
            let _out = map.warp_array3(
                &data,
                (h as usize, w as usize),
                Some(array![128.0, 0.0, 0.0]),
            );
        })
    });
}

pub fn benchmark_distance_transform(c: &mut Criterion) {
    c.bench_function("distance_transform", |b| {
        b.iter(|| {
            let _ = distance_transform((480, 640));
        })
    });
}

pub fn benchmark_iclk(c: &mut Criterion) {
    let img_src = ImageReader::open("tests/source.png")
        .unwrap()
        .decode()
        .unwrap()
        .into_rgb8();
    let img_src = resize(&img_src, 640 / 4, 480 / 4, CatmullRom);

    let img_dst = ImageReader::open("tests/warped.png")
        .unwrap()
        .decode()
        .unwrap()
        .into_rgb8();
    let img_dst = resize(&img_dst, 640 / 4, 480 / 4, CatmullRom);

    c.bench_function("distance_iclk", |b| {
        b.iter(|| {
            // No patience, 25 iters, no early-stop.
            let (_map, _) = iclk(
                &img_src,
                &img_dst,
                Mapping::from_params(vec![0.0; 8]),
                Some(25),
                Some(1e-12),
                Some(1),
                None,
            )
            .unwrap();
        })
    });
}

#[cfg(target_os = "linux")]
criterion_group! {
    name = benches;
    config = Criterion::default()
        .with_profiler(
            PProfProfiler::new(100, Output::Flamegraph(None))
        );
    targets =
        // benchmark_warp_array3,
        // benchmark_distance_transform,
        benchmark_iclk
}

#[cfg(not(target_os = "linux"))]
criterion_group! {
    benches,
    benchmark_warp_array3,
    benchmark_distance_transform,
    benchmark_iclk
}

criterion_main!(benches);
