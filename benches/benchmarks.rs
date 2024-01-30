use criterion::{criterion_group, criterion_main, Criterion};
use image::io::Reader as ImageReader;
use ndarray::{array, Array3};
#[cfg(target_os = "linux")]
use pprof::criterion::{Output, PProfProfiler};
use spano::{
    blend::distance_transform,
    warps::{warp_array3, Mapping, TransformationType},
};

pub fn benchmark_warp_array3(c: &mut Criterion) {
    let img = ImageReader::open("assets/madison1.png")
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
            let _out = warp_array3(
                &map,
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

#[cfg(target_os = "linux")]
criterion_group! {
    name = benches;
    config = Criterion::default()
        .with_profiler(
            PProfProfiler::new(100, Output::Flamegraph(None))
        );
    targets =
        benchmark_warp_array3,
        benchmark_distance_transform
}

#[cfg(not(target_os = "linux"))]
criterion_group! {
    benches,
    benchmark_warp_array3,
    benchmark_distance_transform
}

criterion_main!(benches);
