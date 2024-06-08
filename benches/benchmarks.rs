use burn::{
    backend::wgpu::{AutoGraphicsApi, WgpuRuntime},
    tensor::ops::{BoolTensorOps, FloatTensorOps},
};
use burn_jit::kernel::into_contiguous;
use burn_tensor::{Shape, Tensor};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use image::{
    imageops::{resize, FilterType::CatmullRom},
    io::Reader as ImageReader,
    Rgb,
};
use ndarray::array;
use photoncube2video::transforms::image_to_array3;
#[cfg(target_os = "linux")]
use pprof::criterion::{Output, PProfProfiler};
use spano::{
    blend::{blend_images, distance_transform},
    kernels::Backend,
    lk::iclk,
    warps::{Mapping, TransformationType},
};

fn benchmark_warp_tensor3(c: &mut Criterion) {
    type B = burn::backend::wgpu::JitBackend<WgpuRuntime<AutoGraphicsApi, f32, i32>>;
    let device = &Default::default();

    let mapping = Mapping::<B>::from_matrix(
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
    let shape = Shape::new([h as usize, w as usize, 3]);

    let input = image_to_array3::<Rgb<u8>>(img_src)
        .mapv(|v| (v as f32) / 255.0)
        .into_raw_vec();
    let input = Tensor::<B, 1>::from_floats(&input[..], device).reshape(shape.clone());
    let output = B::float_zeros(shape.clone(), device);
    let output = into_contiguous(output);
    let valid = B::bool_empty(Shape::new([h as usize, w as usize]), device);
    let valid = into_contiguous(valid);

    c.bench_function("warp_tensor3", |b| {
        b.iter(|| {
            B::warp_into_tensor3(
                black_box(mapping.clone()),
                black_box(input.clone()),
                &mut black_box(output.clone()),
                &mut black_box(valid.clone()),
                Some(vec![0.0, 0.0, 0.0]),
            );
        })
    });
}

pub fn benchmark_distance_transform(c: &mut Criterion) {
    type B = burn::backend::wgpu::JitBackend<WgpuRuntime<AutoGraphicsApi, f32, i32>>;
    let device = &Default::default();

    c.bench_function("distance_transform", |b| {
        b.iter(|| {
            let _ = distance_transform::<B, 2>(Shape::new([480, 640]), device);
        })
    });
}

pub fn benchmark_mapping_from_params(c: &mut Criterion) {
    type B = burn::backend::wgpu::JitBackend<WgpuRuntime<AutoGraphicsApi, f32, i32>>;
    let params = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

    c.bench_function("mapping_from_params", |b| {
        b.iter(|| {
            let _ = Mapping::<B>::from_params(black_box(params.clone()));
        })
    });
}

pub fn benchmark_mapping_get_params(c: &mut Criterion) {
    type B = burn::backend::wgpu::JitBackend<WgpuRuntime<AutoGraphicsApi, f32, i32>>;
    let params = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let map = Mapping::<B>::from_params(params);

    c.bench_with_input(BenchmarkId::new("mapping_get_params", map.clone()), &map.clone(), |b, m| {
        b.iter(|| {
            let _ = m.get_params();
        })
    });
}

pub fn benchmark_mapping_inverse(c: &mut Criterion) {
    type B = burn::backend::wgpu::JitBackend<WgpuRuntime<AutoGraphicsApi, f32, i32>>;
    let params = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let map = Mapping::<B>::from_params(params);

    c.bench_with_input(BenchmarkId::new("mapping_inverse", map.clone()), &map.clone(), |b, m| {
        b.iter(|| {
            let _ = m.inverse();
        })
    });
}

pub fn benchmark_iclk(c: &mut Criterion) {
    type B = burn::backend::wgpu::JitBackend<WgpuRuntime<AutoGraphicsApi, f32, i32>>;

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

    c.bench_function("iclk", |b| {
        b.iter(|| {
            // No patience, 25 iters, no early-stop.
            let (_map, _) = iclk(
                &img_src,
                &img_dst,
                Mapping::<B>::from_params(vec![0.0; 8]),
                None,
                Some(25),
                Some(1e-12),
                Some(1),
                None,
            )
            .unwrap();
        })
    });
}

pub fn benchmark_merge_images(c: &mut Criterion) {
    type B = burn::backend::wgpu::JitBackend<WgpuRuntime<AutoGraphicsApi, f32, i32>>;

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
    let imgs = [img_src, img_dst];

    let map = Mapping::<B>::from_matrix(
        array![
            [0.47654548, -0.045553986, 4.847797],
            [-0.14852144, 0.6426208, 2.1364543],
            [-0.009891294, -0.0021317923, 0.88151735]
        ],
        TransformationType::Projective,
    )
    .rescale(1.0 / 16.0);
    let maps = [Mapping::identity(), map];

    c.bench_function("merge_images", |b| {
        b.iter(|| {
            let _out = blend_images(&maps, &imgs, None).unwrap();
            // _out.save("tests/benchmark_merge_images.png").unwrap();
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
        benchmark_warp_tensor3,
        benchmark_distance_transform,
        benchmark_mapping_from_params,
        benchmark_mapping_get_params,
        benchmark_mapping_inverse,
        benchmark_iclk,
        benchmark_merge_images
}

#[cfg(not(target_os = "linux"))]
criterion_group! {
    benches,
    benchmark_warp_tensor3,
    benchmark_mapping_from_params,
    benchmark_mapping_get_params,
    benchmark_distance_transform,
    benchmark_mapping_inverse,
    benchmark_iclk,
    benchmark_merge_images
}

criterion_main!(benches);
