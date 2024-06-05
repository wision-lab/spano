use burn::{
    backend::wgpu::{AutoGraphicsApi, WgpuRuntime},
    tensor::ops::{BoolTensorOps, FloatTensorOps},
};
use burn_jit::kernel::into_contiguous;
use burn_tensor::{Shape, Tensor};
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use image::{
    // imageops::{resize, FilterType::CatmullRom},
    io::Reader as ImageReader,
    Rgb,
};
// use ndarray::{array, Array3};
use photoncube2video::transforms::image_to_array3;
#[cfg(target_os = "linux")]
use pprof::criterion::{Output, PProfProfiler};
use spano::{
    // blend::{distance_transform, merge_images},
    kernel::Backend,
    // lk::iclk,
    // warps::{Mapping, TransformationType},
};

fn benchmark_warp_tensor3(c: &mut Criterion) {
    type B = burn::backend::wgpu::JitBackend<WgpuRuntime<AutoGraphicsApi, f32, i32>>;
    let device = &Default::default();

    let mapping = Tensor::<B, 1>::from_floats(
        [
            1.9068071,
            0.09958228,
            -171.64162,
            0.3666181,
            1.5628628,
            -92.86306,
            0.0013926513,
            0.00030605582,
            1.0,
        ],
        device,
    )
    .reshape(Shape::new([3, 3]))
    .into_primitive();

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
    let input = Tensor::<B, 1>::from_floats(&input[..], device)
        .reshape(shape.clone())
        .into_primitive();
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
                vec![0.0, 0.0, 0.0],
            );
        })
    });
}

// pub fn benchmark_distance_transform(c: &mut Criterion) {
//     c.bench_function("distance_transform", |b| {
//         b.iter(|| {
//             let _ = distance_transform((480, 640));
//         })
//     });
// }

// pub fn benchmark_iclk(c: &mut Criterion) {
//     let img_src = ImageReader::open("tests/source.png")
//         .unwrap()
//         .decode()
//         .unwrap()
//         .into_rgb8();
//     let img_src = resize(&img_src, 640 / 4, 480 / 4, CatmullRom);

//     let img_dst = ImageReader::open("tests/warped.png")
//         .unwrap()
//         .decode()
//         .unwrap()
//         .into_rgb8();
//     let img_dst = resize(&img_dst, 640 / 4, 480 / 4, CatmullRom);

//     c.bench_function("iclk", |b| {
//         b.iter(|| {
//             // No patience, 25 iters, no early-stop.
//             let (_map, _) = iclk(
//                 &img_src,
//                 &img_dst,
//                 Mapping::from_params(vec![0.0; 8]),
//                 None,
//                 Some(25),
//                 Some(1e-12),
//                 Some(1),
//                 None,
//             )
//             .unwrap();
//         })
//     });
// }

// pub fn benchmark_merge_images(c: &mut Criterion) {
//     let img_src = ImageReader::open("tests/source.png")
//         .unwrap()
//         .decode()
//         .unwrap()
//         .into_rgb8();
//     let img_dst = ImageReader::open("tests/warped.png")
//         .unwrap()
//         .decode()
//         .unwrap()
//         .into_rgb8();
//     let imgs = [img_src, img_dst];

//     let map = Mapping::from_matrix(
//         array![
//             [0.47654548, -0.045553986, 4.847797],
//             [-0.14852144, 0.6426208, 2.1364543],
//             [-0.009891294, -0.0021317923, 0.88151735]
//         ],
//         TransformationType::Projective,
//     )
//     .rescale(1.0 / 16.0);
//     let maps = [Mapping::identity(), map];

//     c.bench_function("merge_images", |b| {
//         b.iter(|| {
//             let _ = merge_images(&maps, &imgs, None).unwrap();
//         })
//     });
// }

#[cfg(target_os = "linux")]
criterion_group! {
    name = benches;
    config = Criterion::default()
        .with_profiler(
            PProfProfiler::new(100, Output::Flamegraph(None))
        );
    targets =
        // benchmark_warp_array3,
        benchmark_warp_tensor3,
        // benchmark_distance_transform,
        // benchmark_iclk,
        // benchmark_merge_images
}

#[cfg(not(target_os = "linux"))]
criterion_group! {
    benches,
    // benchmark_warp_array3,
    benchmark_warp_tensor3,
    // benchmark_distance_transform,
    // benchmark_iclk,
    // benchmark_merge_images
}

criterion_main!(benches);
