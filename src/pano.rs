use anyhow::Result;
use image::Pixel;
use imageproc::definitions::Image;
use itertools::Itertools;
use ndarray::{concatenate, s, Array2, Array3, Axis, NewAxis};
use photoncube2video::transforms::ref_image_to_array3;

use crate::{blend::distance_transform, warps::Mapping};

/// Warning: Silently fails if frames aren't all same size, also slow...
pub fn merge_frames<P>(mappings: &[Mapping], frames: &[Image<P>]) -> Result<Array3<f32>>
where
    P: Pixel + Send + Sync,
    f32: From<<P as Pixel>::Subpixel>,
{
    let sizes: Vec<_> = frames
        .iter()
        .map(|f| (f.width() as usize, f.height() as usize))
        .collect();
    let (extent, offset) = Mapping::maximum_extent(&mappings[..], &sizes[..]);
    let (canvas_w, canvas_h) = extent
        .iter()
        .collect_tuple()
        .expect("Canvas should have width and height");
    let (canvas_h, canvas_w) = (canvas_h.ceil() as usize, canvas_w.ceil() as usize);

    println!(
        "Made Canvas of size {:}x{:}, with offset {:?}",
        &canvas_w,
        &canvas_h,
        &offset.get_params()
    );

    let mut canvas: Array3<f32> = Array3::zeros((canvas_h, canvas_w, 2));
    let mut valid: Array2<bool> = Array2::from_elem((canvas_h, canvas_w), false);

    let (size, _) = Mapping::maximum_extent(&[Mapping::identity()], &sizes[..]);
    let weights = distance_transform(
        size.map(|v| *v as usize)
            .into_iter()
            .collect_tuple()
            .unwrap(),
    );
    let weights = weights.slice(s![.., .., NewAxis]);
    let merge = |dst: &mut [f32], src: &[f32]| {
        dst[0] += src[0] * src[1];
        dst[1] += src[1];
    };

    for (frame, map) in frames.iter().zip(mappings) {
        let frame = ref_image_to_array3(frame).mapv(|v| f32::from(v));
        let frame = concatenate(Axis(2), &[frame.view(), weights.view()])?;
        map.warp_array3_into::<_, f32>(
            &frame.as_standard_layout(),
            &mut canvas,
            &mut valid,
            None,
            None,
            Some(merge),
        );
    }
    Ok(canvas)
}
