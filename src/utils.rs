#[allow(dead_code)]
pub fn array2grayimage(frame: Array2<u8>) -> Option<GrayImage> {
    GrayImage::from_raw(
        frame.len_of(Axis(1)) as u32,
        frame.len_of(Axis(0)) as u32,
        frame.into_raw_vec(),
    )
}
