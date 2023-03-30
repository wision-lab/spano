#![allow(dead_code)] // Todo: Remove

use anyhow::Result;
use blend::interpolate_bilinear;
use image::imageops::{resize, FilterType};
use image::{io::Reader as ImageReader, ImageBuffer, Rgb};

use warps::warp;

mod blend;
mod lk;
mod warps;

use crate::lk::iclk;
use crate::warps::{TransformationType, Mapping};

#[allow(dead_code)]
fn print_type_of<T>(_: &T) {
    println!("{}", std::any::type_name::<T>())
}

fn main() -> Result<()> {
    // let dst = polygon_sdf_vec(
    //     array![[0.0, 0.0], [500.0, 100.0], [800.0, 900.0], [10.0, 990.0]]
    // );
    // let values: Vec<f32> = (0..1000).cartesian_product(0..1000).collect::<Vec<_>>().par_iter().map(
    //     |(x, y)| dst(*x as f32, *y as f32)
    // ).collect();

    // let max = values.par_iter().max_by(|x, y| x.partial_cmp(&y).unwrap()).unwrap();
    // let min = values.par_iter().min_by(|x, y| x.partial_cmp(&y).unwrap()).unwrap();
    // println!("{max}, {min}");

    // let out: ImageBuffer<Luma<u8>, Vec<u8>> = ImageBuffer::from_vec(
    //     1000, 1000,
    //     values.iter().map(|p|
    //         ((p-min) / (max-min) * 255.0) as u8
    //     ).collect()
    // ).unwrap();
    // out.save("pano.png")?;

    // -----------------------------------------------

    // let img1 = ImageReader::open("madison.jpg")?.decode()?.into_rgb8();
    // let mut out = ImageBuffer::new(1000, 1000);
    // let get_pixel = |x, y| interpolate_bilinear(&img1, x, y, Rgb([0, 0, 0]));
    // let mapping = Mapping::from_params(vec![-100.0, -100.0]);
    // warp(&mut out, mapping, get_pixel);
    // out.save("pano.png")?;

    // ----------------------------------------------
    let (w, h) = (128, 128);

    let img1 = ImageReader::open("A.png")?.decode()?.into_rgb8();
    let img1 = resize(&img1, w, h, FilterType::CatmullRom);

    let img2 = ImageReader::open("B.png")?.decode()?.into_rgb8();
    let img2 = resize(&img2, w, h, FilterType::CatmullRom);

    let mapping = iclk(&img1, &img2, TransformationType::Translational, Some(5000))?;
    // let mapping = Mapping::from_params(&vec![0.0, 0.0]);
    println!("{:?}", &mapping);

    {
        let mut out = ImageBuffer::new(w, h);  //741x500
        let get_pixel = |x, y| interpolate_bilinear(&img1, x, y).unwrap_or(Rgb([128, 0, 0]));
        warp(&mut out, mapping.warpfn(), get_pixel);
        out.save("out1.png")?;
    }
    {
        let mut out = ImageBuffer::new(w, h);  //741x500
        let get_pixel = |x, y| interpolate_bilinear(&img1, x, y).unwrap_or(Rgb([128, 0, 0]));
        warp(&mut out, mapping.inverse().warpfn(), get_pixel);
        out.save("out2.png")?;
    }
    {
        let mut out = ImageBuffer::new(w, h);  //741x500
        let get_pixel = |x, y| interpolate_bilinear(&img2, x, y).unwrap_or(Rgb([128, 0, 0]));
        warp(&mut out, mapping.warpfn(), get_pixel);
        out.save("out3.png")?;
    }
    {
        let mut out = ImageBuffer::new(w, h);  //741x500
        let get_pixel = |x, y| interpolate_bilinear(&img2, x, y).unwrap_or(Rgb([128, 0, 0]));
        warp(&mut out, mapping.inverse().warpfn(), get_pixel);
        out.save("out4.png")?;
    }

    img1.save("img1.png")?;
    img2.save("img2.png")?;
    Ok(())
}
