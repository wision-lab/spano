#![warn(unused_extern_crates)]

pub mod blend;
pub mod cli;
pub mod lk;
pub mod pano;
pub mod scripts;
pub mod transpose;
pub mod utils;
pub mod warps;

use pyo3::prelude::*;

use crate::{
    blend::merge_arrays_py,
    lk::{iclk_py, img_pyramid_py, pairwise_iclk_py},
    scripts::cli_entrypoint,
    utils::animate_warp_py,
    warps::{Mapping, TransformationType},
};

#[pymodule]
fn spano(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(cli_entrypoint))?;

    m.add_wrapped(wrap_pyfunction!(iclk_py))?;
    m.add_wrapped(wrap_pyfunction!(pairwise_iclk_py))?;
    m.add_wrapped(wrap_pyfunction!(img_pyramid_py))?;

    m.add_class::<Mapping>()?;
    m.add_class::<TransformationType>()?;

    m.add_wrapped(wrap_pyfunction!(animate_warp_py))?;
    m.add_wrapped(wrap_pyfunction!(merge_arrays_py))?;
    Ok(())
}
