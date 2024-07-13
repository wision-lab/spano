#![warn(unused_extern_crates)]

pub mod blend;
pub mod cli;
pub mod lk;
pub mod scripts;
pub mod transpose;
pub mod utils;
pub mod warps;

use pyo3::prelude::*;

use crate::{scripts::cli_entrypoint, warps::Mapping};

#[pymodule]
fn spano(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(cli_entrypoint))?;
    m.add_class::<Mapping>()?;
    Ok(())
}
