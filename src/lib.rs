#![warn(unused_extern_crates)]

pub mod blend;
pub mod cli;
pub mod kernel;
pub mod lk;
pub mod scripts;
pub mod transforms;
pub mod transpose;
pub mod utils;
pub mod warps;

use pyo3::prelude::*;

use crate::{scripts::__pyo3_get_function_cli_entrypoint, warps::Mapping};

#[pymodule]
fn spano(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(cli_entrypoint))?;
    m.add_class::<Mapping>()?;
    Ok(())
}
