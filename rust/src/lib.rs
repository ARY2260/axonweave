use pyo3::prelude::*;

#[pyfunction]
fn version() -> &'static str { "0.1.0" }

#[pyfunction]
fn weighted_leak(state: f32, input: f32, decay: f32) -> f32 { decay * state + input }

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_function(wrap_pyfunction!(weighted_leak, m)?)?;
    Ok(())
}
