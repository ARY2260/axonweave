mod decoders;
mod delays;
mod dynamics;
mod encoders;
mod graph;
mod learning;
mod provisioning;
mod readout;
mod receptors;
mod signals;
mod surrogate;

use pyo3::prelude::*;

#[pyfunction]
fn version() -> &'static str {
    "0.1.0"
}

#[pyfunction]
fn weighted_leak(state: f32, input: f32, decay: f32) -> f32 {
    decay * state + input
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", "0.1.0")?;
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_function(wrap_pyfunction!(weighted_leak, m)?)?;
    graph::register(m)?;
    dynamics::register(m)?;
    learning::register(m)?;
    surrogate::register(m)?;
    signals::register(m)?;
    receptors::register(m)?;
    delays::register(m)?;
    encoders::register(m)?;
    decoders::register(m)?;
    readout::register(m)?;
    provisioning::register(m)?;
    Ok(())
}