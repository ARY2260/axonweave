use numpy::{PyArray1, PyArrayMethods};
use pyo3::prelude::*;

#[pyfunction]
pub fn readout_logits<'py>(
    py: Python<'py>,
    activity: Bound<'py, PyArray1<f32>>,
    weight: Bound<'py, PyArray1<f32>>,
    bias: Bound<'py, PyArray1<f32>>,
    batch: usize,
    n_source: usize,
    n_out: usize,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let a_ro = activity.readonly();
    let w_ro = weight.readonly();
    let b_ro = bias.readonly();
    let av = a_ro.as_slice()?;
    let wv = w_ro.as_slice()?;
    let bv = b_ro.as_slice()?;

    let out = py.allow_threads(|| {
        let mut result = vec![0.0f32; batch * n_out];
        for i in 0..batch {
            for j in 0..n_out {
                let mut acc = bv[j];
                for p in 0..n_source {
                    acc += av[i * n_source + p] * wv[p * n_out + j];
                }
                result[i * n_out + j] = acc;
            }
        }
        result
    });

    Ok(PyArray1::from_vec_bound(py, out))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(readout_logits, m)?)?;
    Ok(())
}