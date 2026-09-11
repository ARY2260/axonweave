use numpy::{PyArray1, PyArrayMethods};
use pyo3::prelude::*;

#[pyfunction]
pub fn decode_argmax<'py>(
    py: Python<'py>,
    vals: Bound<'py, PyArray1<f32>>,
    batch: usize,
    n_in: usize,
    n_actions: usize,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let vals_ro = vals.readonly();
    let v = vals_ro.as_slice()?;

    let take = n_actions.min(n_in);

    let out = py.allow_threads(|| {
        let mut result = vec![0i64; batch];
        for b in 0..batch {
            let base = b * n_in;
            let mut best = 0usize;
            let mut best_val = v[base];
            for a in 1..take {
                if v[base + a] > best_val {
                    best_val = v[base + a];
                    best = a;
                }
            }
            result[b] = best as i64;
        }
        result
    });

    Ok(PyArray1::from_vec_bound(py, out))
}

#[pyfunction]
pub fn decode_clip<'py>(
    py: Python<'py>,
    vals: Bound<'py, PyArray1<f32>>,
    batch: usize,
    n_in: usize,
    n_actions: usize,
    low: f32,
    high: f32,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let vals_ro = vals.readonly();
    let v = vals_ro.as_slice()?;

    let take = n_actions.min(n_in);

    let out = py.allow_threads(|| {
        let mut result = vec![0.0f32; batch * take];
        for b in 0..batch {
            let base = b * n_in;
            for a in 0..take {
                let x = v[base + a];
                result[b * take + a] = if x < low {
                    low
                } else if x > high {
                    high
                } else {
                    x
                };
            }
        }
        result
    });

    Ok(PyArray1::from_vec_bound(py, out))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(decode_argmax, m)?)?;
    m.add_function(wrap_pyfunction!(decode_clip, m)?)?;
    Ok(())
}