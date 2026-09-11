use numpy::{PyArray1, PyArrayMethods};
use pyo3::prelude::*;

#[pyfunction]
pub fn dense_matmul<'py>(
    py: Python<'py>,
    a: Bound<'py, PyArray1<f32>>,
    b: Bound<'py, PyArray1<f32>>,
    m: usize,
    k: usize,
    n: usize,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let a_ro = a.readonly();
    let b_ro = b.readonly();
    let av = a_ro.as_slice()?;
    let bv = b_ro.as_slice()?;

    let out = py.allow_threads(|| {
        let mut result = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0f32;
                for p in 0..k {
                    acc += av[i * k + p] * bv[p * n + j];
                }
                result[i * n + j] = acc;
            }
        }
        result
    });

    Ok(PyArray1::from_vec_bound(py, out))
}

#[pyfunction]
pub fn row_absmax_normalize<'py>(
    py: Python<'py>,
    x: Bound<'py, PyArray1<f32>>,
    m: usize,
    k: usize,
    eps: f32,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let x_ro = x.readonly();
    let xv = x_ro.as_slice()?;

    let out = py.allow_threads(|| {
        let mut result = vec![0.0f32; m * k];
        for i in 0..m {
            let mut row_max = 0.0f32;
            for p in 0..k {
                let v = xv[i * k + p].abs();
                if v > row_max {
                    row_max = v;
                }
            }
            let denom = row_max.max(eps);
            for p in 0..k {
                result[i * k + p] = xv[i * k + p] / denom;
            }
        }
        result
    });

    Ok(PyArray1::from_vec_bound(py, out))
}

#[pyfunction]
pub fn embed_lookup<'py>(
    py: Python<'py>,
    ids: Bound<'py, PyArray1<i64>>,
    embedding: Bound<'py, PyArray1<f32>>,
    vocab: usize,
    dim: usize,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let ids_ro = ids.readonly();
    let emb_ro = embedding.readonly();
    let idv = ids_ro.as_slice()?;
    let emb = emb_ro.as_slice()?;
    let n_ids = idv.len();

    let out = py.allow_threads(|| {
        let mut result = vec![0.0f32; n_ids * dim];
        for (r, &tok) in idv.iter().enumerate() {
            let start = (tok as usize) * dim;
            for p in 0..dim {
                result[r * dim + p] = emb[start + p];
            }
        }
        result
    });

    Ok(PyArray1::from_vec_bound(py, out))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(dense_matmul, m)?)?;
    m.add_function(wrap_pyfunction!(row_absmax_normalize, m)?)?;
    m.add_function(wrap_pyfunction!(embed_lookup, m)?)?;
    Ok(())
}