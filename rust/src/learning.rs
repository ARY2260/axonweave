use numpy::{PyArray1, PyArrayMethods};
use pyo3::prelude::*;

#[pyfunction]
#[pyo3(signature = (
    data, indices, indptr, pre_trace, post_trace, pre_activity, post_activity,
    a_plus, a_minus, tau_pre, tau_post, dt, reward=None, w_min=None, w_max=None))]
pub fn stdp_update<'py>(
    py: Python<'py>,
    data: Bound<'py, PyArray1<f32>>,
    indices: Bound<'py, PyArray1<i64>>,
    indptr: Bound<'py, PyArray1<i64>>,
    pre_trace: Bound<'py, PyArray1<f32>>,
    post_trace: Bound<'py, PyArray1<f32>>,
    pre_activity: Bound<'py, PyArray1<f32>>,
    post_activity: Bound<'py, PyArray1<f32>>,
    a_plus: f32,
    a_minus: f32,
    tau_pre: f32,
    tau_post: f32,
    dt: f32,
    reward: Option<f32>,
    w_min: Option<f32>,
    w_max: Option<f32>,
) -> PyResult<(
    Bound<'py, PyArray1<f32>>,
    Bound<'py, PyArray1<f32>>,
    Bound<'py, PyArray1<f32>>,
)> {
    let data_ro = data.readonly();
    let indices_ro = indices.readonly();
    let indptr_ro = indptr.readonly();
    let pre_ro = pre_trace.readonly();
    let post_ro = post_trace.readonly();
    let pre_act_ro = pre_activity.readonly();
    let post_act_ro = post_activity.readonly();

    let d = data_ro.as_slice()?;
    let idx = indices_ro.as_slice()?;
    let ip = indptr_ro.as_slice()?;
    let pre_tr = pre_ro.as_slice()?;
    let post_tr = post_ro.as_slice()?;
    let pre_act = pre_act_ro.as_slice()?;
    let post_act = post_act_ro.as_slice()?;

    let n_rows = ip.len() - 1;
    let multiplier = reward.unwrap_or(1.0);
    let decay_pre = (-dt / tau_pre).exp();
    let decay_post = (-dt / tau_post).exp();

    let result = py.allow_threads(|| {
        let mut pre_new = vec![0.0f32; n_rows];
        for i in 0..n_rows {
            pre_new[i] = pre_tr[i] * decay_pre + pre_act[i];
        }
        let mut post_new = vec![0.0f32; n_rows];
        for i in 0..n_rows {
            post_new[i] = post_tr[i] * decay_post + post_act[i];
        }

        // Map each CSR edge to its owning row (runs in O(nnz)).
        let mut row_of_edge = vec![0usize; d.len()];
        for r in 0..n_rows {
            for k in ip[r] as usize..ip[r + 1] as usize {
                row_of_edge[k] = r;
            }
        }

        let mut data_new = vec![0.0f32; d.len()];
        for (k, &w) in d.iter().enumerate() {
            let col = idx[k] as usize;
            let row = row_of_edge[k];
            let ltp = a_plus * (post_act[row] * pre_tr[col]);
            let ltd = a_minus * (post_new[row] * pre_act[col]);
            let mut val = w + multiplier * (ltp - ltd);
            if let Some(wmn) = w_min {
                if val < wmn {
                    val = wmn;
                }
            }
            if let Some(wmx) = w_max {
                if val > wmx {
                    val = wmx;
                }
            }
            data_new[k] = val;
        }

        (data_new, pre_new, post_new)
    });

    Ok((
        PyArray1::from_vec_bound(py, result.0),
        PyArray1::from_vec_bound(py, result.1),
        PyArray1::from_vec_bound(py, result.2),
    ))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(stdp_update, m)?)?;
    Ok(())
}