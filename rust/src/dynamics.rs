use numpy::{PyArray1, PyArrayMethods};
use pyo3::prelude::*;

#[pyfunction]
pub fn lif_step<'py>(
    py: Python<'py>,
    v: Bound<'py, PyArray1<f32>>,
    refrac_until: Bound<'py, PyArray1<f32>>,
    current: Bound<'py, PyArray1<f32>>,
    t: f32,
    tau: f32,
    v_rest: f32,
    v_threshold: f32,
    v_reset: f32,
    refractory: f32,
    dt: f32,
) -> PyResult<(Bound<'py, PyArray1<f32>>, Bound<'py, PyArray1<f32>>, Bound<'py, PyArray1<f32>>)> {
    let v_ro = v.readonly();
    let r_ro = refrac_until.readonly();
    let c_ro = current.readonly();

    let vv = v_ro.as_slice()?;
    let rr = r_ro.as_slice()?;
    let cc = c_ro.as_slice()?;
    let n = vv.len();

    let dt_over_tau = dt / tau;
    let t_new = t + dt;

    let out = py.allow_threads(|| {
        let mut spikes = vec![0.0f32; n];
        let mut v_out = vec![0.0f32; n];
        let mut r_out = vec![0.0f32; n];
        for i in 0..n {
            let can_spike = rr[i] <= t_new;
            let dv = (-(vv[i] - v_rest) + cc[i]) * dt_over_tau;
            let v_step = if can_spike { vv[i] + dv } else { vv[i] };
            let spiked = can_spike && v_step >= v_threshold;
            spikes[i] = if spiked { 1.0 } else { 0.0 };
            v_out[i] = if spiked { v_reset } else { v_step };
            r_out[i] = if spiked { t_new + refractory } else { rr[i] };
        }
        (spikes, v_out, r_out)
    });

    Ok((
        PyArray1::from_vec_bound(py, out.0),
        PyArray1::from_vec_bound(py, out.1),
        PyArray1::from_vec_bound(py, out.2),
    ))
}

#[pyfunction]
pub fn adaptive_lif_step<'py>(
    py: Python<'py>,
    v: Bound<'py, PyArray1<f32>>,
    refrac_until: Bound<'py, PyArray1<f32>>,
    threshold: Bound<'py, PyArray1<f32>>,
    current: Bound<'py, PyArray1<f32>>,
    t: f32,
    tau: f32,
    v_rest: f32,
    v_threshold: f32,
    v_reset: f32,
    refractory: f32,
    tau_adapt: f32,
    delta_threshold: f32,
    dt: f32,
) -> PyResult<(
    Bound<'py, PyArray1<f32>>,
    Bound<'py, PyArray1<f32>>,
    Bound<'py, PyArray1<f32>>,
    Bound<'py, PyArray1<f32>>,
)> {
    let v_ro = v.readonly();
    let r_ro = refrac_until.readonly();
    let th_ro = threshold.readonly();
    let c_ro = current.readonly();

    let vv = v_ro.as_slice()?;
    let rr = r_ro.as_slice()?;
    let th = th_ro.as_slice()?;
    let cc = c_ro.as_slice()?;
    let n = vv.len();

    let dt_over_tau = dt / tau;
    let dt_over_tau_adapt = dt / tau_adapt;
    let t_new = t + dt;

    let out = py.allow_threads(|| {
        let mut spikes = vec![0.0f32; n];
        let mut v_out = vec![0.0f32; n];
        let mut th_out = vec![0.0f32; n];
        let mut r_out = vec![0.0f32; n];
        for i in 0..n {
            let can_spike = rr[i] <= t_new;
            let dv = (-(vv[i] - v_rest) + cc[i]) * dt_over_tau;
            let v_step = if can_spike { vv[i] + dv } else { vv[i] };
            let spiked = can_spike && v_step >= th[i];
            let th_relaxed = th[i] + (v_threshold - th[i]) * dt_over_tau_adapt;
            spikes[i] = if spiked { 1.0 } else { 0.0 };
            v_out[i] = if spiked { v_reset } else { v_step };
            th_out[i] = if spiked {
                th_relaxed + delta_threshold
            } else {
                th_relaxed
            };
            r_out[i] = if spiked { t_new + refractory } else { rr[i] };
        }
        (spikes, v_out, th_out, r_out)
    });

    Ok((
        PyArray1::from_vec_bound(py, out.0),
        PyArray1::from_vec_bound(py, out.1),
        PyArray1::from_vec_bound(py, out.2),
        PyArray1::from_vec_bound(py, out.3),
    ))
}

#[pyfunction]
pub fn rate_step<'py>(
    py: Python<'py>,
    current: Bound<'py, PyArray1<f32>>,
    gain: f32,
    baseline: f32,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let c_ro = current.readonly();
    let cc = c_ro.as_slice()?;
    let n = cc.len();

    let out = py.allow_threads(|| {
        let mut result = vec![0.0f32; n];
        for i in 0..n {
            result[i] = baseline + gain * cc[i];
        }
        result
    });

    Ok(PyArray1::from_vec_bound(py, out))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(lif_step, m)?)?;
    m.add_function(wrap_pyfunction!(adaptive_lif_step, m)?)?;
    m.add_function(wrap_pyfunction!(rate_step, m)?)?;
    Ok(())
}