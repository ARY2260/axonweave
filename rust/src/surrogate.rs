use numpy::{PyArray1, PyArrayMethods};
use pyo3::prelude::*;

const KIND_SIGMOID: i32 = 0;
const KIND_ATAN: i32 = 1;
const KIND_PIECEWISE: i32 = 2;
const KIND_STE: i32 = 3;

#[pyfunction]
pub fn surrogate_forward<'py>(
    py: Python<'py>,
    v: Bound<'py, PyArray1<f32>>,
    threshold: f32,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let v_ro = v.readonly();
    let vv = v_ro.as_slice()?;
    let n = vv.len();

    let out = py.allow_threads(|| {
        let mut result = vec![0.0f32; n];
        for i in 0..n {
            result[i] = if vv[i] >= threshold { 1.0 } else { 0.0 };
        }
        result
    });

    Ok(PyArray1::from_vec_bound(py, out))
}

#[pyfunction]
pub fn surrogate_backward<'py>(
    py: Python<'py>,
    v: Bound<'py, PyArray1<f32>>,
    threshold: f32,
    kind: i32,
    k: f32,
    width: f32,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let v_ro = v.readonly();
    let vv = v_ro.as_slice()?;
    let n = vv.len();

    let out = py.allow_threads(|| {
        let mut result = vec![0.0f32; n];
        for i in 0..n {
            let x = vv[i] - threshold;
            result[i] = match kind {
                KIND_SIGMOID => {
                    let clipped = if x * k > 20.0 {
                        20.0
                    } else if x * k < -20.0 {
                        -20.0
                    } else {
                        x * k
                    };
                    let sigma = 1.0 / (1.0 + (-clipped).exp());
                    k * sigma * (1.0 - sigma)
                }
                KIND_ATAN => {
                    let t = std::f32::consts::PI * k * x;
                    k / (1.0 + t * t)
                }
                KIND_PIECEWISE => {
                    if x.abs() <= 1.0 / k {
                        k
                    } else {
                        0.0
                    }
                }
                KIND_STE => {
                    if x.abs() <= width {
                        1.0
                    } else {
                        0.0
                    }
                }
                _ => 0.0,
            };
        }
        result
    });

    Ok(PyArray1::from_vec_bound(py, out))
}

fn surrogate_grad(kind: i32, x: f32, k: f32, width: f32) -> f32 {
    match kind {
        KIND_SIGMOID => {
            let clipped = if x * k > 20.0 {
                20.0
            } else if x * k < -20.0 {
                -20.0
            } else {
                x * k
            };
            let sigma = 1.0 / (1.0 + (-clipped).exp());
            k * sigma * (1.0 - sigma)
        }
        KIND_ATAN => {
            let t = std::f32::consts::PI * k * x;
            k / (1.0 + t * t)
        }
        KIND_PIECEWISE => {
            if x.abs() <= 1.0 / k {
                k
            } else {
                0.0
            }
        }
        KIND_STE => {
            if x.abs() <= width {
                1.0
            } else {
                0.0
            }
        }
        _ => 0.0,
    }
}

#[pyfunction]
pub fn surrogate_lif_step<'py>(
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
    kind: i32,
    k: f32,
    width: f32,
) -> PyResult<(
    Bound<'py, PyArray1<f32>>,
    Bound<'py, PyArray1<f32>>,
    Bound<'py, PyArray1<f32>>,
    Bound<'py, PyArray1<f32>>,
)> {
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
        let mut grad = vec![0.0f32; n];
        for i in 0..n {
            let can_spike = rr[i] <= t_new;
            let dv = (-(vv[i] - v_rest) + cc[i]) * dt_over_tau;
            let v_step = if can_spike { vv[i] + dv } else { vv[i] };
            let x = v_step - v_threshold;
            if can_spike {
                grad[i] = surrogate_grad(kind, x, k, width);
            }
            let spiked = can_spike && v_step >= v_threshold;
            spikes[i] = if spiked { 1.0 } else { 0.0 };
            v_out[i] = if spiked { v_reset } else { v_step };
            r_out[i] = if spiked { t_new + refractory } else { rr[i] };
        }
        (spikes, v_out, r_out, grad)
    });

    Ok((
        PyArray1::from_vec_bound(py, out.0),
        PyArray1::from_vec_bound(py, out.1),
        PyArray1::from_vec_bound(py, out.2),
        PyArray1::from_vec_bound(py, out.3),
    ))
}

#[pyfunction]
pub fn surrogate_adaptive_lif_step<'py>(
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
    kind: i32,
    k: f32,
    width: f32,
) -> PyResult<(
    Bound<'py, PyArray1<f32>>,
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
        let mut grad = vec![0.0f32; n];
        for i in 0..n {
            let can_spike = rr[i] <= t_new;
            let dv = (-(vv[i] - v_rest) + cc[i]) * dt_over_tau;
            let v_step = if can_spike { vv[i] + dv } else { vv[i] };
            let x = v_step - th[i];
            if can_spike {
                grad[i] = surrogate_grad(kind, x, k, width);
            }
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
        (spikes, v_out, th_out, r_out, grad)
    });

    Ok((
        PyArray1::from_vec_bound(py, out.0),
        PyArray1::from_vec_bound(py, out.1),
        PyArray1::from_vec_bound(py, out.2),
        PyArray1::from_vec_bound(py, out.3),
        PyArray1::from_vec_bound(py, out.4),
    ))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(surrogate_forward, m)?)?;
    m.add_function(wrap_pyfunction!(surrogate_backward, m)?)?;
    m.add_function(wrap_pyfunction!(surrogate_lif_step, m)?)?;
    m.add_function(wrap_pyfunction!(surrogate_adaptive_lif_step, m)?)?;
    Ok(())
}