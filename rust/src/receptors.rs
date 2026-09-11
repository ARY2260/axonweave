use numpy::{PyArray1, PyArrayMethods};
use pyo3::prelude::*;

const KIND_AMPA: i32 = 0;
const KIND_GABA: i32 = 1;
const KIND_NMDA: i32 = 2;
const KIND_DOPAMINE: i32 = 3;

#[pyfunction]
pub fn receptor_step<'py>(
    py: Python<'py>,
    g: Bound<'py, PyArray1<f32>>,
    pre_activity: Bound<'py, PyArray1<f32>>,
    voltage: Bound<'py, PyArray1<f32>>,
    dt: f32,
    kind: i32,
    decay_time_constant: f32,
    reverse_potential: f32,
    gain: f32,
    sign: f32,
    mg_concentration: f32,
    mg_slope: f32,
    mg_offset: f32,
) -> PyResult<(Bound<'py, PyArray1<f32>>, Bound<'py, PyArray1<f32>>)> {
    let g_ro = g.readonly();
    let pre_ro = pre_activity.readonly();
    let vol_ro = voltage.readonly();

    let gv = g_ro.as_slice()?;
    let pre = pre_ro.as_slice()?;
    let vol = vol_ro.as_slice()?;
    let n = gv.len();

    let dt_over_tau = dt / decay_time_constant;

    let out = py.allow_threads(|| {
        let mut current = vec![0.0f32; n];
        let mut g_new = vec![0.0f32; n];
        for i in 0..n {
            let c = match kind {
                KIND_AMPA | KIND_GABA => gv[i] * reverse_potential,
                KIND_NMDA => {
                    let mg_block =
                        1.0f32 / (1.0 + mg_concentration * (-mg_slope * (vol[i] - mg_offset)).exp());
                    gv[i] * mg_block * reverse_potential
                }
                KIND_DOPAMINE => gv[i] * gain * sign,
                _ => 0.0,
            };
            current[i] = c;
            g_new[i] = gv[i] + (pre[i] - gv[i]) * dt_over_tau;
        }
        (current, g_new)
    });

    Ok((
        PyArray1::from_vec_bound(py, out.0),
        PyArray1::from_vec_bound(py, out.1),
    ))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(receptor_step, m)?)?;
    Ok(())
}