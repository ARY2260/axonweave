use numpy::{PyArray1, PyArrayMethods};
use pyo3::prelude::*;

#[pyfunction]
pub fn vesicle_release_step<'py>(
    py: Python<'py>,
    vesicle_pool: Bound<'py, PyArray1<f64>>,
    concentration: Bound<'py, PyArray1<f64>>,
    pre_activity: Bound<'py, PyArray1<f64>>,
    release_probs: Bound<'py, PyArray1<f64>>,
    decay_rates: Bound<'py, PyArray1<f64>>,
    stochastic: Bound<'py, PyArray1<f64>>,
    signs: Bound<'py, PyArray1<f64>>,
    weights: Bound<'py, PyArray1<f64>>,
) -> PyResult<(
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    f64,
)> {
    let vp_ro = vesicle_pool.readonly();
    let conc_ro = concentration.readonly();
    let pre_ro = pre_activity.readonly();
    let rel_ro = release_probs.readonly();
    let dec_ro = decay_rates.readonly();
    let sto_ro = stochastic.readonly();
    let sign_ro = signs.readonly();
    let w_ro = weights.readonly();

    let vp = vp_ro.as_slice()?;
    let conc = conc_ro.as_slice()?;
    let pre = pre_ro.as_slice()?;
    let rel = rel_ro.as_slice()?;
    let dec = dec_ro.as_slice()?;
    let sto = sto_ro.as_slice()?;
    let sign = sign_ro.as_slice()?;
    let w = w_ro.as_slice()?;

    let n = vp.len();

    let out = py.allow_threads(|| {
        let mut conc_new = vec![0.0f64; n];
        let mut vp_new = vec![0.0f64; n];
        let mut per_syn = vec![0.0f64; n];
        let mut post_current = 0.0f64;
        for i in 0..n {
            let will_release = sto[i] < rel[i] * pre[i];
            let release_amount = if will_release { vp[i] } else { 0.0 };

            let mut c = conc[i] * dec[i] + release_amount;
            c = c.clamp(0.0, 1.0);

            let mut v = (vp[i] - release_amount).clamp(0.0, 1.0);
            v = v + (1.0 - v) * (1.0 - dec[i]);

            let per = c * sign[i] * w[i];
            conc_new[i] = c;
            vp_new[i] = v;
            per_syn[i] = per;
            post_current += per;
        }
        (conc_new, vp_new, per_syn, post_current)
    });

    Ok((
        PyArray1::from_vec_bound(py, out.0),
        PyArray1::from_vec_bound(py, out.1),
        PyArray1::from_vec_bound(py, out.2),
        out.3,
    ))
}

#[pyfunction]
pub fn nt_currents<'py>(
    py: Python<'py>,
    signs: Bound<'py, PyArray1<f64>>,
    weights: Bound<'py, PyArray1<f64>>,
    pre_activity: Bound<'py, PyArray1<f64>>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let sign_ro = signs.readonly();
    let w_ro = weights.readonly();
    let pre_ro = pre_activity.readonly();

    let sign = sign_ro.as_slice()?;
    let w = w_ro.as_slice()?;
    let pre = pre_ro.as_slice()?;
    let n = sign.len();

    let out = py.allow_threads(|| {
        let mut result = vec![0.0f64; n];
        for i in 0..n {
            result[i] = sign[i] * w[i] * pre[i];
        }
        result
    });

    Ok(PyArray1::from_vec_bound(py, out))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(vesicle_release_step, m)?)?;
    m.add_function(wrap_pyfunction!(nt_currents, m)?)?;
    Ok(())
}