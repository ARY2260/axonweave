use base64::{engine::general_purpose::STANDARD, Engine as _};
use md5::Md5;
use pyo3::prelude::*;
use sha2::{Digest, Sha256};

use std::fs::File;
use std::io::Read;
use std::path::Path;

const CHUNK_SIZE: usize = 8 * 1024 * 1024;

#[pyfunction]
pub fn sha256_file(path: &str) -> PyResult<String> {
    let mut file = File::open(Path::new(path))
        .map_err(|e| pyo3::exceptions::PyOSError::new_err(format!("{}: {}", path, e)))?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; CHUNK_SIZE];
    loop {
        let n = file
            .read(&mut buf)
            .map_err(|e| pyo3::exceptions::PyOSError::new_err(format!("{}: {}", path, e)))?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    let digest = hasher.finalize();
    let hex: String = digest.iter().map(|b| format!("{:02x}", b)).collect();
    Ok(hex)
}

#[pyfunction]
pub fn md5_base64_file(path: &str) -> PyResult<String> {
    let mut file = File::open(Path::new(path))
        .map_err(|e| pyo3::exceptions::PyOSError::new_err(format!("{}: {}", path, e)))?;
    let mut hasher = Md5::new();
    let mut buf = vec![0u8; CHUNK_SIZE];
    loop {
        let n = file
            .read(&mut buf)
            .map_err(|e| pyo3::exceptions::PyOSError::new_err(format!("{}: {}", path, e)))?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    let digest = hasher.finalize();
    Ok(STANDARD.encode(digest))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sha256_file, m)?)?;
    m.add_function(wrap_pyfunction!(md5_base64_file, m)?)?;
    Ok(())
}