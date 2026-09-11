import numpy as np
from scipy import sparse

from axonweave import ConnectomeGraph, native


def test_sparse_matmul_matches_scipy():
    rng = np.random.default_rng(0)
    W = sparse.random(8, 6, density=0.4, random_state=0, format="csr")
    x = rng.normal(size=6)
    assert np.allclose(native.sparse_matmul(W, x), W.dot(x).astype(np.float32))


def test_build_csr_sums_duplicates():
    rng = np.random.default_rng(0)
    rows = rng.integers(0, 3, size=20).astype(np.int64)
    cols = rng.integers(0, 4, size=20).astype(np.int64)
    weights = rng.normal(size=20).astype(np.float32)
    got = native.build_csr_from_coo(rows, cols, weights, (3, 4))
    expected = sparse.coo_matrix((weights, (rows, cols)), shape=(3, 4)).tocsr()
    assert got.shape == expected.shape
    assert np.allclose(got.toarray(), expected.toarray())


def test_apply_delayed_propagation_first_step():
    g = ConnectomeGraph(
        sparse.csr_matrix(np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 2.0], [0.0, 0.0, 0.0]], dtype=np.float32)),
        np.array([1, 2, 3]),
    )
    pre = np.array([3.0, 4.0, 5.0], dtype=np.float32)
    delays = np.zeros(g.n_edges, dtype=np.float32)
    out = native.apply_delayed_propagation(g.weights, pre, delays, 5.0, 1.0)
    assert np.allclose(out, g.weights.T.dot(pre))


def test_apply_delayed_propagation_defers_positive_delays():
    g = ConnectomeGraph(
        sparse.csr_matrix(np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32)),
        np.array([1, 2, 3]),
    )
    pre = np.array([3.0, 4.0, 5.0], dtype=np.float32)
    delays = np.array([2.0], dtype=np.float32)
    out = native.apply_delayed_propagation(g.weights, pre, delays, 5.0, 1.0)
    assert np.allclose(out, np.zeros(g.n_neurons, dtype=np.float32))