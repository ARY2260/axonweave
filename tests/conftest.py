import numpy as np
import pytest
from scipy import sparse

from axonweave import ConnectomeGraph


def make_graph(n=8, seed=7):
    """Small deterministic directed graph used across layer tests."""
    rng = np.random.default_rng(seed)
    dense = (rng.random((n, n)) > 0.5).astype(np.float32)
    np.fill_diagonal(dense, 0.0)
    m = sparse.csr_matrix(dense)
    return ConnectomeGraph(m, np.arange(n, dtype=np.int64) * 10)


@pytest.fixture
def graph():
    return make_graph()


@pytest.fixture
def torch():
    pytest.importorskip("torch")
    import torch  # noqa: F401

    return torch
