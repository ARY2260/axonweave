import numpy as np
from scipy import sparse
from axonweave import ConnectomeGraph

def test_graph_roundtrip(tmp_path):
    g = ConnectomeGraph(sparse.csr_matrix(np.eye(3, dtype=np.float32)), np.array([10,20,30]))
    p = tmp_path / 'g.npz'
    g.save(p)
    h = ConnectomeGraph.load(p)
    assert h.n_neurons == 3
    assert h.n_edges == 3
    assert h.neighbors(20)[0].tolist() == [20]
