from dataclasses import dataclass
import numpy as np
from scipy import sparse

@dataclass
class ConnectomeGraph:
    weights: sparse.csr_matrix
    body_ids: np.ndarray

    def __post_init__(self):
        self.weights = self.weights.tocsr().astype(np.float32)
        self.body_ids = np.asarray(self.body_ids, dtype=np.int64)
        if self.weights.shape[0] != self.weights.shape[1] or self.weights.shape[0] != len(self.body_ids):
            raise ValueError("graph dimensions and body_ids do not match")
        self._index = {int(v): i for i, v in enumerate(self.body_ids)}

    @property
    def n_neurons(self): return self.weights.shape[0]
    @property
    def n_edges(self): return self.weights.nnz

    def index(self, body_id: int) -> int:
        return self._index[int(body_id)]

    def neighbors(self, body_id: int):
        i = self.index(body_id)
        a, b = self.weights.indptr[i:i+2]
        return self.body_ids[self.weights.indices[a:b]], self.weights.data[a:b]

    def save(self, path):
        m = self.weights
        np.savez_compressed(path, data=m.data, indices=m.indices, indptr=m.indptr,
                            shape=np.asarray(m.shape), body_ids=self.body_ids)

    @classmethod
    def load(cls, path):
        z = np.load(path, allow_pickle=False)
        m = sparse.csr_matrix((z["data"], z["indices"], z["indptr"]), shape=tuple(z["shape"]))
        return cls(m, z["body_ids"])
