from __future__ import annotations

import numpy as np

from .. import native as _native


class TokenEncoder:
    """Maps discrete token IDs to input currents via a fixed embedding.

    ``token_ids`` may be shape (batch,) or (batch, seq). Output currents
    have shape (batch, n_target) or (batch, seq, n_target).
    """

    def __init__(self, vocab_size: int, n_target: int, embedding_dim: int = 128,
                 seed: int = 0, select: str | None = None):
        self.vocab_size = vocab_size
        self.n_target = n_target
        self.select = select
        rng = np.random.default_rng(seed)
        # Fixed random embedding: token -> embedding_dim -> n_target currents.
        self.embedding = (rng.standard_normal((vocab_size, embedding_dim)) *
                          (1.0 / np.sqrt(embedding_dim))).astype(np.float32)
        self.project = (rng.standard_normal((embedding_dim, n_target)) *
                        (1.0 / np.sqrt(embedding_dim))).astype(np.float32)

    def __call__(self, token_ids) -> np.ndarray:
        ids = np.asarray(token_ids, dtype=np.int64)
        if ids.min(initial=0) < 0 or ids.max(initial=0) >= self.vocab_size:
            from ..errors import ApiUsageError
            raise ApiUsageError(
                f"AXW010: token ids must be in [0, {self.vocab_size}); "
                f"got range [{ids.min()}, {ids.max()}]"
            )
        emb = _native.embed_lookup(ids, self.embedding)      # (n_tokens, embedding_dim)
        out = _native.dense_matmul(emb, self.project)         # (n_tokens, n_target)
        return out.reshape(*ids.shape, self.n_target)
