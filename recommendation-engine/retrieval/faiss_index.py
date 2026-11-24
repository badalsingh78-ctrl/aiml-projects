"""
Milestone 4: FAISS Vector Index for Sub-Millisecond Nearest-Neighbor Candidate Retrieval.

Supports:
- `IndexFlatIP`: Exact inner-product search (cosine similarity on L2-normalized embeddings).
- `IndexIVFFlat`: Inverted file index for 100k–10M+ catalog scale with tunable `nlist` & `nprobe`.
- `IndexHNSWFlat`: Hierarchical Navigable Small World graph for ultra-low-latency ANN search.
- Automatic NumPy BLAS inner-product fallback when `faiss` is not installed.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


class VectorCandidateIndex:
    """Unified FAISS + NumPy-fallback ANN retrieval index."""

    def __init__(self, dim: int = 64, index_type: str = "flat_ip", nlist: int = 16, nprobe: int = 4):
        self.dim = dim
        self.index_type = index_type
        self.nlist = nlist
        self.nprobe = nprobe
        self.item_ids: np.ndarray = np.array([], dtype=np.int64)
        self._embeddings: np.ndarray = np.empty((0, dim), dtype=np.float32)
        self._faiss_index = None
        self.backend = "faiss" if FAISS_AVAILABLE else "numpy_exact_ip"

    @staticmethod
    def _l2_normalize(vectors: np.ndarray) -> np.ndarray:
        vecs = np.asarray(vectors, dtype=np.float32)
        norms = np.linalg.norm(vecs, axis=-1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return vecs / norms

    def build(self, item_embeddings: np.ndarray, item_ids: np.ndarray) -> Dict[str, object]:
        """Build and train the vector index over L2-normalized item embeddings."""
        t0 = time.perf_counter()
        vecs = self._l2_normalize(item_embeddings)
        self.dim = vecs.shape[1]
        self.item_ids = np.asarray(item_ids, dtype=np.int64)
        self._embeddings = vecs

        if FAISS_AVAILABLE:
            if self.index_type == "ivf_flat" and len(vecs) >= self.nlist * 4:
                quantizer = faiss.IndexFlatIP(self.dim)
                index = faiss.IndexIVFFlat(quantizer, self.dim, self.nlist, faiss.METRIC_INNER_PRODUCT)
                index.train(vecs)
                index.add(vecs)
                index.nprobe = self.nprobe
            elif self.index_type == "hnsw":
                index = faiss.IndexHNSWFlat(self.dim, 32, faiss.METRIC_INNER_PRODUCT)
                index.add(vecs)
            else:
                index = faiss.IndexFlatIP(self.dim)
                index.add(vecs)
            self._faiss_index = index

        build_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "backend": self.backend,
            "index_type": self.index_type,
            "num_items": int(len(self.item_ids)),
            "dim": int(self.dim),
            "build_time_ms": round(build_ms, 3),
        }

    def search(self, query_embedding: np.ndarray, top_k: int = 100) -> Tuple[List[int], List[float], float]:
        """Retrieve top-K candidate item IDs and cosine similarity scores."""
        t0 = time.perf_counter()
        q = self._l2_normalize(np.asarray(query_embedding, dtype=np.float32).reshape(1, -1))
        k = min(top_k, len(self.item_ids))

        if FAISS_AVAILABLE and self._faiss_index is not None:
            scores, indices = self._faiss_index.search(q, k)
            cand_ids = [int(self.item_ids[i]) for i in indices[0] if i >= 0]
            cand_scores = [float(s) for s, i in zip(scores[0], indices[0]) if i >= 0]
        else:
            sims = (self._embeddings @ q.T).ravel()
            top_idx = np.argpartition(-sims, k - 1)[:k]
            top_idx = top_idx[np.argsort(-sims[top_idx])]
            cand_ids = [int(self.item_ids[i]) for i in top_idx]
            cand_scores = [float(sims[i]) for i in top_idx]

        latency_ms = (time.perf_counter() - t0) * 1000.0
        return cand_ids, cand_scores, latency_ms

    def save(self, out_dir: Path) -> None:
        """Persist FAISS index + ID mapping to disk."""
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / "item_embeddings.npy", self._embeddings)
        np.save(out_dir / "item_ids.npy", self.item_ids)
        meta = {"dim": self.dim, "index_type": self.index_type, "backend": self.backend}
        (out_dir / "faiss_meta.json").write_text(json.dumps(meta, indent=2))
        if FAISS_AVAILABLE and self._faiss_index is not None:
            faiss.write_index(self._faiss_index, str(out_dir / "items.faiss"))

    @classmethod
    def load(cls, out_dir: Path) -> "VectorCandidateIndex":
        """Load persisted vector index from disk."""
        meta = json.loads((out_dir / "faiss_meta.json").read_text())
        obj = cls(dim=meta["dim"], index_type=meta.get("index_type", "flat_ip"))
        embs = np.load(out_dir / "item_embeddings.npy")
        ids = np.load(out_dir / "item_ids.npy")
        obj.build(embs, ids)
        return obj
