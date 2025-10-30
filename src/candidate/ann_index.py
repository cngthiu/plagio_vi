# =========================
# file: src/candidate/ann_index.py
# =========================
from __future__ import annotations
from typing import Optional, List
import numpy as np

try:
    import faiss
except Exception:
    faiss = None

class ANNIndex:
    """
    FAISS HNSW (cosine/L2). Nếu FAISS không khả dụng → fallback brute-force.
    - dim: có thể None; nếu None sẽ suy ra từ emb trong build().
    """
    def __init__(self, dim: Optional[int], cfg: dict):
        self.dim = dim
        self.cfg = cfg
        self.index = None
        self._emb = None  # dùng cho brute-force

    def build(self, emb: np.ndarray):
        if self.dim is None:
            self.dim = int(emb.shape[1])

        # Fallback brute-force nếu không có FAISS hoặc không dùng HNSW
        if faiss is None or self.cfg.get("type", "HNSW") != "HNSW":
            self.index = None
            self._emb = emb.astype("float32")
            return

        metric = self.cfg.get("metric", "cosine")
        if metric == "cosine":
            faiss.normalize_L2(emb)

        m = int(self.cfg.get("hnsw", {}).get("M", 64))
        ef_c = int(self.cfg.get("hnsw", {}).get("ef_construction", 200))
        ef_s = int(self.cfg.get("hnsw", {}).get("ef_search", 64))

        # Cosine = inner product; L2 = L2
        metric_type = faiss.METRIC_INNER_PRODUCT if metric == "cosine" else faiss.METRIC_L2
        self.index = faiss.IndexHNSWFlat(self.dim, m, metric_type)
        self.index.hnsw.efConstruction = ef_c
        self.index.hnsw.efSearch = ef_s
        self.index.add(emb.astype("float32"))

    def search(self, q: np.ndarray, topk: int = 50) -> List[int]:
        if self.index is None:
            # brute-force
            if self._emb is None or self._emb.size == 0:
                return []
            A = q.astype("float32")
            B = self._emb
            if self.cfg.get("metric", "cosine") == "cosine":
                A = A / (np.linalg.norm(A) + 1e-9)
                Bn = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-9)
                sims = (Bn @ A)
                idx = np.argsort(-sims)[:topk]
            else:
                d = np.linalg.norm(B - A[None, :], axis=1)
                idx = np.argsort(d)[:topk]
            return idx.tolist()

        qv = q.astype("float32")[None, :]
        if self.cfg.get("metric", "cosine") == "cosine" and faiss is not None:
            faiss.normalize_L2(qv)
        D, I = self.index.search(qv, topk)
        return I[0].tolist()

    def save(self, path):
        if faiss is not None and self.index is not None:
            faiss.write_index(self.index, str(path))