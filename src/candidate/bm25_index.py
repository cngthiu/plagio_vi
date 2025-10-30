# file: src/candidate/bm25_index.py  (REPLACE)
from typing import List
import numpy as np
from rank_bm25 import BM25Okapi
from src.preprocess.tokenizer_vi import simple_tokenize


class BM25Index:
    def __init__(self, corpus_chunks: List[str]):
        self.docs = [simple_tokenize(c) for c in corpus_chunks]
        self.model = BM25Okapi(self.docs)
        self._last_scores: np.ndarray | None = None

    def search(self, query_text: str, topk: int = 30) -> List[int]:
        query = simple_tokenize(query_text)
        scores = self.model.get_scores(query)  # list-like
        # store as numpy array for safe indexing
        self._last_scores = np.asarray(scores, dtype=np.float32)
        idx = np.argsort(-self._last_scores)[:topk]
        return idx.tolist()

    def score_of_cached(self, idx: int) -> float:
        # Why: tránh truth-value ambiguous của numpy array
        if self._last_scores is None:
            return 0.0
        if idx < 0 or idx >= self._last_scores.shape[0]:
            return 0.0
        return float(self._last_scores[idx])
