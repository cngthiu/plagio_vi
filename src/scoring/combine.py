# file: src/scoring/combine.py
from __future__ import annotations
import numpy as np
from dataclasses import dataclass

@dataclass
class Weights:
    w_cross: float = 0.55
    w_bi: float = 0.25
    w_lex: float = 0.10
    w_bm25: float = 0.10

# very_high/high/medium/low thresholds
LEVELS = {
    "very_high": 0.82,
    "high": 0.72,
    "medium": 0.60,
    "low": 0.0,
}

def _minmax(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return x
    mn, mx = float(np.min(x)), float(np.max(x))
    if mx - mn < 1e-9:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - mn) / (mx - mn)).astype(np.float32)

def fuse_scores(cross: np.ndarray, bi: np.ndarray, lex: np.ndarray, bm25: np.ndarray,
                weights: Weights = Weights()) -> np.ndarray:
    """
    Trả điểm hợp nhất S ∈ [0,1].
    - cross/bi/lex: đã ∈[0,1]
    - bm25: chuẩn hoá min-max theo tập candidate
    """
    bm25n = _minmax(bm25.astype(np.float32))
    S = (weights.w_cross * cross.astype(np.float32) +
         weights.w_bi    * bi.astype(np.float32) +
         weights.w_lex   * lex.astype(np.float32) +
         weights.w_bm25  * bm25n)
    return np.clip(S, 0.0, 1.0)

def level_of(score: float) -> str:
    for name, th in (("very_high", LEVELS["very_high"]),
                     ("high", LEVELS["high"]),
                     ("medium", LEVELS["medium"])):
        if score >= th:
            return name
    return "low"
