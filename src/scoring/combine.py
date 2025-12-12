# =========================
# file: src/scoring/combine.py
# =========================
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from src.utils.common import minmax_scale

@dataclass
class Weights:
    w_cross: float = 0.55
    w_bi: float = 0.25
    w_lex: float = 0.10
    w_bm25: float = 0.10

# Các ngưỡng đánh giá mức độ trùng lặp
LEVELS = {
    "very_high": 0.82,
    "high": 0.72,
    "medium": 0.60,
    "low": 0.0,
}

def get_weights_from_cfg(cfg: dict) -> Weights:
    """Trích xuất trọng số từ config dict."""
    w = cfg.get("ranking_weights", {})
    legacy = cfg.get("ranking", {})
    return Weights(
        w_cross=float(w.get("w_cross", legacy.get("alpha", 0.55))),
        w_bi=float(w.get("w_bi", legacy.get("beta", 0.25))),
        w_lex=float(w.get("w_lex", legacy.get("gamma", 0.10))),
        w_bm25=float(w.get("w_bm25", legacy.get("delta", 0.10))),
    )

def fuse_scores(cross: list | np.ndarray, bi: list | np.ndarray, 
                lex: list | np.ndarray, bm25: list | np.ndarray, 
                weights: Weights) -> list[float]:
    """
    Tính điểm tổng hợp S từ các thành phần.
    Trả về list[float] đã được clip trong đoạn [0, 1].
    """
    c = np.array(cross, dtype=np.float32)
    b = np.array(bi, dtype=np.float32)
    l = np.array(lex, dtype=np.float32)
    bm = minmax_scale(np.array(bm25, dtype=np.float32))

    S = (weights.w_cross * c +
         weights.w_bi    * b +
         weights.w_lex   * l +
         weights.w_bm25  * bm)
    return np.clip(S, 0.0, 1.0).tolist()

def level_of(score: float) -> str:
    """Xác định mức độ trùng lặp dựa trên điểm số."""
    for name, th in [("very_high", LEVELS["very_high"]),
                     ("high", LEVELS["high"]),
                     ("medium", LEVELS["medium"])]:
        if score >= th:
            return name
    return "low"