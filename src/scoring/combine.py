# =========================
# file: src/scoring/combine.py
# =========================
from __future__ import annotations
import numpy as np
from dataclasses import dataclass

@dataclass
class Weights:
    w_cross: float = 0.55
    w_bi: float = 0.25
    w_lex: float = 0.15
    w_bm25: float = 0.05

# Ngưỡng đánh giá
LEVELS = {
    "very_high": 0.82,
    "high": 0.72,
    "medium": 0.60,
    "low": 0.0,
}

def get_weights_from_cfg(cfg: dict) -> Weights:
    """Trích xuất trọng số từ config dict."""
    w = cfg.get("ranking_weights", {})
    return Weights(
        w_cross=float(w.get("alpha", 0.55)),
        w_bi=float(w.get("beta", 0.25)),
        w_lex=float(w.get("gamma", 0.15)),
        w_bm25=float(w.get("delta", 0.05)),
    )

def robust_bm25_norm(scores: np.ndarray, k: float = 10.0) -> np.ndarray:
    """
    Chuẩn hóa điểm BM25 sử dụng hàm bão hòa (Saturation).
    Công thức: S = 1 - exp(-score / k)
    - Giải quyết vấn đề outlier của MinMax.
    - k: hệ số scale (BM25 thường từ 10-30, chọn k=10 là hợp lý).
    """
    # Đảm bảo không âm
    s = np.maximum(scores, 0.0)
    return 1.0 - np.exp(-s / k)

def fuse_scores(cross: list | np.ndarray, bi: list | np.ndarray, 
                lex: list | np.ndarray, bm25: list | np.ndarray, 
                weights: Weights) -> list[float]:
    """
    Tính điểm tổng hợp S.
    """
    c = np.array(cross, dtype=np.float32)
    b = np.array(bi, dtype=np.float32)
    l = np.array(lex, dtype=np.float32)
    
    # [UPDATE] Fix logic sai lầm khi dùng MinMax cho BM25
    bm_raw = np.array(bm25, dtype=np.float32)
    bm = robust_bm25_norm(bm_raw, k=10.0)

    S = (weights.w_cross * c +
         weights.w_bi    * b +
         weights.w_lex   * l +
         weights.w_bm25  * bm)
    
    return np.clip(S, 0.0, 1.0).tolist()

def level_of(score: float) -> str:
    for name, th in [("very_high", LEVELS["very_high"]),
                     ("high", LEVELS["high"]),
                     ("medium", LEVELS["medium"])]:
        if score >= th:
            return name
    return "low"