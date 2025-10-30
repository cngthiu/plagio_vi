# =========================
# file: src/ranking/features.py
# =========================
import numpy as np
from typing import Iterable, Set

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

def jaccard_char_ngrams(a: str, b: str, n: int = 5) -> float:
    sa = {a[i:i+n] for i in range(max(0, len(a)-n+1))}
    sb = {b[i:i+n] for i in range(max(0, len(b)-n+1))}
    inter = len(sa & sb); uni = len(sa | sb)
    if uni == 0: return 0.0
    return inter / uni

def hamming_similarity(h1: int, h2: int, bits: int = 64) -> float:
    # import local to avoid cycle
    from src.candidate.simhash import hamming
    return 1.0 - hamming(h1, h2) / bits

def boilerplate_penalty(text_a: str, text_b: str, boiler: Set[str]) -> float:
    """
    Trả về tỉ lệ (0..1) biểu thị mức 'nhiễu boilerplate' giữa hai đoạn.
    Cách tính: tỉ lệ số cụm boilerplate xuất hiện ở cả A và B so với tổng cụm đếm được.
    Heuristic đơn giản, chi phí thấp.
    """
    if not boiler:
        return 0.0
    a_hits = {p for p in boiler if p.lower() in text_a.lower()}
    b_hits = {p for p in boiler if p.lower() in text_b.lower()}
    both = a_hits & b_hits
    total = len(a_hits) + len(b_hits) - len(both)
    if total <= 0:
        return 0.0
    return min(1.0, len(both) / total)