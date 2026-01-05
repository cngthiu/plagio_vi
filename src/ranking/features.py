# =========================
# file: src/ranking/features.py
# =========================
import re
import numpy as np
from typing import Iterable, List, Set

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

def stopphrase_penalty(text_a: str, text_b: str, stop_phrases: Set[str]) -> float:
    if not stop_phrases:
        return 0.0
    a_hits = {p for p in stop_phrases if p in text_a}
    b_hits = {p for p in stop_phrases if p in text_b}
    both = a_hits & b_hits
    total = len(a_hits) + len(b_hits) - len(both)
    if total <= 0:
        return 0.0
    return min(1.0, len(both) / total)

def citation_penalty(text_a: str, text_b: str, patterns: List[str]) -> float:
    if not patterns:
        return 0.0
    regs = [re.compile(p) for p in patterns]
    a_hits = sum(1 for r in regs if r.search(text_a))
    b_hits = sum(1 for r in regs if r.search(text_b))
    if a_hits == 0 or b_hits == 0:
        return 0.0
    return min(1.0, min(a_hits, b_hits) / max(a_hits, b_hits))
