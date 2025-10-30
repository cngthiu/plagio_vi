# file: src/candidate/postfilter.py
from __future__ import annotations
import numpy as np
from typing import List, Tuple

def iou_span(a: Tuple[int,int], b: Tuple[int,int]) -> float:
    l = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    u = (a[1]-a[0]) + (b[1]-b[0]) - l
    return (l / u) if u else 0.0

def nms_spans(spans: List[Tuple[int,int]], scores: List[float], iou_th: float=0.5) -> List[int]:
    order = np.argsort(-np.asarray(scores))
    kept = []
    used = np.zeros(len(spans), dtype=bool)
    for i in order:
        if used[i]: continue
        kept.append(i)
        for j in order:
            if used[j] or j == i: continue
            if iou_span(spans[i], spans[j]) >= iou_th:
                used[j] = True
    return kept

def per_chunk_topk(indices: np.ndarray, scores: np.ndarray, a_chunk_ids: np.ndarray, topk:int=5):
    out = []
    for aid in np.unique(a_chunk_ids):
        mask = (a_chunk_ids == aid)
        sub_idx = indices[mask]
        sub_scores = scores[mask]
        ord_ = np.argsort(-sub_scores)[:topk]
        out.append((sub_idx[ord_], sub_scores[ord_]))
    if not out:
        return np.array([], dtype=int), np.array([], dtype=float)
    idx = np.concatenate([x[0] for x in out])
    sc  = np.concatenate([x[1] for x in out])
    return idx, sc
