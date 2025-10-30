# file: src/preprocess/span_mapper.py
from __future__ import annotations
from typing import List, Tuple

def build_char_map(orig: str, norm: str, norm_to_orig_idx: List[int]) -> List[int]:
    """
    norm_to_orig_idx: mảng same-length với `norm`, tại mỗi char idx của `norm`
    cho biết vị trí char tương ứng trong `orig`. Sinh ra trong bước normalize.
    """
    return norm_to_orig_idx  # đã đảm bảo 1-1 sau tokenize/normalize

def map_span_to_orig(span_norm: Tuple[int,int], cmap: List[int]) -> Tuple[int,int]:
    s, e = span_norm
    s = max(0, min(s, len(cmap)-1))
    e = max(0, min(e, len(cmap)))
    return (cmap[s], cmap[e-1]+1) if e>0 else (cmap[s], cmap[s])
