# =========================
# file: src/alignment/tiling.py
# =========================
from typing import List, Tuple
import numpy as np

def _merge_nearby_spans(matches: List[Tuple[int, int, int, int]], gap: int = 10) -> List[Tuple[int, int, int, int]]:
    merged: List[Tuple[int, int, int, int]] = []
    for s in sorted(matches):
        if not merged:
            merged.append(s)
            continue
        a0, a1, b0, b1 = s
        A0, A1, B0, B1 = merged[-1]
        if a0 <= A1 + gap and b0 <= B1 + gap:
            merged[-1] = (A0, max(A1, a1), B0, max(B1, b1))
        else:
            merged.append(s)
    return merged

def _gst_core_logic(a_codes, b_codes, min_match_len):
    """Logic core của thuật toán Greedy String Tiling."""
    n = len(a_codes)
    m = len(b_codes)
    marked_a = [0] * n
    marked_b = [0] * m
    matches = []
    max_match = min_match_len
    
    while max_match >= min_match_len:
        max_match = 0
        new_matches = []
        for i in range(n):
            for j in range(m):
                k = 0
                while (
                    i + k < n
                    and j + k < m
                    and a_codes[i + k] == b_codes[j + k]
                    and marked_a[i + k] == 0
                    and marked_b[j + k] == 0
                ):
                    k += 1
                if k >= min_match_len:
                    new_matches.append((i, i + k, j, j + k))
                    if k > max_match:
                        max_match = k
        if len(new_matches) == 0:
            break
        for t in new_matches:
            i0, i1, j0, j1 = t
            for x in range(i0, i1):
                marked_a[x] = 1
            for y in range(j0, j1):
                marked_b[y] = 1
            matches.append(t)
    return matches

# Thử compile bằng Numba
_gst_jit = None
try:
    from numba import njit
    _gst_jit = njit(_gst_core_logic)
except ImportError:
    pass

def greedy_string_tiling(a: str, b: str, min_match_len: int = 20) -> List[Tuple[int, int, int, int]]:
    """
    Trả về danh sách (a_start, a_end, b_start, b_end).
    Ưu tiên dùng bản Numba JIT để tăng tốc.
    """
    if _gst_jit:
        try:
            # Chuyển string sang mảng int để numba xử lý
            a_codes = np.array([ord(c) for c in a], dtype=np.int32)
            b_codes = np.array([ord(c) for c in b], dtype=np.int32)
            matches_nb = _gst_jit(a_codes, b_codes, min_match_len)
            # Convert numba list/tuple back to python int tuple
            matches = [(int(t[0]), int(t[1]), int(t[2]), int(t[3])) for t in matches_nb]
            return _merge_nearby_spans(matches, gap=10)
        except Exception:
            pass # Fallback nếu lỗi runtime numba

    # Fallback Python thuần
    a_codes = [ord(c) for c in a]
    b_codes = [ord(c) for c in b]
    matches = _gst_core_logic(a_codes, b_codes, min_match_len)
    return _merge_nearby_spans(matches, gap=10)