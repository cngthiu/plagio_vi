# =========================
# file: src/alignment/tiling.py
# =========================
from typing import List, Tuple

try:
    from numba import njit, types
    from numba.typed import List as NumbaList
except Exception:
    njit = None  # type: ignore


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


def _try_compile_gst():
    if njit is None:
        return None

    @njit
    def gst_numba(a_codes, b_codes, min_match_len: int):
        n = len(a_codes)
        m = len(b_codes)
        marked_a = [0] * n
        marked_b = [0] * m
        matches = NumbaList.empty_list(types.UniTuple(types.int64, 4))
        max_match = min_match_len
        while max_match >= min_match_len:
            max_match = 0
            new_matches = NumbaList.empty_list(types.UniTuple(types.int64, 4))
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

    return gst_numba


_GST_NUMBA = _try_compile_gst()


def greedy_string_tiling(a: str, b: str, min_match_len: int = 20) -> List[Tuple[int, int, int, int]]:
    """
    Returns list of (a_start, a_end, b_start, b_end)
    Ưu tiên bản numba JIT để tăng tốc; fallback Python nếu không có numba.
    """
    if _GST_NUMBA is not None:
        try:
            a_codes = [ord(c) for c in a]
            b_codes = [ord(c) for c in b]
            matches_nb = _GST_NUMBA(a_codes, b_codes, min_match_len)
            matches = [(int(t[0]), int(t[1]), int(t[2]), int(t[3])) for t in matches_nb]
            return _merge_nearby_spans(matches, gap=10)
        except Exception:
            pass  # fallback

    A, B = a, b
    matches: List[Tuple[int, int, int, int]] = []
    marked_a = set()
    marked_b = set()
    max_match = min_match_len
    while max_match >= min_match_len:
        max_match = 0
        new_matches: List[Tuple[int, int, int, int]] = []
        for i in range(len(A)):
            for j in range(len(B)):
                k = 0
                while (
                    i + k < len(A)
                    and j + k < len(B)
                    and A[i + k] == B[j + k]
                    and (i + k) not in marked_a
                    and (j + k) not in marked_b
                ):
                    k += 1
                if k >= min_match_len:
                    new_matches.append((i, i + k, j, j + k))
                    if k > max_match:
                        max_match = k
        for m in new_matches:
            for x in range(m[0], m[1]):
                marked_a.add(x)
            for y in range(m[2], m[3]):
                marked_b.add(y)
            matches.append(m)
    return _merge_nearby_spans(matches, gap=10)
