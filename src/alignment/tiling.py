# =========================
# file: src/alignment/tiling.py
# =========================
from typing import List, Tuple

def greedy_string_tiling(a: str, b: str, min_match_len: int = 20) -> List[Tuple[int, int, int, int]]:
    # Returns list of (a_start, a_end, b_start, b_end)
    A, B = a, b
    matches = []
    marked_a = set(); marked_b = set()
    max_match = min_match_len
    while max_match >= min_match_len:
        max_match = 0
        new_matches = []
        for i in range(len(A)):
            for j in range(len(B)):
                k = 0
                while i+k < len(A) and j+k < len(B) and A[i+k] == B[j+k] and (i+k) not in marked_a and (j+k) not in marked_b:
                    k += 1
                if k >= min_match_len:
                    new_matches.append((i, i+k, j, j+k))
                    if k > max_match:
                        max_match = k
        for m in new_matches:
            for x in range(m[0], m[1]): marked_a.add(x)
            for y in range(m[2], m[3]): marked_b.add(y)
            matches.append(m)
    # merge nearby
    merged = []
    for s in sorted(matches):
        if not merged: merged.append(s); continue
        a0,a1,b0,b1 = s
        A0,A1,B0,B1 = merged[-1]
        if a0 <= A1+10 and b0 <= B1+10:
            merged[-1] = (A0, max(A1,a1), B0, max(B1,b1))
        else:
            merged.append(s)
    return merged
