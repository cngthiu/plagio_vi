# file: src/alignment/sw_align.py
from typing import List, Tuple
from difflib import SequenceMatcher

def _merge_blocks(blocks: List[Tuple[int,int,int,int]], gap: int = 5) -> List[Tuple[int,int,int,int]]:
    if not blocks:
        return []
    blocks = sorted(blocks)
    merged = [blocks[0]]
    for a0,a1,b0,b1 in blocks[1:]:
        A0,A1,B0,B1 = merged[-1]
        if a0 <= A1 + gap and b0 <= B1 + gap:
            merged[-1] = (A0, max(A1, a1), B0, max(B1, b1))
        else:
            merged.append((a0,a1,b0,b1))
    return merged

def local_align_spans(a: str, b: str, min_len: int = 20) -> List[Tuple[int,int,int,int]]:
    """
    Fallback local alignment dùng difflib:
    - Lấy matching blocks liên tiếp giữa a và b.
    - Lọc theo độ dài tối thiểu.
    - Gộp block gần nhau thành span lớn.
    Trả về: [(a_start,a_end,b_start,b_end), ...]
    """
    sm = SequenceMatcher(None, a, b, autojunk=False)
    blocks = []
    for m in sm.get_matching_blocks():
        if m.size >= min_len:
            blocks.append((m.a, m.a + m.size, m.b, m.b + m.size))
    return _merge_blocks(blocks, gap=5)
