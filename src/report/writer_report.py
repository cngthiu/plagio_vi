# file: src/report/writer_report.py
from __future__ import annotations
from typing import List, Tuple

def highlight(orig: str, spans: List[Tuple[int,int]], ctx: int = 40) -> List[dict]:
    """Trả danh sách snippet có context 2 bên để xem trong UI."""
    out = []
    for s,e in spans:
        s0 = max(0, s - ctx); e0 = min(len(orig), e + ctx)
        out.append({
            "before": orig[s0:s],
            "hit": orig[s:e],
            "after": orig[e:e0],
            "offset": [s, e]
        })
    return out
