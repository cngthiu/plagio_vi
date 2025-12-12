# =========================
# file: src/report/highlight.py
# =========================
from typing import List, Tuple, Dict, Any

def _merge_spans(spans: List[Tuple[int,int]], gap: int = 5) -> List[Tuple[int,int]]:
    if not spans:
        return []
    spans = sorted(spans)
    out = [spans[0]]
    for s, e in spans[1:]:
        S, E = out[-1]
        if s <= E + gap:
            out[-1] = (S, max(E, e))
        else:
            out.append((s, e))
    return out

def render_marked(text: str, spans: List[Tuple[int,int]]) -> str:
    """Tạo HTML chuỗi với <mark> bao quanh các đoạn trùng lặp."""
    if not spans:
        from html import escape
        return f"<pre>{escape(text)}</pre>"
    from html import escape
    spans = _merge_spans(spans)
    html = ["<pre>"]
    cur = 0
    for s, e in spans:
        s = max(0, min(len(text), s))
        e = max(0, min(len(text), e))
        if cur < s:
            html.append(escape(text[cur:s]))
        html.append("<mark>")
        html.append(escape(text[s:e]))
        html.append("</mark>")
        cur = e
    if cur < len(text):
        html.append(escape(text[cur:]))
    html.append("</pre>")
    return "".join(html)

def spans_from_pairs(pairs: List[Tuple[int,int,int,int]]) -> Dict[str, List[Tuple[int,int]]]:
    """Chuyển (a0,a1,b0,b1) -> hai danh sách spans riêng cho A và B."""
    a_sp = [(a0, a1) for (a0,a1,_,_) in pairs]
    b_sp = [(b0, b1) for (_,_,b0,b1) in pairs]
    return {"a": a_sp, "b": b_sp}

def make_snippets(orig: str, spans: List[Tuple[int,int]], ctx: int = 40) -> List[Dict[str, Any]]:
    """Tạo danh sách snippet (trích đoạn) có ngữ cảnh 2 bên."""
    snips = []
    length = len(orig)
    for s, e in spans:
        s0 = max(0, s - ctx)
        e0 = min(length, e + ctx)
        snips.append({
            "before": orig[s0:s],
            "hit": orig[s:e],
            "after": orig[e:e0],
            "offset": [s, e]
        })
    return snips