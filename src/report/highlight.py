# file: src/report/highlight.py
# =========================
# Updated logic for safe HTML rendering
# =========================
from typing import List, Tuple, Dict, Any
from html import escape

def _merge_spans(spans: List[Tuple[int,int]], gap: int = 5) -> List[Tuple[int,int]]:
    if not spans:
        return []
    # Sort by start position
    spans = sorted(spans, key=lambda x: x[0])
    out = [spans[0]]
    for s, e in spans[1:]:
        last_s, last_e = out[-1]
        # Nếu đoạn hiện tại bắt đầu trước (hoặc sát) đoạn cũ kết thúc + gap
        if s <= last_e + gap:
            out[-1] = (last_s, max(last_e, e))
        else:
            out.append((s, e))
    return out

def render_marked(text: str, spans: List[Tuple[int,int]]) -> str:
    """
    Tạo HTML chuỗi với <mark> bao quanh.
    Sử dụng html.escape để tránh XSS.
    """
    if not text:
        return ""
    if not spans:
        return f"<pre>{escape(text)}</pre>"
    
    spans = _merge_spans(spans)
    html_parts = ["<pre>"] # Dùng pre để giữ format văn bản gốc
    cur = 0
    
    for s, e in spans:
        s = max(0, min(len(text), s))
        e = max(0, min(len(text), e))
        
        # Phần text không trùng (trước mark)
        if cur < s:
            html_parts.append(escape(text[cur:s]))
            
        # Phần trùng (trong mark)
        html_parts.append("<mark>")
        html_parts.append(escape(text[s:e]))
        html_parts.append("</mark>")
        
        cur = e
        
    # Phần còn lại
    if cur < len(text):
        html_parts.append(escape(text[cur:]))
        
    html_parts.append("</pre>")
    return "".join(html_parts)

def make_snippets(orig: str, spans: List[Tuple[int,int]], ctx: int = 40) -> List[Dict[str, Any]]:
    # Giữ nguyên logic cũ, chỉ reformat cho gọn
    snips = []
    length = len(orig)
    spans = _merge_spans(spans) # Merge trước khi cắt snippet cho đỡ trùng
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