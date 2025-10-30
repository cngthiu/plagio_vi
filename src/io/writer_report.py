# =========================
# file: src/io/writer_report.py
# =========================
from pathlib import Path
import json
from typing import Dict, Any, List, Tuple

from src.io.docx_reader import DocumentStruct
from src.report.highlight import render_marked, spans_from_pairs

def write_json_report(path: Path, docA: DocumentStruct, docB: DocumentStruct,
                      results: List[Dict[str, Any]], stats: Dict[str, Any], cfg, hw):
    obj = {
        "summary": stats["summary"],
        "pairs": results,
        "meta": {"cfg": cfg, "hw": hw}
    }
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

def _html_escape(s: str) -> str:
    import html
    return html.escape(s, quote=False)

def _guess_cells_for_matches(doc: DocumentStruct, matched_texts: List[str]) -> List[Tuple[int,int,int]]:
    """
    Cố gắng gán (table_id,row,col) nếu chuỗi match nằm trong nội dung cell.
    Heuristic: text con nằm trong cell.text (lowered).
    """
    if not doc.mapping_cells:
        return []
    out = []
    lowers = [(i, (c.table_id, c.row, c.col), c.text.lower()) for i, c in enumerate(doc.mapping_cells)]
    for mt in matched_texts:
        q = mt.strip().lower()
        if not q:
            continue
        for _i, triple, cell_text in lowers:
            if q and q in cell_text:
                out.append(triple)
                break
    return sorted(set(out))

def _extract_matched_substrings(text: str, spans: List[Tuple[int,int]]) -> List[str]:
    subs = []
    for s, e in spans:
        s = max(0, min(len(text), s))
        e = max(0, min(len(text), e))
        if e > s:
            subs.append(text[s:e])
    return subs

def write_html_report(path: Path, docA: DocumentStruct, docB: DocumentStruct,
                      results: List[Dict[str, Any]], stats: Dict[str, Any], cfg, hw):
    head = """<meta charset="utf-8"><style>
    body{font-family:system-ui,Arial;margin:16px;}
    .score{font-weight:600}
    mark{background:yellow}
    .pair{border:1px solid #ddd;margin:12px;padding:12px;border-radius:10px}
    .two{display:grid;grid-template-columns:1fr 1fr;gap:12px}
    .meta{color:#666;font-size:12px;margin-top:6px}
    .cells{font-size:12px;color:#333;margin-top:6px}
    .badge{display:inline-block;background:#eef;padding:2px 6px;border-radius:8px;margin:2px}
    </style>"""
    html = [f"<html><head>{head}</head><body>"]
    html.append(f"<h2>Summary</h2><pre>{_html_escape(json.dumps(stats['summary'], ensure_ascii=False, indent=2))}</pre>")
    html.append("<h2>Top matches</h2>")
    for r in sorted(results, key=lambda x: x["score"], reverse=True)[:200]:
        a_text = r["a_text"]
        b_text = r["b_text"]
        spans_pairs = r["spans"]["pairs"]  # list of [a0,a1,b0,b1]
        spans_map = spans_from_pairs([(p["a"][0],p["a"][1],p["b"][0],p["b"][1]) for p in spans_pairs])
        html.append('<div class="pair">')
        html.append(f'<div class="score">Score: {r["score"]:.3f} — cross={r["scores"]["cross"]:.3f} bi={r["scores"]["bi"]:.3f} lex={r["scores"]["lex"]:.3f}</div>')
        html.append('<div class="two">')
        html.append(f'<div><h4>A[{r["a_chunk_id"]}]</h4>{render_marked(a_text, spans_map["a"])}</div>')
        html.append(f'<div><h4>B[{r["b_chunk_id"]}]</h4>{render_marked(b_text, spans_map["b"])}</div>')
        html.append("</div>")  # .two
        # Cells guess
        matched_in_b = _extract_matched_substrings(b_text, spans_map["b"])
        cell_hits = _guess_cells_for_matches(docB, matched_in_b)
        if cell_hits:
            tags = " ".join([f'<span class="badge">table {t}, r{r0}, c{c0}</span>' for (t, r0, c0) in cell_hits])
            html.append(f'<div class="cells">Cells (B): {tags}</div>')
        html.append(f'<div class="meta">{_html_escape(json.dumps(r["spans"], ensure_ascii=False))}</div>')
        html.append("</div>")
    html.append("</body></html>")
    path.write_text("\n".join(html), encoding="utf-8")
