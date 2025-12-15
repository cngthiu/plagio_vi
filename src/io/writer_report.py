# file: src/io/writer_report.py
# =========================
# Refactored by Senior Tech Lead
# Focus: Standalone, Modern UI, Semantic Coloring
# =========================
from pathlib import Path
import json
import html
from typing import Dict, Any, List, Tuple
from src.io.docx_reader import DocumentStruct
from src.report.highlight import render_marked

def write_json_report(path: Path, docA: DocumentStruct, docB: DocumentStruct,
                      results: List[Dict[str, Any]], stats: Dict[str, Any], cfg, hw):
    obj = {
        "summary": stats["summary"],
        "pairs": results,
        "meta": {"cfg": cfg, "hw": hw}
    }
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

def _esc(s: str) -> str:
    return html.escape(s, quote=False)

def _get_css() -> str:
    """Design System embedded directly into HTML for portability."""
    return """
    :root {
      --bg: #f8fafc; --card: #ffffff; --text: #0f172a; --muted: #64748b; --border: #e2e8f0;
      --primary: #2563eb; --danger: #ef4444; --warning: #f59e0b; --success: #10b981;
      --hl-very-high: #fecaca; --hl-high: #fed7aa; --hl-medium: #fef08a; --hl-low: #e2e8f0;
    }
    body { font-family: 'Inter', system-ui, -apple-system, sans-serif; background: var(--bg); color: var(--text); margin: 0; padding: 20px; line-height: 1.5; }
    .container { max-width: 1200px; margin: 0 auto; }
    
    /* Header */
    header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 24px; }
    h1 { font-size: 24px; font-weight: 700; color: var(--primary); margin: 0; }
    .meta { font-size: 14px; color: var(--muted); }

    /* Cards */
    .card { background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 20px; margin-bottom: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }
    
    /* Metrics */
    .metrics-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; }
    .metric { text-align: center; padding: 16px; background: #f1f5f9; border-radius: 8px; }
    .metric-val { font-size: 24px; font-weight: 700; display: block; }
    .metric-label { font-size: 12px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.5px; }
    
    /* List */
    .pair { border: 1px solid var(--border); border-radius: 8px; margin-bottom: 16px; overflow: hidden; background: #fff; transition: box-shadow 0.2s; }
    .pair:hover { box-shadow: 0 4px 12px rgba(0,0,0,0.08); }
    
    .pair-header { 
        padding: 12px 16px; background: #f8fafc; border-bottom: 1px solid var(--border);
        display: flex; justify-content: space-between; align-items: center; 
    }
    .pair-score { font-weight: 700; font-size: 16px; }
    .badge { padding: 4px 8px; border-radius: 6px; font-size: 12px; font-weight: 600; text-transform: uppercase; }
    
    /* Semantic Colors */
    .lvl-very_high { color: var(--danger); background: #fef2f2; border: 1px solid #fecaca; }
    .lvl-high { color: var(--warning); background: #fffbeb; border: 1px solid #fde68a; }
    .lvl-medium { color: #b45309; background: #fff7ed; border: 1px solid #fed7aa; }
    .lvl-low { color: var(--muted); background: #f1f5f9; }

    /* Content Diff */
    .pair-body { display: grid; grid-template-columns: 1fr 1fr; gap: 1px; background: var(--border); }
    .col { background: #fff; padding: 16px; font-size: 14px; font-family: 'Consolas', monospace; white-space: pre-wrap; }
    .col-head { font-weight: 600; margin-bottom: 8px; color: var(--muted); font-size: 12px; }

    /* Marks */
    mark { padding: 2px 0; border-radius: 2px; }
    .ctx-very_high mark { background: var(--hl-very-high); border-bottom: 2px solid var(--danger); }
    .ctx-high mark { background: var(--hl-high); border-bottom: 2px solid var(--warning); }
    .ctx-medium mark { background: var(--hl-medium); }
    .ctx-low mark { background: var(--hl-low); color: var(--muted); }
    
    /* Utility */
    .tech-info { font-size: 11px; color: var(--muted); margin-top: 6px; font-family: monospace; }
    .expand-btn { cursor: pointer; color: var(--primary); text-decoration: underline; font-size: 13px; }
    """

def write_html_report(path: Path, docA: DocumentStruct, docB: DocumentStruct,
                      results: List[Dict[str, Any]], stats: Dict[str, Any], cfg, hw):
    
    summary = stats["summary"]
    # Sort results by score desc
    sorted_results = sorted(results, key=lambda x: x["score"], reverse=True)
    
    html_parts = []
    
    # 1. Head
    html_parts.append(f"""
    <!doctype html>
    <html lang="vi">
    <head>
      <meta charset="utf-8">
      <meta name="viewport" content="width=device-width, initial-scale=1">
      <title>PlagioVi Report</title>
      <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
      <style>{_get_css()}</style>
    </head>
    <body>
    <div class="container">
      <header>
        <div>
           <h1>PlagioVi Analysis Report</h1>
           <div class="meta">Generated automatically by PlagioVi Engine</div>
        </div>
        <div>
           <span class="badge" style="background:#e0f2fe; color:#0369a1;">{len(sorted_results)} Matches Found</span>
        </div>
      </header>
    """)

    # 2. Overview Metrics
    html_parts.append(f"""
      <section class="card">
        <div class="metrics-grid">
           <div class="metric">
             <span class="metric-val" style="color:var(--danger)">{summary['levels'].get('very_high', 0)}</span>
             <span class="metric-label">Very High</span>
           </div>
           <div class="metric">
             <span class="metric-val" style="color:var(--warning)">{summary['levels'].get('high', 0)}</span>
             <span class="metric-label">High</span>
           </div>
           <div class="metric">
             <span class="metric-val">{summary['levels'].get('medium', 0)}</span>
             <span class="metric-label">Medium</span>
           </div>
           <div class="metric">
             <span class="metric-val">{summary['chunks_A']} / {summary['chunks_B']}</span>
             <span class="metric-label">Chunks Processed</span>
           </div>
        </div>
      </section>
    """)

    # 3. Detailed List
    html_parts.append('<section class="details">')
    
    for i, r in enumerate(sorted_results):
        score_pct = r["score"] * 100
        level = r.get("level", "low")
        
        # Determine texts
        a_text = r.get("a", {}).get("text", "")
        b_text = r.get("b", {}).get("text", "")
        spans_a = r.get("a", {}).get("spans", [])
        spans_b = r.get("b", {}).get("spans", [])
        
        # Scores detail
        sc = r.get("scores", {})
        tech_detail = f"Cross: {sc.get('cross',0):.3f} | Bi: {sc.get('bi',0):.3f} | Lex: {sc.get('lex',0):.3f} | BM25: {sc.get('bm25_raw',0):.1f}"

        html_parts.append(f"""
        <div class="pair ctx-{level}">
          <div class="pair-header">
             <div style="display:flex; align-items:center; gap:12px">
                <span class="badge lvl-{level}">{level.replace('_', ' ').upper()}</span>
                <span class="pair-score">Match #{i+1}: {score_pct:.1f}%</span>
             </div>
             <div class="tech-info">{tech_detail}</div>
          </div>
          <div class="pair-body">
             <div class="col">
               <div class="col-head">DOCUMENT A (Chunk {r.get('a_chunk_id')})</div>
               {render_marked(a_text, spans_a)}
             </div>
             <div class="col">
               <div class="col-head">DOCUMENT B (Chunk {r.get('b_chunk_id')})</div>
               {render_marked(b_text, spans_b)}
             </div>
          </div>
        </div>
        """)
        
    html_parts.append('</section>') # end details
    
    # Footer
    html_parts.append("""
      <footer style="text-align:center; margin-top:40px; color:var(--muted); font-size:12px;">
        &copy; 2025 PlagioVi. High-Performance Plagiarism Detection.
      </footer>
    </div></body></html>
    """)

    path.write_text("\n".join(html_parts), encoding="utf-8")