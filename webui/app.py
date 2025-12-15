# file: webui/app.py
# ================================
# Cleanup & Optimized by Senior Tech Lead
# ================================
from __future__ import annotations
import io
import csv
import json
import shutil
import tempfile
import logging
from pathlib import Path
from typing import Optional, Dict, Any

import yaml
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from src.service.runner import run_compare

# Setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("webui")

app = FastAPI(title="plagio_vi WebUI", version="2.3.0")
ASSETS_DIR = Path(__file__).parent / "assets"
# Removed unused TEMPLATES_DIR

app.mount("/assets", StaticFiles(directory=str(ASSETS_DIR)), name="assets")

# ---------- utils ----------
def _read_json_file(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.error(f"Read JSON failed: {e}")
        raise HTTPException(status_code=500, detail="Invalid JSON file")

def _read_yaml_file(path: Path) -> Dict[str, Any]:
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.error(f"Read YAML failed: {e}")
        raise HTTPException(status_code=500, detail="Invalid YAML file")

def _deep_update(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = v
    return out

# ---------- routes ----------
@app.get("/", response_class=HTMLResponse)
def home():
    index_path = ASSETS_DIR / "index.html"
    if not index_path.exists():
        return "<h1>Missing assets/index.html</h1>"
    return HTMLResponse(index_path.read_text(encoding="utf-8"))

@app.get("/api/report")
def get_report(path: str):
    p = Path(path)
    if not p.exists() or p.suffix.lower() != ".json":
        raise HTTPException(status_code=404, detail="Report not found")
    return JSONResponse(content=_read_json_file(p))

@app.get("/api/settings")
def get_settings(config: str = "configs/default.yaml", hardware: str = "configs/hardware.gpu1650.yaml"):
    # Load defaults securely
    cfg = _read_yaml_file(Path(config))
    # Extract params for UI mapping
    # (Giữ nguyên logic mapping cũ vì nó đang hoạt động tốt với Frontend mới)
    c_chunk = cfg.get("chunking", {})
    c_bm25  = cfg.get("bm25", {})
    c_ann   = cfg.get("ann", {})
    c_pen   = cfg.get("penalties", {})
    
    ui = {
        "weights": {"w_cross": 0.55, "w_bi": 0.25, "w_lex": 0.10, "w_bm25": 0.10}, # Default fallback
        "thresholds": {"very_high": 0.82, "high": 0.72, "medium": 0.60},
        "penalties": {
            "lex_boilerplate_lambda": float(c_pen.get("lex_boilerplate_lambda", 0.5)),
            "min_span_chars": int(c_pen.get("min_span_chars", 24)),
            "small_span_chars": int(c_pen.get("small_span_chars", 12)),
            "min_small_spans": int(c_pen.get("min_small_spans", 2)),
        },
        "retrieval": {
            "bm25_topk": int(c_bm25.get("topk", 30)),
            "ann_topk_recall": int(c_ann.get("topk_recall", 50)),
            "rerank_topk": int(c_ann.get("topk_rerank", 8)),
            "simhash_topk": int(cfg.get("simhash", {}).get("topk", 40)),
        },
        "chunking": {
            "size_tokens": int(c_chunk.get("size_tokens", 160)),
            "overlap": int(c_chunk.get("overlap", 40)),
            "max_seq_len_bi": int(c_chunk.get("max_seq_len_bi", 256)),
            "max_seq_len_cross": int(c_chunk.get("max_seq_len_cross", 384)),
        },
    }
    return ui

@app.post("/api/compare")
async def compare(
    a: UploadFile = File(...),
    b: UploadFile = File(...),
    out: str = Form("outputs/webui_run"),
    config: str = Form("configs/default.yaml"),
    hardware: str = Form("configs/hardware.gpu1650.yaml"),
    settings: Optional[str] = Form(None),
):
    if not a.filename.lower().endswith(".docx") or not b.filename.lower().endswith(".docx"):
        raise HTTPException(status_code=400, detail="Only .docx files are supported")

    # Use secure temp dir handling
    tmpdir = Path(tempfile.mkdtemp(prefix="plagio_ui_"))
    try:
        pa = tmpdir / f"A_{a.filename}"
        pb = tmpdir / f"B_{b.filename}"
        
        # Save uploaded files
        with open(pa, "wb") as fa: shutil.copyfileobj(a.file, fa)
        with open(pb, "wb") as fb: shutil.copyfileobj(b.file, fb)

        # Config Override Logic
        base_cfg = _read_yaml_file(Path(config))
        override_cfg = {}
        if settings:
            try:
                s = json.loads(settings)
                # Map UI structure back to YAML structure
                if "penalties" in s: override_cfg["penalties"] = s["penalties"]
                if "weights" in s: override_cfg["ranking_weights"] = s["weights"]
                if "thresholds" in s: override_cfg["decision_thresholds"] = s["thresholds"]
                
                # Complex mappings
                if "retrieval" in s:
                    r = s["retrieval"]
                    if "bm25_topk" in r: override_cfg.setdefault("bm25", {})["topk"] = r["bm25_topk"]
                    if "ann_topk_recall" in r: override_cfg.setdefault("ann", {})["topk_recall"] = r["ann_topk_recall"]
                    if "rerank_topk" in r: override_cfg.setdefault("ann", {})["topk_rerank"] = r["rerank_topk"]
                    if "simhash_topk" in r: override_cfg.setdefault("simhash", {})["topk"] = r["simhash_topk"]
                
                if "chunking" in s:
                    override_cfg["chunking"] = s["chunking"]

            except Exception as e:
                logger.warning(f"Failed to parse settings JSON: {e}")

        merged_cfg = _deep_update(base_cfg, override_cfg)
        tmp_cfg = tmpdir / "override.yaml"
        tmp_cfg.write_text(yaml.safe_dump(merged_cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")

        # Run Core Logic
        out_dir = Path(out)
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # Blocking call (for simplicity in this architecture)
        res = run_compare(str(tmp_cfg), hardware, str(pa), str(pb), str(out_dir))
        
        return {
            "ok": True,
            "out_dir": str(out_dir),
            "report_json": str(out_dir / "report.json"),
            "report_html": str(out_dir / "report.html"), # Now points to the beautified HTML
            "summary": res.get("summary", {}),
        }
        
    except Exception as e:
        logger.exception("Compare Error")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

@app.get("/api/export_csv")
def export_csv(report_path: str):
    p = Path(report_path)
    if not p.exists():
        raise HTTPException(status_code=404, detail="Report not found")
        
    data = _read_json_file(p)
    pairs = data.get("pairs") or []
    
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["score", "level", "a_chunk_id", "b_chunk_id", "a_text", "b_text"])
    
    for r in pairs:
        w.writerow([
            f"{r.get('score', 0):.4f}",
            r.get("level", ""),
            r.get("a_chunk_id", ""),
            r.get("b_chunk_id", ""),
            (r.get("a") or {}).get("text", "").replace("\n", " ")[:500],
            (r.get("b") or {}).get("text", "").replace("\n", " ")[:500]
        ])
        
    buf.seek(0)
    return StreamingResponse(
        iter([buf.getvalue()]), 
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=report_export.csv"}
    )

@app.get("/api/download_report_html")
def download_report_html(path: str):
    p = Path(path)
    if not p.exists():
        raise HTTPException(status_code=404, detail="Report HTML not found")
    return FileResponse(str(p), media_type="text/html", filename=f"PlagioVi_Report_{p.name}")