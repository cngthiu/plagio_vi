# ================================
# file: webui/app.py
# ================================
from __future__ import annotations
import io
import csv
import json
import shutil
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any

import yaml
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from src.service.runner import run_compare

app = FastAPI(title="plagio_vi WebUI", version="2.2.0")
ASSETS_DIR = Path(__file__).parent / "assets"
TEMPLATES_DIR = Path(__file__).parent / "templates"
app.mount("/assets", StaticFiles(directory=str(ASSETS_DIR)), name="assets")


# ---------- utils ----------
def _read_json_file(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"invalid json: {e}")

def _read_yaml_file(path: Path) -> Dict[str, Any]:
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"invalid yaml: {e}")

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
        return HTMLResponse("<h1>plagio_vi WebUI missing assets</h1>", status_code=200)
    return HTMLResponse(index_path.read_text(encoding="utf-8"))

@app.get("/api/report")
def get_report(path: str):
    p = Path(path)
    if not p.exists():
        raise HTTPException(status_code=404, detail=f"report not found: {p}")
    if p.suffix.lower() != ".json":
        raise HTTPException(status_code=400, detail="path must be a .json file")
    return JSONResponse(content=_read_json_file(p), media_type="application/json")

@app.get("/api/settings")
def get_settings(config: str = "configs/default.yaml", hardware: str = "configs/hardware.gpu1650.yaml"):
    cfg = _read_yaml_file(Path(config))
    ui = {
        "config_path": config,
        "hardware_path": hardware,
        "weights": {"w_cross": 0.55, "w_bi": 0.25, "w_lex": 0.10, "w_bm25": 0.10},
        "thresholds": {"very_high": 0.82, "high": 0.72, "medium": 0.60},
        "penalties": {
            "lex_boilerplate_lambda": float(cfg.get("penalties", {}).get("lex_boilerplate_lambda", 0.5)),
            "min_span_chars": int(cfg.get("penalties", {}).get("min_span_chars", 24)),
            "small_span_chars": int(cfg.get("penalties", {}).get("small_span_chars", 12)),
            "min_small_spans": int(cfg.get("penalties", {}).get("min_small_spans", 2)),
        },
        "retrieval": {
            "bm25_topk": int(cfg.get("bm25", {}).get("topk", 30)),
            "ann_topk_recall": int(cfg.get("ann", {}).get("topk_recall", 50)),
            "rerank_topk": int(cfg.get("ann", {}).get("topk_rerank", 8)),
            "simhash_topk": int(cfg.get("simhash", {}).get("topk", 40)),
        },
        "chunking": {
            "size_tokens": int(cfg.get("chunking", {}).get("size_tokens", 160)),
            "overlap": int(cfg.get("chunking", {}).get("overlap", 40)),
            "max_seq_len_bi": int(cfg.get("chunking", {}).get("max_seq_len_bi", 256)),
            "max_seq_len_cross": int(cfg.get("chunking", {}).get("max_seq_len_cross", 384)),
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
        raise HTTPException(status_code=400, detail="Both files must be .docx")

    tmpdir = Path(tempfile.mkdtemp(prefix="plagio_ui_"))
    try:
        pa = tmpdir / f"A_{a.filename}"
        pb = tmpdir / f"B_{b.filename}"
        with open(pa, "wb") as fa: shutil.copyfileobj(a.file, fa)
        with open(pb, "wb") as fb: shutil.copyfileobj(b.file, fb)

        base_cfg = _read_yaml_file(Path(config))
        override_cfg: Dict[str, Any] = {}
        if settings:
            try:
                s = json.loads(settings)
            except Exception:
                s = {}
            # map UI → YAML
            if "penalties" in s:
                override_cfg.setdefault("penalties", {}).update(s["penalties"])
            if "retrieval" in s:
                r = s["retrieval"]
                m = override_cfg
                if "bm25_topk" in r: m.setdefault("bm25", {})["topk"] = r["bm25_topk"]
                if "ann_topk_recall" in r: m.setdefault("ann", {})["topk_recall"] = r["ann_topk_recall"]
                if "rerank_topk" in r: m.setdefault("ann", {})["topk_rerank"] = r["rerank_topk"]
                if "simhash_topk" in r: m.setdefault("simhash", {})["topk"] = r["simhash_topk"]
            if "chunking" in s:
                oc = override_cfg.setdefault("chunking", {})
                for k in ["size_tokens", "overlap", "max_seq_len_bi", "max_seq_len_cross"]:
                    if k in s["chunking"]: oc[k] = s["chunking"][k]
            if "weights" in s:
                override_cfg.setdefault("ranking_weights", {}).update(s["weights"])
            if "thresholds" in s:
                override_cfg.setdefault("decision_thresholds", {}).update(s["thresholds"])

        merged_cfg = _deep_update(base_cfg, override_cfg)
        tmp_cfg = tmpdir / "override.yaml"
        tmp_cfg.write_text(yaml.safe_dump(merged_cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")

        out_dir = Path(out); out_dir.mkdir(parents=True, exist_ok=True)
        res = run_compare(str(tmp_cfg), hardware, str(pa), str(pb), str(out_dir))
        return {
            "ok": True,
            "out_dir": str(out_dir),
            "report_json": str(out_dir / "report.json"),
            "report_html": str(out_dir / "report.html"),
            "summary": res.get("summary", {}),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"compare failed: {e}")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

@app.get("/api/export_csv")
def export_csv(report_path: str):
    p = Path(report_path)
    if not p.exists():
        raise HTTPException(status_code=404, detail="report not found")
    data = _read_json_file(p)
    pairs = data.get("pairs") or []
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["score","level","a_chunk_id","b_chunk_id","cross","bi","lex","bm25_raw","a_text","b_text"])
    for r in pairs:
        a_text = (r.get("a") or {}).get("text","")
        b_text = (r.get("b") or {}).get("text","")
        w.writerow([
            r.get("score",""), r.get("level",""), r.get("a_chunk_id",""), r.get("b_chunk_id",""),
            (r.get("scores") or {}).get("cross",""),
            (r.get("scores") or {}).get("bi",""),
            (r.get("scores") or {}).get("lex",""),
            (r.get("scores") or {}).get("bm25_raw",""),
            a_text.replace("\n"," ").strip(),
            b_text.replace("\n"," ").strip(),
        ])
    buf.seek(0)
    return StreamingResponse(iter([buf.getvalue()]), media_type="text/csv",
                             headers={"Content-Disposition": "attachment; filename=report.csv"})

@app.get("/api/download_report_html")
def download_report_html(path: str):
    p = Path(path)
    if not p.exists():
        raise HTTPException(status_code=404, detail="report.html not found")
    return FileResponse(str(p), media_type="text/html", filename=p.name)
