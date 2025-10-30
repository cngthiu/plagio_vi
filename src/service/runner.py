# file: src/service/runner.py
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Tuple
import re
import json
import numpy as np
import yaml

from src.io.docx_reader import read_docx_with_tables
from src.candidate.bm25_index import BM25Index
from src.candidate.simhash import SimHashIndex, simhash
from src.candidate.ann_index import ANNIndex
from src.models.embedder import BiEncoder
from src.models.cross_encoder import CrossEncoder
from src.ranking.features import (
    cosine_sim,
    jaccard_char_ngrams,
    hamming_similarity,
    boilerplate_penalty,
)
from src.alignment.tiling import greedy_string_tiling
from src.alignment.sw_align import local_align_spans
from src.utils.timing import Timer
from src.utils.hardware import set_num_threads, select_device_config
from src.utils.logging import JsonLogger
from src.utils.cache import DiskCache


# ---------- helpers ----------
TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)

def _html(s: str) -> str:
    return s.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")

def _tokens_with_offsets(text: str) -> List[Tuple[str, int, int]]:
    out = []
    for m in TOKEN_RE.finditer(text):
        out.append((m.group(0), m.start(), m.end()))
    return out

def _chunk_by_tokens(
    text_display: str,
    lowercase: bool,
    size_tokens: int,
    overlap: int
) -> Tuple[List[str], List[str], List[Tuple[int,int]]]:
    """
    Trả về:
      - chunks_align: dùng cho BM25/SimHash/align (lowercase)
      - chunks_disp : hiển thị (gốc)
      - spans_char  : (start,end) theo chuỗi gốc/display
    """
    disp = text_display
    align = disp.lower() if lowercase else disp
    toks = _tokens_with_offsets(align)
    if not toks:
        return [], [], []
    step = max(1, size_tokens - overlap)
    chunks_align, chunks_disp, spans_char = [], [], []
    for i in range(0, len(toks), step):
        sub = toks[i:i+size_tokens]
        if not sub:
            break
        s = sub[0][1]; e = sub[-1][2]
        spans_char.append((s, e))
        chunks_align.append(align[s:e])
        chunks_disp.append(disp[s:e])
        if i + size_tokens >= len(toks):
            break
    return chunks_align, chunks_disp, spans_char

def _minmax(a: np.ndarray) -> np.ndarray:
    if a.size == 0:
        return a
    mn = float(np.min(a)); mx = float(np.max(a))
    if mx - mn < 1e-9:
        return np.zeros_like(a, dtype=np.float32)
    return ((a - mn) / (mx - mn)).astype(np.float32)

def _nms_spans(pairs: List[Tuple[int,int,int,int]], iou_th: float = 0.6) -> List[Tuple[int,int,int,int]]:
    if not pairs:
        return []
    def iou_pair(a, b):
        a0,a1,b0,b1 = a; c0,c1,d0,d1 = b
        ia = max(0, min(a1, c1) - max(a0, c0))
        ib = max(0, min(b1, d1) - max(b0, d0))
        ua = (a1-a0) + (c1-c0) - ia
        ub = (b1-b0) + (d1-d0) - ib
        i = min(ia, ib)
        u = (ua + ub) / 2 if (ua>0 and ub>0) else max(ua, ub)
        return (i / u) if u else 0.0
    out: List[Tuple[int,int,int,int]] = []
    for sp in sorted(pairs, key=lambda x: (x[1]-x[0])+(x[3]-x[2]), reverse=True):
        if all(iou_pair(sp, x) < iou_th for x in out):
            out.append(sp)
    return out

def _snippets(orig: str, spans: List[Tuple[int,int]], ctx: int = 40) -> List[Dict[str, Any]]:
    snips = []
    for s,e in spans:
        s0 = max(0, s-ctx); e0 = min(len(orig), e+ctx)
        snips.append({"before": orig[s0:s], "hit": orig[s:e], "after": orig[e:e0], "offset": [s,e]})
    return snips

def _fuse_scores(
    cross: List[float],
    bi: List[float],
    lex: List[float],
    bm25: List[float],
    w=(0.55,0.25,0.10,0.10)
) -> List[float]:
    cross = np.asarray(cross, np.float32)
    bi    = np.asarray(bi,    np.float32)
    lex   = np.asarray(lex,   np.float32)
    bm25n = _minmax(np.asarray(bm25, np.float32))
    S = w[0]*cross + w[1]*bi + w[2]*lex + w[3]*bm25n
    return np.clip(S, 0.0, 1.0).tolist()

def _level(score: float) -> str:
    if score >= 0.82: return "very_high"
    if score >= 0.72: return "high"
    if score >= 0.60: return "medium"
    return "low"

def _load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------- main ----------
def run_compare(config_path: str, hardware_path: str, path_a: str, path_b: str, out_dir: str) -> Dict[str, Any]:
    cfg = _load_yaml(config_path)
    hw = _load_yaml(hardware_path)
    faiss_cfg = _load_yaml(hw["index"]["faiss"]["cfg_path"])

    outp = Path(out_dir); outp.mkdir(parents=True, exist_ok=True)
    set_num_threads(hw["device"].get("omp_threads", 8), hw["device"].get("mkl_threads", 8))
    device_cfg = select_device_config(hw)
    logger = JsonLogger(outp / "logs.jsonl")
    timer = Timer(logger=logger)
    cache = DiskCache(base_dir="outputs/cache")

    # --- load boilerplate & penalties from config ---
    boiler_paths = []
    db_cfg = cfg.get("domain_boilerplate", {})
    if db_cfg.get("enabled", False):
        boiler_paths = db_cfg.get("paths", []) or []
    boiler_set = set()
    for p in boiler_paths:
        try:
            for line in open(p, encoding="utf-8"):
                line = line.strip()
                if line:
                    boiler_set.add(line.lower())
        except Exception:
            pass
    pen = cfg.get("penalties", {})
    LAMBDA = float(pen.get("lex_boilerplate_lambda", 0.5))
    MIN_SPAN = int(pen.get("min_span_chars", 24))
    SMALL_SPAN = int(pen.get("small_span_chars", 12))
    MIN_SMALL = int(pen.get("min_small_spans", 2))

    # --- Read & linearize ---
    with timer.section("read_docs"):
        docA = read_docx_with_tables(path_a, keep_tables=cfg["io"]["keep_tables"])
        docB = read_docx_with_tables(path_b, keep_tables=cfg["io"]["keep_tables"])
        docA.linearize(); docB.linearize()
        docA.apply_ignores(cfg["io"]["ignore_sections"])
        docB.apply_ignores(cfg["io"]["ignore_sections"])
        docA.linearize(); docB.linearize()
        textA_disp = docA.linearized_text
        textB_disp = docB.linearized_text

    # --- Chunk ---
    with timer.section("chunking"):
        chunksA_align, chunksA_disp, spansA_char = _chunk_by_tokens(
            textA_disp, lowercase=True,
            size_tokens=cfg["chunking"]["size_tokens"], overlap=cfg["chunking"]["overlap"]
        )
        chunksB_align, chunksB_disp, spansB_char = _chunk_by_tokens(
            textB_disp, lowercase=True,
            size_tokens=cfg["chunking"]["size_tokens"], overlap=cfg["chunking"]["overlap"]
        )

    # --- BM25 / SimHash ---
    with timer.section("indexes"):
        bm25 = BM25Index(chunksB_align)
        shB  = SimHashIndex([simhash(c) for c in chunksB_align])

    # --- Models ---
    with timer.section("models_load"):
        bi = BiEncoder(
            device_cfg=device_cfg,
            max_seq_len=cfg["chunking"]["max_seq_len_bi"],
            backend=hw["inference"]["bi_encoder"]["backend"],
            quantize=hw["inference"]["bi_encoder"]["quantize"]
        )
        ce = CrossEncoder(
            device_cfg=device_cfg,
            max_seq_len=cfg["chunking"]["max_seq_len_cross"],
            backend=hw["inference"]["cross_encoder"]["backend"],
            quantize=hw["inference"]["cross_encoder"]["quantize"]
        )

    # --- Embedding B -> ANN (dim inferred) ---
    with timer.section("embed_B"):
        kB = cache.key_for_texts(chunksB_disp, getattr(bi, "model_name", "bi"))
        embB = cache.get_numpy("embeddings", kB)
        if embB is None:
            embB = bi.encode(
                chunksB_disp,
                batch_size=(hw["inference"]["bi_encoder"].get("batch_size_gpu")
                            or hw["inference"]["bi_encoder"].get("batch_size_cpu", 32))
            )
            cache.set_numpy("embeddings", kB, embB)
        ann = ANNIndex(dim=None, cfg=faiss_cfg)  # dim suy ra trong build()
        ann.build(embB)

    # --- Embedding A + SimHash A ---
    with timer.section("embed_A_and_simhash"):
        kA = cache.key_for_texts(chunksA_disp, getattr(bi, "model_name", "bi"))
        embA = cache.get_numpy("embeddings", kA)
        if embA is None:
            embA = bi.encode(
                chunksA_disp,
                batch_size=(hw["inference"]["bi_encoder"].get("batch_size_gpu")
                            or hw["inference"]["bi_encoder"].get("batch_size_cpu", 32))
            )
            cache.set_numpy("embeddings", kA, embA)
        shA_vals = [simhash(c) for c in chunksA_align]

    bm25_topk   = cfg["bm25"]["topk"]
    ann_topk    = cfg["ann"]["topk_recall"]
    sim_topk    = cfg["simhash"]["topk"]
    rerank_topk = cfg["ann"]["topk_rerank"]

    results: List[Dict[str, Any]] = []

    def _passes_span_rule(spans: List[Tuple[int,int]]) -> bool:
        if not spans:
            return False
        if any((e - s) >= MIN_SPAN for s, e in spans):
            return True
        cnt = sum(1 for s, e in spans if (e - s) >= SMALL_SPAN)
        return cnt >= MIN_SMALL

    # --- Match → rerank → align ---
    with timer.section("match_rerank_align"):
        for i, (a_align, a_disp) in enumerate(zip(chunksA_align, chunksA_disp)):
            bm_idx  = bm25.search(a_align, topk=bm25_topk)
            sim_idx = shB.topk_by_hamming(shA_vals[i], k=sim_topk)
            ann_idx = ann.search(embA[i], topk=ann_topk)
            cand = sorted(set(bm_idx + sim_idx + ann_idx))[:max(bm25_topk, ann_topk, sim_topk)]
            if not cand:
                continue

            s_bi_list, s_lex_list, s_bm_list = [], [], []
            for j in cand:
                b_align = chunksB_align[j]
                s_bi  = cosine_sim(embA[i], embB[j])
                base_jac = jaccard_char_ngrams(a_align, b_align, n=5)
                base_ham = hamming_similarity(shA_vals[i], simhash(b_align))
                base_lex = 0.5*base_jac + 0.5*base_ham
                bp = boilerplate_penalty(a_disp, chunksB_disp[j], boiler_set)
                s_lex = base_lex * (1.0 - LAMBDA * bp)
                s_bm  = bm25.score_of_cached(j)
                s_bi_list.append(s_bi); s_lex_list.append(s_lex); s_bm_list.append(s_bm)

            order_by_bi = np.argsort(-np.asarray(s_bi_list))[:rerank_topk]
            pairs = [(a_disp, chunksB_disp[cand[ix]]) for ix in order_by_bi]
            s_cross = ce.score_pairs(
                pairs,
                batch_size=(hw["inference"]["cross_encoder"].get("batch_size_gpu")
                            or hw["inference"]["cross_encoder"].get("batch_size_cpu", 4))
            )
            cross_map = {cand[order_by_bi[k]]: s_cross[k] for k in range(len(order_by_bi))}
            cross_full = [cross_map.get(j, 0.0) for j in cand]

            S = _fuse_scores(cross_full, s_bi_list, s_lex_list, s_bm_list, w=(0.55,0.25,0.10,0.10))
            keep_ord = np.argsort(-np.asarray(S))[:5]

            for idx_in_keep in keep_ord:
                j = cand[idx_in_keep]
                a_text = chunksA_disp[i]; b_text = chunksB_disp[j]
                a_align_text = chunksA_align[i]; b_align_text = chunksB_align[j]

                spans_pairs = greedy_string_tiling(a_align_text, b_align_text)
                if not spans_pairs:
                    spans_pairs = local_align_spans(a_align_text, b_align_text)
                spans_pairs = _nms_spans(spans_pairs, iou_th=0.6)

                spans_a = [(a0,a1) for (a0,a1,_,_) in spans_pairs]
                spans_b = [(b0,b1) for (_,_,b0,b1) in spans_pairs]

                # enforce span rules both sides
                if not _passes_span_rule(spans_a) or not _passes_span_rule(spans_b):
                    continue

                rec = {
                    "a_chunk_id": i,
                    "b_chunk_id": j,
                    "score": float(S[idx_in_keep]),
                    "level": _level(float(S[idx_in_keep])),
                    "scores": {
                        "cross": float(cross_full[idx_in_keep]),
                        "bi": float(s_bi_list[idx_in_keep]),
                        "lex": float(s_lex_list[idx_in_keep]),
                        "bm25_raw": float(s_bm_list[idx_in_keep]),
                    },
                    "a": {
                        "text": a_text,
                        "spans": spans_a,
                        "snippets": _snippets(a_text, spans_a, ctx=40),
                        "char_span_in_doc": [spansA_char[i][0], spansA_char[i][1]],
                    },
                    "b": {
                        "text": b_text,
                        "spans": spans_b,
                        "snippets": _snippets(b_text, spans_b, ctx=40),
                        "char_span_in_doc": [spansB_char[j][0], spansB_char[j][1]],
                    },
                }
                results.append(rec)

    # --- Summary ---
    with timer.section("aggregate"):
        levels = {"very_high":0,"high":0,"medium":0,"low":0}
        for r in results: levels[r["level"]] += 1
        summary = {"chunks_A": len(chunksA_disp), "chunks_B": len(chunksB_disp), "pairs": len(results), "levels": levels}

    # --- JSON ---
    (outp / "report.json").write_text(
        json.dumps({"summary": summary, "pairs": results}, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    # --- HTML (visual) ---
    html = []
    html.append("""<meta charset="utf-8"><style>
    body{font-family:system-ui,Arial;margin:16px;line-height:1.45}
    .grid{display:grid;grid-template-columns:1fr 1fr;gap:12px}
    .pair{border:1px solid #e5e7eb;border-radius:12px;padding:12px;margin:14px 0}
    .score{font-weight:600;margin-bottom:6px}
    .badge{display:inline-block;background:#eef;padding:2px 8px;border-radius:10px;margin-left:6px}
    pre{white-space:pre-wrap;word-wrap:break-word}
    mark{background:#fff59d}
    .snippet{border:1px dashed #ddd;border-radius:8px;padding:8px;margin-top:8px}
    .meta{color:#666;font-size:12px}
    h3{margin:8px 0}
    </style>""")
    html.append(f"<h2>Summary</h2><pre>{_html(json.dumps(summary, ensure_ascii=False, indent=2))}</pre>")
    html.append("<h2>Top Matches</h2>")
    for r in sorted(results, key=lambda x: x["score"], reverse=True)[:200]:
        html.append('<div class="pair">')
        html.append(f'<div class="score">Score: {r["score"]:.3f}<span class="badge">{r["level"]}</span> — cross={r["scores"]["cross"]:.3f} bi={r["scores"]["bi"]:.3f} lex={r["scores"]["lex"]:.3f}</div>')
        html.append('<div class="grid">')
        # A
        a_text = r["a"]["text"]; spans = sorted(r["a"]["spans"])
        cur = 0; buf = ["<pre>"]
        for s,e in spans:
            s = max(0, min(len(a_text), s)); e = max(0, min(len(a_text), e))
            if cur < s: buf.append(_html(a_text[cur:s]))
            buf.append("<mark>"+_html(a_text[s:e])+"</mark>"); cur = e
        if cur < len(a_text): buf.append(_html(a_text[cur:]))
        buf.append("</pre>")
        html.append(f'<div><h3>A[{r["a_chunk_id"]}]</h3>{"".join(buf)}')
        html.append('<div class="snippet"><b>Snippets:</b><br/>')
        for sn in r["a"]["snippets"][:3]:
            html.append(f'<pre>...{_html(sn["before"])}<mark>{_html(sn["hit"])}</mark>{_html(sn["after"])}...</pre>')
        html.append('</div></div>')
        # B
        b_text = r["b"]["text"]; spans = sorted(r["b"]["spans"])
        cur = 0; buf = ["<pre>"]
        for s,e in spans:
            s = max(0, min(len(b_text), s)); e = max(0, min(len(b_text), e))
            if cur < s: buf.append(_html(b_text[cur:s]))
            buf.append("<mark>"+_html(b_text[s:e])+"</mark>"); cur = e
        if cur < len(b_text): buf.append(_html(b_text[cur:]))
        buf.append("</pre>")
        html.append(f'<div><h3>B[{r["b_chunk_id"]}]</h3>{"".join(buf)}')
        html.append('<div class="snippet"><b>Snippets:</b><br/>')
        for sn in r["b"]["snippets"][:3]:
            html.append(f'<pre>...{_html(sn["before"])}<mark>{_html(sn["hit"])}</mark>{_html(sn["after"])}...</pre>')
        html.append('</div></div>')
        html.append('</div>')
        html.append(f'<div class="meta">A char-span in doc: {r["a"]["char_span_in_doc"]} | B char-span in doc: {r["b"]["char_span_in_doc"]}</div>')
        html.append('</div>')
    (outp / "report.html").write_text("\n".join(html), encoding="utf-8")

    return {"ok": True, "out": str(outp), "summary": summary}
