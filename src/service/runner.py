# =========================
# file: src/service/runner.py
# =========================
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Tuple
import concurrent.futures
import json
import numpy as np

# Imports giữ nguyên...
from src.io.docx_reader import read_docx_with_tables
from src.io.writer_report import write_json_report, write_html_report 
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
    stopphrase_penalty,
    citation_penalty,
)
from src.alignment.tiling import greedy_string_tiling
from src.preprocess.sentence_seg import split_sentences_with_offsets_vi
from src.preprocess.chunker import chunk_sentences_window, chunk_text_by_paragraphs
from src.preprocess.normalize_vi import normalize_for_alignment, normalize_for_retrieval
from src.dicts.loader import load_dictionaries
from src.utils.timing import Timer
from src.utils.hardware import set_num_threads, select_device_config
from src.utils.logging import JsonLogger
from src.utils.cache import DiskCache
from src.utils.common import load_yaml
from src.scoring.combine import fuse_scores, get_weights_from_cfg, level_of
from src.report.highlight import make_snippets

# [UPDATE] Đọc tham số từ config thay vì hardcode
def _nms_spans(pairs: List[Tuple[int,int,int,int]], iou_th: float) -> List[Tuple[int,int,int,int]]:
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

def _span_coverage(spans: List[Tuple[int, int]], text_len: int) -> float:
    if not spans or text_len <= 0:
        return 0.0
    covered = sum(max(0, e - s) for s, e in spans)
    return min(1.0, covered / text_len)

def _sentence_offsets_in_text(text: str, sentences: List[str]) -> List[Tuple[int, int]]:
    offsets = []
    cur = 0
    for s in sentences:
        start = text.find(s, cur)
        if start == -1:
            start = cur
        end = start + len(s)
        offsets.append((start, end))
        cur = end
    return offsets

def run_compare(
    config_path: str,
    hardware_path: str,
    path_a: str,
    path_b: str,
    out_dir: str,
    bi_encoder: BiEncoder | None = None,
    cross_encoder: CrossEncoder | None = None,
) -> Dict[str, Any]:
    # 1. Load Config & Init
    cfg = load_yaml(config_path)
    hw = load_yaml(hardware_path)
    faiss_cfg = load_yaml(hw["index"]["faiss"]["cfg_path"])
    dicts = load_dictionaries(cfg.get("dictionary_paths", {}))

    # [UPDATE] Load params từ config
    ALIGN_GAP = cfg.get("alignment", {}).get("gap_merge", 5)
    ALIGN_MIN_LEN = cfg.get("alignment", {}).get("min_len_local", 20)
    IOU_TH = cfg.get("alignment", {}).get("iou_threshold", 0.6)
    SEM_ALIGN = cfg.get("alignment", {}).get("semantic_enabled", True)
    SEM_TH = float(cfg.get("alignment", {}).get("semantic_threshold", 0.78))
    SEM_MAX = int(cfg.get("alignment", {}).get("semantic_topk", 3))
    SEM_MIN_COV = float(cfg.get("alignment", {}).get("semantic_min_coverage", 0.15))
    CTX_CHARS = cfg.get("report", {}).get("snippet_context_chars", 40)
    ANN_BF_THRESH = cfg.get("ann", {}).get("force_brute_force_threshold", 5000)

    outp = Path(out_dir); outp.mkdir(parents=True, exist_ok=True)
    set_num_threads(hw["device"].get("omp_threads", 8), hw["device"].get("mkl_threads", 8))
    device_cfg = select_device_config(hw)
    logger = JsonLogger(outp / "logs.jsonl")
    timer = Timer(logger=logger)
    cache = DiskCache(base_dir="outputs/cache")

    # Load Boilerplate logic... (giữ nguyên)
    boiler_paths = []
    if cfg.get("domain_boilerplate", {}).get("enabled", False):
        boiler_paths = cfg["domain_boilerplate"].get("paths", [])
    boiler_set = set()
    boiler_set.update(getattr(dicts, "boilerplate", set()))
    for p in boiler_paths:
        try:
            for line in open(p, encoding="utf-8"):
                line = line.strip()
                if line: boiler_set.add(line.lower())
        except Exception: pass

    section_headers = []
    section_headers_path = cfg.get("dictionary_paths", {}).get("section_headers", "")
    if section_headers_path:
        try:
            with open(section_headers_path, encoding="utf-8") as f:
                section_headers = [l.strip() for l in f if l.strip()]
        except Exception:
            section_headers = []
    
    weights_obj = get_weights_from_cfg(cfg)
    pen = cfg.get("penalties", {})
    LAMBDA = float(pen.get("lex_boilerplate_lambda", 0.5))
    STOP_LAMBDA = float(pen.get("stopphrase_lambda", 0.2))
    CIT_LAMBDA = float(pen.get("citation_lambda", 0.15))
    MIN_SPAN = int(pen.get("min_span_chars", 24))
    SMALL_SPAN = int(pen.get("small_span_chars", 12))
    MIN_SMALL = int(pen.get("min_small_spans", 2))

    # 2. Read Docs & Chunking... (giữ nguyên)
    with timer.section("read_docs"):
        docA = read_docx_with_tables(path_a, keep_tables=cfg["io"]["keep_tables"])
        docB = read_docx_with_tables(path_b, keep_tables=cfg["io"]["keep_tables"])
        docA.linearize(); docB.linearize()
        docA.apply_ignores(cfg["io"]["ignore_sections"])
        docB.apply_ignores(cfg["io"]["ignore_sections"])
        docA.linearize(); docB.linearize()
        textA_disp = docA.linearized_text
        textB_disp = docB.linearized_text

    with timer.section("chunking"):
        chunk_mode = cfg.get("chunking", {}).get("mode", "sentence")
        if chunk_mode == "paragraph":
            _, chunksA_disp, spansA_char = chunk_text_by_paragraphs(
                textA_disp,
                size_tokens=cfg["chunking"]["size_tokens"],
                overlap_tokens=cfg["chunking"]["overlap"],
                lowercase=cfg["preprocess"].get("lowercase", True),
                heading_lines=section_headers,
            )
            _, chunksB_disp, spansB_char = chunk_text_by_paragraphs(
                textB_disp,
                size_tokens=cfg["chunking"]["size_tokens"],
                overlap_tokens=cfg["chunking"]["overlap"],
                lowercase=cfg["preprocess"].get("lowercase", True),
                heading_lines=section_headers,
            )
        else:
            sentsA = split_sentences_with_offsets_vi(textA_disp)
            sentsB = split_sentences_with_offsets_vi(textB_disp)
            _, chunksA_disp, spansA_char = chunk_sentences_window(
                textA_disp, sentsA,
                size_tokens=cfg["chunking"]["size_tokens"],
                overlap_tokens=cfg["chunking"]["overlap"],
                lowercase=cfg["preprocess"].get("lowercase", True)
            )
            _, chunksB_disp, spansB_char = chunk_sentences_window(
                textB_disp, sentsB,
                size_tokens=cfg["chunking"]["size_tokens"],
                overlap_tokens=cfg["chunking"]["overlap"],
                lowercase=cfg["preprocess"].get("lowercase", True)
            )

        chunksA_align = [
            normalize_for_alignment(
                c,
                lowercase=cfg["preprocess"].get("lowercase", True),
                mask_numbers=cfg["preprocess"].get("mask_numbers", True),
                strip_zero_width=cfg["preprocess"].get("strip_zero_width", True),
            )
            for c in chunksA_disp
        ]
        chunksB_align = [
            normalize_for_alignment(
                c,
                lowercase=cfg["preprocess"].get("lowercase", True),
                mask_numbers=cfg["preprocess"].get("mask_numbers", True),
                strip_zero_width=cfg["preprocess"].get("strip_zero_width", True),
            )
            for c in chunksB_disp
        ]
        chunksA_retr = [
            normalize_for_retrieval(
                c,
                lowercase=cfg["preprocess"].get("lowercase", True),
                unicode_norm=cfg["preprocess"].get("unicode_normalize", "NFC"),
                expand_abbrev=cfg["preprocess"].get("abbreviations_expand", True),
                apply_synonyms=cfg["preprocess"].get("synonyms_expand", True),
                remove_stop_phrases=cfg["preprocess"].get("remove_stop_phrases", True),
                remove_boilerplate=cfg["preprocess"].get("remove_boilerplate", True),
                abbreviations=getattr(dicts, "abbreviations", {}),
                synonyms=getattr(dicts, "synonyms", {}),
                stop_phrases=getattr(dicts, "stop_phrases", set()),
                boilerplate=boiler_set,
            )
            for c in chunksA_disp
        ]
        chunksB_retr = [
            normalize_for_retrieval(
                c,
                lowercase=cfg["preprocess"].get("lowercase", True),
                unicode_norm=cfg["preprocess"].get("unicode_normalize", "NFC"),
                expand_abbrev=cfg["preprocess"].get("abbreviations_expand", True),
                apply_synonyms=cfg["preprocess"].get("synonyms_expand", True),
                remove_stop_phrases=cfg["preprocess"].get("remove_stop_phrases", True),
                remove_boilerplate=cfg["preprocess"].get("remove_boilerplate", True),
                abbreviations=getattr(dicts, "abbreviations", {}),
                synonyms=getattr(dicts, "synonyms", {}),
                stop_phrases=getattr(dicts, "stop_phrases", set()),
                boilerplate=boiler_set,
            )
            for c in chunksB_disp
        ]

    # 3. Retrieval Indexes
    with timer.section("indexes"):
        bm25 = BM25Index(chunksB_retr)
        shB  = SimHashIndex([simhash(c) for c in chunksB_retr])

    # 4. Load Models & Embedding
    with timer.section("models_load"):
        bi = bi_encoder or BiEncoder(
            device_cfg=device_cfg,
            max_seq_len=cfg["chunking"]["max_seq_len_bi"],
            backend=hw["inference"]["bi_encoder"]["backend"],
            quantize=hw["inference"]["bi_encoder"]["quantize"]
        )
        ce = cross_encoder or CrossEncoder(
            device_cfg=device_cfg,
            max_seq_len=cfg["chunking"]["max_seq_len_cross"],
            backend=hw["inference"]["cross_encoder"]["backend"],
            quantize=hw["inference"]["cross_encoder"]["quantize"]
        )

    with timer.section("embed_B"):
        kB = cache.key_for_texts(chunksB_retr, getattr(bi, "model_name", "bi"))
        embB = cache.get_numpy("embeddings", kB)
        if embB is None:
            embB = bi.encode(chunksB_retr, batch_size=hw["inference"]["bi_encoder"].get("batch_size_cpu", 32))
            cache.set_numpy("embeddings", kB, embB)
        
        # [UPDATE] Smart switching: Nếu Corpus B nhỏ -> Dùng Brute Force (FLAT)
        if len(chunksB_disp) < ANN_BF_THRESH:
            # Override config to force brute-force (bypass FAISS HNSW)
            # ANNIndex fallbacks to brute-force if type != HNSW
            local_ann_cfg = faiss_cfg.copy()
            local_ann_cfg["type"] = "FLAT_BRUTE_FORCE"
            ann = ANNIndex(dim=None, cfg=local_ann_cfg)
        else:
            ann = ANNIndex(dim=None, cfg=faiss_cfg)
            
        ann.build(embB)

    with timer.section("embed_A"):
        kA = cache.key_for_texts(chunksA_retr, getattr(bi, "model_name", "bi"))
        embA = cache.get_numpy("embeddings", kA)
        if embA is None:
            embA = bi.encode(chunksA_retr, batch_size=hw["inference"]["bi_encoder"].get("batch_size_cpu", 32))
            cache.set_numpy("embeddings", kA, embA)
        shA_vals = [simhash(c) for c in chunksA_retr]

    # 5. Matching & Reranking...
    bm25_topk   = cfg["bm25"]["topk"]
    ann_topk    = cfg["ann"]["topk_recall"]
    sim_topk    = cfg["simhash"]["topk"]
    rerank_topk = cfg["ann"]["topk_rerank"]
    # ... (giữ nguyên config gating/prefilter)
    CE_MIN = float(cfg.get("gating", {}).get("ce_min_bi", 0.25))
    CE_MAX = float(cfg.get("gating", {}).get("ce_max_bi", 0.92))
    PREFILTER_BI = float(cfg.get("gating", {}).get("prefilter_bi", 0.30))
    PREFILTER_LEX = float(cfg.get("gating", {}).get("prefilter_lex", 0.20))
    ALIGN_WORKERS = int(cfg.get("parallel", {}).get("alignment_workers", 0))

    results: List[Dict[str, Any]] = []
    citation_patterns = cfg.get("penalties", {}).get("citation_patterns", [])
    stop_phrases = getattr(dicts, "stop_phrases", set())

    def _passes_span_rule(spans: List[Tuple[int,int]]) -> bool:
        if not spans: return False
        if any((e - s) >= MIN_SPAN for s, e in spans): return True
        return sum(1 for s, e in spans if (e - s) >= SMALL_SPAN) >= MIN_SMALL

    def _semantic_span_pairs(a_text: str, b_text: str) -> List[Tuple[int,int,int,int]]:
        sentsA = split_sentences_with_offsets_vi(a_text)
        sentsB = split_sentences_with_offsets_vi(b_text)
        if not sentsA or not sentsB:
            return []
        offsA = _sentence_offsets_in_text(a_text, sentsA)
        offsB = _sentence_offsets_in_text(b_text, sentsB)
        normA = [
            normalize_for_retrieval(
                s,
                lowercase=cfg["preprocess"].get("lowercase", True),
                unicode_norm=cfg["preprocess"].get("unicode_normalize", "NFC"),
                expand_abbrev=cfg["preprocess"].get("abbreviations_expand", True),
                apply_synonyms=cfg["preprocess"].get("synonyms_expand", True),
                remove_stop_phrases=False,
                remove_boilerplate=False,
                abbreviations=getattr(dicts, "abbreviations", {}),
                synonyms=getattr(dicts, "synonyms", {}),
                stop_phrases=None,
                boilerplate=None,
            )
            for s in sentsA
        ]
        normB = [
            normalize_for_retrieval(
                s,
                lowercase=cfg["preprocess"].get("lowercase", True),
                unicode_norm=cfg["preprocess"].get("unicode_normalize", "NFC"),
                expand_abbrev=cfg["preprocess"].get("abbreviations_expand", True),
                apply_synonyms=cfg["preprocess"].get("synonyms_expand", True),
                remove_stop_phrases=False,
                remove_boilerplate=False,
                abbreviations=getattr(dicts, "abbreviations", {}),
                synonyms=getattr(dicts, "synonyms", {}),
                stop_phrases=None,
                boilerplate=None,
            )
            for s in sentsB
        ]
        emb_sa = bi.encode(normA, batch_size=hw["inference"]["bi_encoder"].get("batch_size_cpu", 32))
        emb_sb = bi.encode(normB, batch_size=hw["inference"]["bi_encoder"].get("batch_size_cpu", 32))
        sim = emb_sa @ emb_sb.T
        pairs = []
        for i in range(sim.shape[0]):
            j = int(np.argmax(sim[i]))
            score = float(sim[i, j])
            if score >= SEM_TH:
                pairs.append((offsA[i][0], offsA[i][1], offsB[j][0], offsB[j][1], score))
        pairs = sorted(pairs, key=lambda x: x[4], reverse=True)[:SEM_MAX]
        return [(a0, a1, b0, b1) for a0, a1, b0, b1, _s in pairs]

    with timer.section("match_rerank_align"):
        for i, (a_align, a_disp, a_retr) in enumerate(zip(chunksA_align, chunksA_disp, chunksA_retr)):
            # Search... (giữ nguyên)
            bm_idx  = bm25.search(a_retr, topk=bm25_topk)
            sim_idx = shB.topk_by_hamming(shA_vals[i], k=sim_topk)
            ann_idx = ann.search(embA[i], topk=ann_topk)
            cand = sorted(set(bm_idx + sim_idx + ann_idx))[:max(bm25_topk, ann_topk, sim_topk)]
            if not cand: continue

            # Scoring... (giữ nguyên)
            s_bi_list, s_lex_list, s_bm_list = [], [], []
            for j in cand:
                b_align = chunksB_align[j]
                b_retr = chunksB_retr[j]
                s_bi  = cosine_sim(embA[i], embB[j])
                base_jac = jaccard_char_ngrams(a_retr, b_retr, n=5)
                base_ham = hamming_similarity(shA_vals[i], simhash(b_retr))
                base_lex = 0.5*base_jac + 0.5*base_ham
                bp = boilerplate_penalty(a_disp, chunksB_disp[j], boiler_set)
                # Giảm false positives do cụm phổ biến và trích dẫn hợp lệ.
                sp = stopphrase_penalty(a_retr, b_retr, stop_phrases)
                cp = citation_penalty(a_disp, chunksB_disp[j], citation_patterns)
                s_lex = base_lex * (1.0 - LAMBDA * bp) * (1.0 - STOP_LAMBDA * sp) * (1.0 - CIT_LAMBDA * cp)
                s_bm  = bm25.score_of_cached(j)
                s_bi_list.append(s_bi); s_lex_list.append(s_lex); s_bm_list.append(s_bm)

            # Prefilter & Rerank... (giữ nguyên logic)
            filtered_idx = []
            for idx_c, (sb, sl) in zip(cand, zip(s_bi_list, s_lex_list)):
                if sb >= PREFILTER_BI or sl >= PREFILTER_LEX:
                    filtered_idx.append(idx_c)
            if not filtered_idx: filtered_idx = cand[:3]

            order_by_bi = np.argsort(-np.asarray([s_bi_list[cand.index(x)] for x in filtered_idx]))[:rerank_topk]
            ce_inputs = [(a_disp, chunksB_disp[filtered_idx[ix]]) for ix in order_by_bi]

            ce_scores = {}
            if ce_inputs:
                ce_mask = []
                for idx_local, gidx in enumerate(order_by_bi):
                    sb = s_bi_list[cand.index(filtered_idx[gidx])]
                    if CE_MIN <= sb <= CE_MAX:
                        ce_mask.append((idx_local, filtered_idx[gidx]))
                if ce_mask:
                    pairs = [ce_inputs[x[0]] for x in ce_mask]
                    s_cross = ce.score_pairs(pairs, batch_size=hw["inference"]["cross_encoder"].get("batch_size_cpu", 4))
                    for (mask_idx, gidx), sc in zip(ce_mask, s_cross):
                        ce_scores[gidx] = sc

            cross_full = []
            for j in cand:
                cross_full.append(ce_scores.get(j, 0.0))

            # Fuse Scores
            S = fuse_scores(cross_full, s_bi_list, s_lex_list, s_bm_list, weights_obj)
            keep_ord = np.argsort(-np.asarray(S))[:5]

            def _align_job(pair_idx: int):
                j_local = cand[pair_idx]
                a_txt_a, b_txt_a = chunksA_align[i], chunksB_align[j_local]
                
                # [TECH LEAD UPDATE] Chỉ dùng Greedy Tiling (RapidFuzz)
                # Nhanh gấp 100 lần so với difflib mà độ chính xác tương đương cho mục đích bắt đạo văn
                spans_pairs = greedy_string_tiling(a_txt_a, b_txt_a, min_match_len=ALIGN_MIN_LEN)
                
                # Fallback cũ: if not spans_pairs: ... -> DELETE THIS BLOCK
                
                if SEM_ALIGN:
                    spans_a = [(a0, a1) for (a0, a1, _, _) in spans_pairs]
                    cov_a = _span_coverage(spans_a, len(a_txt_a))
                    if cov_a < SEM_MIN_COV:
                        # Semantic alignment để bắt paraphrase mạnh khi tiling yếu.
                        sem_pairs = _semantic_span_pairs(a_disp, chunksB_disp[j_local])
                        spans_pairs = spans_pairs + sem_pairs

                spans_pairs = _nms_spans(spans_pairs, iou_th=IOU_TH)
                spans_a = [(a0,a1) for (a0,a1,_,_) in spans_pairs]
                spans_b = [(b0,b1) for (_,_,b0,b1) in spans_pairs]
                
                return {
                    "j": j_local,
                    "spans_pairs": spans_pairs,
                    "spans_a": spans_a,
                    "spans_b": spans_b,
                    "score": float(S[pair_idx]),
                    "scores": {
                        "cross": float(cross_full[pair_idx]),
                        "bi": float(s_bi_list[pair_idx]),
                        "lex": float(s_lex_list[pair_idx]),
                        "bm25_raw": float(s_bm_list[pair_idx]),
                    },
                }

            align_inputs = list(keep_ord)
            align_results = []
            if ALIGN_WORKERS > 1 and len(align_inputs) > 1:
                with concurrent.futures.ThreadPoolExecutor(max_workers=ALIGN_WORKERS) as ex:
                    for res in ex.map(_align_job, align_inputs):
                        align_results.append(res)
            else:
                for idx_in_keep in align_inputs:
                    align_results.append(_align_job(idx_in_keep))

            for ar in align_results:
                if not _passes_span_rule(ar["spans_a"]) or not _passes_span_rule(ar["spans_b"]):
                    continue
                j = ar["j"]
                rec = {
                    "a_chunk_id": i,
                    "b_chunk_id": j,
                    "score": ar["score"],
                    "level": level_of(float(ar["score"])),
                    "scores": ar["scores"],
                    "a": {
                        "text": chunksA_disp[i],
                        "spans": ar["spans_a"],
                        # [UPDATE] Pass tham số ctx
                        "snippets": make_snippets(chunksA_disp[i], ar["spans_a"], ctx=CTX_CHARS),
                        "char_span_in_doc": [spansA_char[i][0], spansA_char[i][1]],
                    },
                    "b": {
                        "text": chunksB_disp[j],
                        "spans": ar["spans_b"],
                        "snippets": make_snippets(chunksB_disp[j], ar["spans_b"], ctx=CTX_CHARS),
                        "char_span_in_doc": [spansB_char[j][0], spansB_char[j][1]],
                    },
                }
                results.append(rec)

    # 6. Summary & Export... (giữ nguyên)
    with timer.section("aggregate"):
        levels = {"very_high":0,"high":0,"medium":0,"low":0}
        for r in results: levels[r["level"]] += 1
        summary = {"chunks_A": len(chunksA_disp), "chunks_B": len(chunksB_disp), "pairs": len(results), "levels": levels}

    write_json_report(outp / "report.json", docA, docB, results, {"summary": summary}, cfg, hw)
    write_html_report(outp / "report.html", docA, docB, results, {"summary": summary}, cfg, hw)

    return {"ok": True, "out": str(outp), "summary": summary}
