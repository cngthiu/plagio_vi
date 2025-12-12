# =========================
# file: src/cli/index_corpus.py
# =========================
import argparse
import json
from pathlib import Path

from src.utils.common import load_yaml  # Dùng hàm chung
from src.io.docx_reader import read_docx_with_tables
from src.preprocess.normalize_vi import normalize_text_dual
from src.preprocess.sentence_seg import split_sentences_with_offsets_vi
from src.preprocess.chunker import chunk_sentences_window
from src.models.embedder import BiEncoder
from src.candidate.ann_index import ANNIndex
from src.utils.hardware import select_device_config

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--hardware", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--index_dir", required=True)
    args = ap.parse_args()

    # Code gọn hơn nhờ dùng hàm chung
    cfg = load_yaml(args.config)
    hw = load_yaml(args.hardware)
    faiss_cfg = load_yaml(hw["index"]["faiss"]["cfg_path"])

    device_cfg = select_device_config(hw)
    bi = BiEncoder(device_cfg=device_cfg, max_seq_len=cfg["chunking"]["max_seq_len_bi"],
                   backend=hw["inference"]["bi_encoder"]["backend"],
                   quantize=hw["inference"]["bi_encoder"]["quantize"])

    all_chunks = []
    # Quét đệ quy tất cả file docx
    for p in Path(args.corpus).glob("**/*.docx"):
        doc = read_docx_with_tables(str(p), keep_tables=cfg["io"]["keep_tables"])
        doc.linearize()
        norm_acc, _ = normalize_text_dual(doc.linearized_text,
                                          lowercase=cfg["preprocess"]["lowercase"],
                                          unicode_norm=cfg["preprocess"]["unicode_normalize"])
        sents_with_off = split_sentences_with_offsets_vi(norm_acc)
        align, disp, _ = chunk_sentences_window(
            norm_acc,
            sents_with_off,
            size_tokens=cfg["chunking"]["size_tokens"],
            overlap_tokens=cfg["chunking"]["overlap"],
            lowercase=cfg["preprocess"]["lowercase"]
        )
        all_chunks.extend(disp)

    emb = bi.encode(all_chunks, batch_size=(hw["inference"]["bi_encoder"].get("batch_size_gpu")
                                            or hw["inference"]["bi_encoder"].get("batch_size_cpu", 32)))
    
    # ANN Build
    ann = ANNIndex(dim=emb.shape[1], cfg=faiss_cfg)
    ann.build(emb)
    
    # Save
    out_path = Path(args.index_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    ann.save(out_path / "faiss.index")
    (out_path / "meta.json").write_text(json.dumps({"num_chunks": len(all_chunks)}, ensure_ascii=False), encoding="utf-8")
    
    print(json.dumps({"ok": True, "index_dir": args.index_dir, "num_chunks": len(all_chunks)}))

if __name__ == "__main__":
    main()