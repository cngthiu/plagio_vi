# =========================
# file: src/cli/index_corpus.py
# =========================
import argparse
import json
from pathlib import Path
import yaml

from src.io.docx_reader import read_docx_with_tables
from src.preprocess.normalize_vi import normalize_text_dual
from src.preprocess.sentence_seg import split_sentences_vi
from src.preprocess.tokenizer_vi import simple_tokenize
from src.preprocess.chunker import chunk_tokens_with_overlap
from src.models.embedder import BiEncoder
from src.candidate.ann_index import ANNIndex
from src.utils.hardware import select_device_config

def load_yaml(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--hardware", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--index_dir", required=True)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    hw = load_yaml(args.hardware)
    faiss_cfg = load_yaml(hw["index"]["faiss"]["cfg_path"])

    device_cfg = select_device_config(hw)
    bi = BiEncoder(device_cfg=device_cfg, max_seq_len=cfg["chunking"]["max_seq_len_bi"],
                   backend=hw["inference"]["bi_encoder"]["backend"],
                   quantize=hw["inference"]["bi_encoder"]["quantize"])

    all_chunks = []
    for p in Path(args.corpus).glob("**/*.docx"):
        doc = read_docx_with_tables(str(p), keep_tables=cfg["io"]["keep_tables"])
        doc.linearize()
        norm_acc, _ = normalize_text_dual(doc.linearized_text,
                                          lowercase=cfg["preprocess"]["lowercase"],
                                          unicode_norm=cfg["preprocess"]["unicode_normalize"])
        sents = split_sentences_vi(norm_acc)
        toks = [t for s in sents for t in simple_tokenize(s)]
        chunks, _ = chunk_tokens_with_overlap(toks, size=cfg["chunking"]["size_tokens"], overlap=cfg["chunking"]["overlap"])
        all_chunks.extend(chunks)

    emb = bi.encode(all_chunks, batch_size=(hw["inference"]["bi_encoder"].get("batch_size_gpu")
                                            or hw["inference"]["bi_encoder"].get("batch_size_cpu", 32)))
    ann = ANNIndex(dim=emb.shape[1], cfg=faiss_cfg)
    ann.build(emb)
    Path(args.index_dir).mkdir(parents=True, exist_ok=True)
    ann.save(Path(args.index_dir, "faiss.index"))
    Path(args.index_dir, "meta.json").write_text(json.dumps({"num_chunks": len(all_chunks)}, ensure_ascii=False))
    print(json.dumps({"ok": True, "index_dir": args.index_dir, "num_chunks": len(all_chunks)}))

if __name__ == "__main__":
    main()