# =========================
# file: src/preprocess/chunker.py
# =========================
from typing import List, Tuple

def chunk_tokens_with_overlap(tokens: List[str], size: int = 260, overlap: int = 50) -> Tuple[List[str], List[Tuple[int, int]]]:
    chunks, spans = [], []
    if size <= 0:
        return chunks, spans
    step = max(1, size - overlap)
    for i in range(0, len(tokens), step):
        sl = tokens[i:i+size]
        if not sl:
            break
        chunks.append(" ".join(sl))
        spans.append((i, i+len(sl)))
        if i + size >= len(tokens):
            break
    return chunks, spans