# =========================
# file: src/preprocess/chunker.py
# =========================
from typing import List, Tuple

from src.preprocess.tokenizer_vi import vi_word_tokenize

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

def chunk_sentences_window(text: str, sentences_with_offsets: List[Tuple[str,int,int]],
                           size_tokens: int = 260, overlap_tokens: int = 50, lowercase: bool = True) -> Tuple[List[str], List[str], List[Tuple[int,int]]]:
    """
    Chunk theo cửa sổ trượt nhưng bảo toàn ranh giới câu.
    - text: chuỗi gốc (để cắt substring hiển thị)
    - sentences_with_offsets: list (sentence, start, end) trên text
    """
    if not sentences_with_offsets:
        return [], [], []
    toks_per_sent = [len(vi_word_tokenize(s[0])) for s in sentences_with_offsets]
    step_tokens = max(1, size_tokens - overlap_tokens)
    chunks_align, chunks_disp, spans_char = [], [], []
    idx = 0
    n = len(sentences_with_offsets)
    while idx < n:
        tokens = 0
        start_char = sentences_with_offsets[idx][1]
        end_char = sentences_with_offsets[idx][2]
        j = idx
        while j < n and (tokens < size_tokens or j == idx):
            tokens += toks_per_sent[j]
            end_char = sentences_with_offsets[j][2]
            j += 1
        chunk_disp = text[start_char:end_char]
        chunk_align = chunk_disp.lower() if lowercase else chunk_disp
        chunks_disp.append(chunk_disp)
        chunks_align.append(chunk_align)
        spans_char.append((start_char, end_char))

        # bước tiếp theo theo ngưỡng token (có overlap)
        tokens_step = 0
        idx_next = idx
        while idx_next < n and tokens_step < step_tokens:
            tokens_step += toks_per_sent[idx_next]
            idx_next += 1
        if idx_next == idx:  # an toàn
            idx_next += 1
        idx = idx_next
    return chunks_align, chunks_disp, spans_char
