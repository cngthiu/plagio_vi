# =========================
# file: src/preprocess/chunker.py
# =========================
from typing import Iterable, List, Tuple
from src.preprocess.tokenizer_vi import simple_tokenize
from src.preprocess.sentence_seg import split_sentences_with_offsets_vi

def _build_chunks(
    sentences: List[str],
    sent_offsets: List[Tuple[int, int]],
    size_tokens: int,
    overlap_tokens: int,
    lowercase: bool,
) -> Tuple[List[str], List[str], List[Tuple[int, int]]]:
    if not sentences:
        return [], [], []

    sent_tokens = [simple_tokenize(s) for s in sentences]
    sent_lens = [len(t) for t in sent_tokens]

    chunks_align = []
    chunks_disp = []
    spans_char = []

    step_tokens = max(1, size_tokens - overlap_tokens)
    i = 0
    while i < len(sentences):
        chunk_tok_count = 0
        j = i
        current_chunk_sents = []

        while j < len(sentences):
            wl = sent_lens[j]
            if chunk_tok_count + wl > size_tokens and chunk_tok_count > 0:
                break
            chunk_tok_count += wl
            current_chunk_sents.append(j)
            j += 1

        if not current_chunk_sents:
            i += 1
            continue

        s_idxs = current_chunk_sents
        c_disp = " ".join([sentences[k] for k in s_idxs])
        c_align = c_disp.lower() if lowercase else c_disp

        c_start = sent_offsets[s_idxs[0]][0]
        c_end = sent_offsets[s_idxs[-1]][1]

        chunks_disp.append(c_disp)
        chunks_align.append(c_align)
        spans_char.append((c_start, c_end))

        if j >= len(sentences):
            break

        tokens_skipped = 0
        next_i = i + 1
        for k in range(i, j):
            tokens_skipped += sent_lens[k]
            if tokens_skipped >= step_tokens:
                next_i = k + 1
                break

        if next_i <= i:
            next_i = i + 1
        i = next_i

    return chunks_align, chunks_disp, spans_char

def _sentence_offsets(text: str, sentences: List[str], start_offset: int = 0) -> List[Tuple[int, int]]:
    sent_offsets = []
    cur = 0
    for s in sentences:
        start = text.find(s, cur)
        if start == -1:
            start = cur
        end = start + len(s)
        sent_offsets.append((start_offset + start, start_offset + end))
        cur = end
    return sent_offsets

def chunk_sentences_window(
    text: str,
    sentences: List[str],
    size_tokens: int = 260,
    overlap_tokens: int = 50,
    lowercase: bool = True,
) -> Tuple[List[str], List[str], List[Tuple[int, int]]]:
    """
    Hàm mới cần thiết cho runner.py
    Input:
      - text: Văn bản gốc (để tính offset)
      - sentences: List các câu đã tách
    Output:
      - chunks_align: List chunk đã normalize (để model so sánh)
      - chunks_display: List chunk gốc (để hiển thị)
      - spans_char: Vị trí bắt đầu/kết thúc (char index) của chunk trong text gốc
    """
    if not sentences:
        return [], [], []

    sent_offsets = _sentence_offsets(text, sentences)
    return _build_chunks(sentences, sent_offsets, size_tokens, overlap_tokens, lowercase)

def _split_paragraphs_with_offsets(
    text: str,
    heading_lines: Iterable[str] | None = None,
) -> List[Tuple[str, int, int, bool]]:
    headings = {h.strip().lower() for h in (heading_lines or []) if h.strip()}
    blocks: List[Tuple[str, int, int, bool]] = []
    lines = text.splitlines(keepends=True)
    cur = []
    cur_start = None
    offset = 0
    for line in lines:
        stripped = line.strip()
        if not stripped:
            if cur:
                blk = "".join(cur)
                blocks.append((blk, cur_start or 0, offset, False))
                cur = []
                cur_start = None
            offset += len(line)
            continue

        is_heading = stripped.lower() in headings
        if is_heading:
            if cur:
                blk = "".join(cur)
                blocks.append((blk, cur_start or 0, offset, False))
                cur = []
                cur_start = None
            line_start = offset + line.find(stripped)
            blocks.append((stripped, line_start, line_start + len(stripped), True))
            offset += len(line)
            continue

        if cur_start is None:
            cur_start = offset
        cur.append(line)
        offset += len(line)

    if cur:
        blk = "".join(cur)
        blocks.append((blk, cur_start or 0, offset, False))
    return blocks

def chunk_text_by_paragraphs(
    text: str,
    size_tokens: int = 260,
    overlap_tokens: int = 50,
    lowercase: bool = True,
    heading_lines: Iterable[str] | None = None,
) -> Tuple[List[str], List[str], List[Tuple[int, int]]]:
    """
    Chunk theo paragraph/heading để tránh nối ngữ cảnh giữa các đoạn.
    """
    if not text:
        return [], [], []

    blocks = _split_paragraphs_with_offsets(text, heading_lines=heading_lines)
    chunks_align: List[str] = []
    chunks_disp: List[str] = []
    spans_char: List[Tuple[int, int]] = []

    for blk_text, blk_start, _blk_end, is_heading in blocks:
        if is_heading:
            disp = blk_text
            align = disp.lower() if lowercase else disp
            chunks_disp.append(disp)
            chunks_align.append(align)
            spans_char.append((blk_start, blk_start + len(disp)))
            continue

        sentences = split_sentences_with_offsets_vi(blk_text)
        if not sentences:
            continue
        sent_offsets = _sentence_offsets(blk_text, sentences, start_offset=blk_start)
        c_align, c_disp, c_spans = _build_chunks(
            sentences, sent_offsets, size_tokens, overlap_tokens, lowercase
        )
        chunks_align.extend(c_align)
        chunks_disp.extend(c_disp)
        spans_char.extend(c_spans)

    return chunks_align, chunks_disp, spans_char
