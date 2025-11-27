# =========================
# file: src/preprocess/sentence_seg.py
# =========================
import re
from typing import List

_SENT_SPLIT = re.compile(r"(?<=[\.\!\?\:])\s+")

def split_sentences_vi(text: str) -> List[str]:
    return [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]

def split_sentences_with_offsets_vi(text: str) -> List[tuple[str,int,int]]:
    """
    Trả về list (sentence, start, end) theo chỉ số ký tự trong chuỗi gốc.
    Dựa trên split_sentences_vi nhưng bảo toàn offset để chunk theo câu.
    """
    out = []
    cursor = 0
    for sent in split_sentences_vi(text):
        idx = text.find(sent, cursor)
        if idx == -1:
            idx = cursor
        start = idx
        end = start + len(sent)
        out.append((sent, start, end))
        cursor = end
    return out
