# =========================
# file: src/preprocess/sentence_seg.py
# =========================
import re
from typing import List

_SENT_SPLIT = re.compile(r"(?<=[\.\!\?\:])\s+")

def split_sentences_vi(text: str) -> List[str]:
    return [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]

