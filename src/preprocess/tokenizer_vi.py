# =========================
# file: src/preprocess/tokenizer_vi.py
# =========================
"""
Tokenizer ưu tiên tách từ tiếng Việt (pyvi/underthesea). Fallback regex.
Trả về danh sách token đã word-seg cho downstream (BM25/SimHash/ANN).
"""
import re
from typing import List

try:
    from pyvi import ViTokenizer  # type: ignore
except Exception:
    ViTokenizer = None

try:
    import underthesea  # type: ignore
except Exception:
    underthesea = None

_TOKEN_FALLBACK = re.compile(r"\w+|[^\w\s]", re.UNICODE)

def vi_word_tokenize(text: str) -> List[str]:
    if ViTokenizer is not None:
        try:
            return ViTokenizer.tokenize(text).split()
        except Exception:
            pass
    if underthesea is not None:
        try:
            return underthesea.word_tokenize(text)
        except Exception:
            pass
    return _TOKEN_FALLBACK.findall(text)
