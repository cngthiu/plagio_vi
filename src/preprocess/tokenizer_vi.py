# =========================
# file: src/preprocess/tokenizer_vi.py
# =========================
import re
from typing import List

_TOKEN = re.compile(r"\w+|[^\w\s]", re.UNICODE)

def simple_tokenize(text: str) -> List[str]:
    return _TOKEN.findall(text)