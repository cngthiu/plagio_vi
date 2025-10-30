# =========================
# file: src/dicts/filters.py
# =========================
from typing import Set
from .loader import Dicts

def apply_boilerplate_filters(text: str, dicts: Dicts) -> str:
    # Why: giảm trùng lặp giả do văn mẫu.
    for phrase in dicts.boilerplate:
        text = text.replace(phrase, " ")
    return text

def expand_abbreviations(text: str, abbr: dict) -> str:
    for k, v in abbr.items():
        text = text.replace(k, v)
    return text