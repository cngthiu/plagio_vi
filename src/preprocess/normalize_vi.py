# =========================
# file: src/preprocess/normalize_vi.py
# =========================
import re
import unicodedata
from typing import Dict, Iterable, Optional, Tuple

_ACCENT_MAP = str.maketrans(
    "àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ",
    "aaaaaaaaaaaaaaaaaeeeeeeeeeeeiiiii" + "ooooooooooooooooo" + "uuuuuuuuuuu" + "yyyyyd"
)  # coarse

_ZERO_WIDTH = {
    "\u200b",  # ZERO WIDTH SPACE
    "\u200c",  # ZERO WIDTH NON-JOINER
    "\u200d",  # ZERO WIDTH JOINER
    "\u2060",  # WORD JOINER
    "\ufeff",  # ZERO WIDTH NO-BREAK SPACE
}

def remove_accents(text: str) -> str:
    # Why: song song có dấu/không dấu tăng recall.
    nfkd = unicodedata.normalize('NFKD', text)
    no_diacritics = "".join([c for c in nfkd if not unicodedata.combining(c)])
    return no_diacritics.translate(_ACCENT_MAP)

def _replace_zero_width(text: str, preserve_len: bool = True) -> str:
    if not text:
        return text
    if preserve_len:
        return "".join((" " if c in _ZERO_WIDTH else c) for c in text)
    return "".join((c for c in text if c not in _ZERO_WIDTH))

def _mask_numbers(text: str) -> str:
    # Preserve length to keep span offsets stable.
    return re.sub(r"\d", "0", text)

def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()

def _apply_word_map(text: str, mapping: Dict[str, str]) -> str:
    # Replace whole words only to avoid partial matches.
    if not mapping:
        return text
    keys = sorted(mapping.keys(), key=len, reverse=True)
    pattern = r"\b(" + "|".join(re.escape(k) for k in keys) + r")\b"
    def _repl(m):
        return mapping.get(m.group(0).lower(), m.group(0))
    return re.sub(pattern, _repl, text, flags=re.IGNORECASE)

def _remove_phrases(text: str, phrases: Iterable[str], preserve_len: bool = True) -> str:
    if not phrases:
        return text
    out = text
    for p in phrases:
        if not p:
            continue
        if preserve_len:
            out = out.replace(p, " " * len(p))
        else:
            out = out.replace(p, " ")
    return out

def normalize_text_dual(text: str, lowercase: bool = True, unicode_norm: str = "NFC") -> Tuple[str, str]:
    t = unicodedata.normalize(unicode_norm, text)
    if lowercase:
        t = t.lower()
    return t, remove_accents(t)

def normalize_for_alignment(
    text: str,
    lowercase: bool = True,
    mask_numbers: bool = True,
    strip_zero_width: bool = True,
) -> str:
    # Length-preserving normalization for alignment spans.
    t = text
    if strip_zero_width:
        t = _replace_zero_width(t, preserve_len=True)
    if lowercase:
        t = t.lower()
    if mask_numbers:
        t = _mask_numbers(t)
    return t

def normalize_for_retrieval(
    text: str,
    lowercase: bool = True,
    unicode_norm: str = "NFC",
    expand_abbrev: bool = True,
    apply_synonyms: bool = True,
    remove_stop_phrases: bool = True,
    remove_boilerplate: bool = False,
    abbreviations: Optional[Dict[str, str]] = None,
    synonyms: Optional[Dict[str, str]] = None,
    stop_phrases: Optional[Iterable[str]] = None,
    boilerplate: Optional[Iterable[str]] = None,
) -> str:
    # Semantic-friendly normalization for retrieval/embedding.
    t = unicodedata.normalize(unicode_norm, text)
    if lowercase:
        t = t.lower()
    t = _replace_zero_width(t, preserve_len=False)
    if expand_abbrev and abbreviations:
        t = _apply_word_map(t, abbreviations)
    if apply_synonyms and synonyms:
        t = _apply_word_map(t, synonyms)
    if remove_stop_phrases and stop_phrases:
        t = _remove_phrases(t, stop_phrases, preserve_len=False)
    if remove_boilerplate and boilerplate:
        t = _remove_phrases(t, boilerplate, preserve_len=False)
    t = _normalize_whitespace(t)
    return t
