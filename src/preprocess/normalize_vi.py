# =========================
# file: src/preprocess/normalize_vi.py
# =========================
import unicodedata
from typing import Tuple

_ACCENT_MAP = str.maketrans(
    "àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ",
    "aaaaaaaaaaaaaaaaaeeeeeeeeeeeiiiii" + "ooooooooooooooooo" + "uuuuuuuuuuu" + "yyyyyd"
)  # coarse

def remove_accents(text: str) -> str:
    # Why: song song có dấu/không dấu tăng recall.
    nfkd = unicodedata.normalize('NFKD', text)
    no_diacritics = "".join([c for c in nfkd if not unicodedata.combining(c)])
    return no_diacritics.translate(_ACCENT_MAP)

def normalize_text_dual(text: str, lowercase: bool = True, unicode_norm: str = "NFC") -> Tuple[str, str]:
    t = unicodedata.normalize(unicode_norm, text)
    if lowercase:
        t = t.lower()
    return t, remove_accents(t)
