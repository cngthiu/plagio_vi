# =========================
# file: src/preprocess/sentence_seg.py
# =========================
from typing import List
from underthesea import sent_tokenize

def split_sentences_vi(text: str) -> List[str]:
    """
    Tách câu sử dụng model chuyên dụng cho tiếng Việt.
    Xử lý tốt các trường hợp viết tắt (Tp. HCM, PGS. TS.).
    """
    if not text or not text.strip():
        return []
    
    # sent_tokenize của underthesea trả về list các câu
    sentences = sent_tokenize(text)
    return [s.strip() for s in sentences if s.strip()]

# Giữ lại hàm cũ để tương thích ngược nếu cần, hoặc alias sang hàm mới
split_sentences_with_offsets_vi = split_sentences_vi