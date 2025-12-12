# =========================
# file: src/preprocess/tokenizer_vi.py
# =========================
from typing import List
from underthesea import word_tokenize

def simple_tokenize(text: str) -> List[str]:
    """
    Tách từ (Word Segmentation) thay vì tách ký tự trắng.
    Ví dụ: "sinh viên đại học" -> ["sinh viên", "đại học"]
    """
    if not text:
        return []
    
    # format="text" sẽ trả về chuỗi có gạch nối: "sinh_viên"
    # format=None (default) trả về list: ["sinh viên", "đại học"]
    # Với BM25, ta nên dùng list các từ ghép.
    return word_tokenize(text)