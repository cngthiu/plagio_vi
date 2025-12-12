# =========================
# file: src/alignment/tiling.py
# =========================
from __future__ import annotations
from typing import List, Tuple
import logging

# Sử dụng RapidFuzz (C++ optimized) thay vì viết tay thuật toán chậm chạp.
# Indel distance ~ Longest Common Subsequence logic, phù hợp để tìm chuỗi chung.
try:
    from rapidfuzz.distance import Indel
except ImportError:
    # Fail fast: Báo lỗi ngay lập tức nếu thiếu dependency quan trọng
    raise ImportError("Module 'rapidfuzz' is required but not installed. Please check requirements.txt.")

# Cấu hình Logger riêng cho module
logger = logging.getLogger(__name__)

def _merge_nearby_spans(matches: List[Tuple[int, int, int, int]], gap: int = 10) -> List[Tuple[int, int, int, int]]:
    """
    Gộp các span nằm gần nhau.
    Input: list of (a_start, a_end, b_start, b_end)
    """
    if not matches:
        return []

    # Sắp xếp theo vị trí xuất hiện bên A
    matches.sort(key=lambda x: x[0])
    
    merged: List[Tuple[int, int, int, int]] = [matches[0]]
    
    for next_m in matches[1:]:
        a0, a1, b0, b1 = next_m
        last_a0, last_a1, last_b0, last_b1 = merged[-1]
        
        # Logic gộp: Nếu đoạn tiếp theo bắt đầu gần điểm kết thúc của đoạn trước
        # TRÊN CẢ HAI VĂN BẢN (A và B), thì gộp lại.
        # Điều kiện này đảm bảo tính liên tục của Alignment (monotonic).
        if (a0 <= last_a1 + gap) and (b0 <= last_b1 + gap):
            # Mở rộng span cũ
            new_a1 = max(last_a1, a1)
            new_b1 = max(last_b1, b1)
            merged[-1] = (last_a0, new_a1, last_b0, new_b1)
        else:
            merged.append(next_m)
            
    return merged

def greedy_string_tiling(a: str, b: str, min_match_len: int = 20) -> List[Tuple[int, int, int, int]]:
    """
    Tìm các đoạn trùng lặp giữa 2 chuỗi (Local Alignment).
    
    Refactored by Senior Tech Lead:
    - Thay thế thuật toán GST thủ công (O(N^3)) bằng RapidFuzz Indel Opcodes (O(N) ~ O(N*D)).
    - Logic: Tìm Longest Common Subsequence, trích xuất các block 'equal'.
    
    Args:
        a (str): Văn bản nguồn A
        b (str): Văn bản nguồn B
        min_match_len (int): Độ dài tối thiểu của span để được coi là trùng lặp.

    Returns:
        List[(a_start, a_end, b_start, b_end)]
    """
    if not a or not b:
        return []

    try:
        # Indel.opcodes trả về danh sách các thao tác (insert, delete, equal)
        # để biến đổi a thành b. Chúng ta chỉ quan tâm đến 'equal'.
        # Format op: (tag, a_start, a_end, b_start, b_end)
        ops = Indel.opcodes(a, b)
    except Exception as e:
        logger.error(f"Error computing alignment: {e}")
        return []

    matches = []
    for tag, a0, a1, b0, b1 in ops:
        if tag == 'equal':
            length = a1 - a0  # Hoặc b1 - b0, vì equal thì độ dài bằng nhau
            if length >= min_match_len:
                matches.append((a0, a1, b0, b1))
    
    # Bước hậu xử lý: Gộp các span bị đứt gãy nhỏ (do nhiễu 1-2 ký tự) thành span lớn
    # để báo cáo nhìn sạch sẽ hơn.
    final_spans = _merge_nearby_spans(matches, gap=10)
    
    return final_spans