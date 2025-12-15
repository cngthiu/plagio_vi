# =========================
# file: src/preprocess/chunker.py
# =========================
from typing import List, Tuple
from src.preprocess.tokenizer_vi import simple_tokenize

def chunk_sentences_window(text: str, sentences: List[str], size_tokens: int = 260, overlap_tokens: int = 50, lowercase: bool = True) -> Tuple[List[str], List[str], List[Tuple[int, int]]]:
    """
    Hàm mới cần thiết cho runner.py
    Input:
      - text: Văn bản gốc (để tính offset)
      - sentences: List các câu đã tách
    Output:
      - chunks_align: List chunk đã normalize (để model so sánh)
      - chunks_display: List chunk gốc (để hiển thị)
      - spans_char: Vị trí bắt đầu/kết thúc (char index) của chunk trong text gốc
    """
    if not sentences:
        return [], [], []

    # 1. Tokenize sơ bộ từng câu để đếm độ dài token
    sent_tokens = [simple_tokenize(s) for s in sentences]
    sent_lens = [len(t) for t in sent_tokens]
    
    # 2. Map sentence -> character offset trong text gốc
    # (Giả định sentences khớp thứ tự trong text)
    sent_offsets = []
    cur = 0
    text_lower = text.lower() if lowercase else text
    for s in sentences:
        # Tìm vị trí câu trong text (heuristic đơn giản)
        # Lưu ý: Cần logic tìm kiếm chính xác hơn nếu text có nhiều đoạn giống nhau
        # Ở đây dùng find tiếp diễn
        start = text.find(s, cur)
        if start == -1:
            # Fallback nếu không tìm thấy (do preprocessing làm lệch)
            start = cur
        end = start + len(s)
        sent_offsets.append((start, end))
        cur = end

    chunks_align = []
    chunks_disp = []
    spans_char = []

    # 3. Sliding window over sentences
    current_tokens = 0
    window_sents_idx = [] # Index các câu trong window hiện tại
    
    # Logic sliding window đơn giản: gom câu cho đến khi đủ token
    # Thực tế runner.py cần logic phức tạp hơn (overlap).
    # Đây là bản implement đơn giản nhất để chạy được:
    
    # Cách tiếp cận: Duyệt từng câu làm điểm bắt đầu, gom tiếp các câu sau cho đến khi đủ size
    step_tokens = max(1, size_tokens - overlap_tokens)
    
    # Gom tất cả token lại thành 1 list phẳng để slide dễ hơn? 
    # Không, vì ta cần giữ boundary câu để highlight cho đẹp.
    
    # Implement kiểu cửa sổ trượt trên danh sách câu (xấp xỉ token)
    i = 0
    while i < len(sentences):
        # Build 1 chunk bắt đầu từ câu i
        chunk_tok_count = 0
        j = i
        current_chunk_sents = []
        
        while j < len(sentences):
            wl = sent_lens[j]
            if chunk_tok_count + wl > size_tokens and chunk_tok_count > 0:
                # Quá size, dừng (trừ khi câu j quá dài thì vẫn phải lấy ít nhất 1 câu)
                break
            chunk_tok_count += wl
            current_chunk_sents.append(j)
            j += 1
            
        if not current_chunk_sents:
            i += 1
            continue

        # Tạo chunk text
        s_idxs = current_chunk_sents
        # Text hiển thị
        c_disp = " ".join([sentences[k] for k in s_idxs])
        # Text align (để model embedding)
        c_align = c_disp.lower() if lowercase else c_disp
        
        # Char span: từ đầu câu đầu tiên -> cuối câu cuối cùng
        c_start = sent_offsets[s_idxs[0]][0]
        c_end = sent_offsets[s_idxs[-1]][1]
        
        chunks_disp.append(c_disp)
        chunks_align.append(c_align)
        spans_char.append((c_start, c_end))

        # Move next stride
        # Logic overlap: Chúng ta muốn lùi lại bao nhiêu câu?
        # Tính toán để bước nhảy (step) xấp xỉ step_tokens
        
        if j >= len(sentences):
            break # Hết bài
            
        # Tìm vị trí i tiếp theo sao cho overlap đảm bảo
        # i mới phải > i cũ. 
        # Token overlap = chunk_tok_count - step_tokens (mong muốn)
        # => Tokens của phần giữ lại = overlap_tokens
        
        tokens_skipped = 0
        next_i = i + 1
        for k in range(i, j):
            tokens_skipped += sent_lens[k]
            if tokens_skipped >= step_tokens:
                next_i = k + 1
                break
        
        if next_i <= i: next_i = i + 1 # Luôn tiến lên
        i = next_i

    return chunks_align, chunks_disp, spans_char