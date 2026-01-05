## Plagio_VI

### 1. Project description
Plagio_VI là hệ thống phát hiện đạo văn tiếng Việt dựa trên semantic similarity, tối ưu cho văn bản dài như luận văn, hợp đồng, báo cáo.

### 2. Features
- So sánh hai tài liệu DOCX và tạo báo cáo HTML/JSON
- Phát hiện trùng lặp theo semantic + lexical (paraphrase nhẹ)
- Highlight đoạn trùng theo span và mức độ (low/medium/high)
- Lọc boilerplate, stop phrases, viết tắt phổ biến
- Hỗ trợ WebUI để xem kết quả trực quan

### 3. System pipeline
1) Đọc tài liệu → chuẩn hóa tiếng Việt  
2) Tách câu/đoạn → chunking  
3) Retrieval: BM25 + SimHash + ANN  
4) Rerank: Bi-encoder + Cross-encoder  
5) Alignment + Highlight  
6) Xuất báo cáo HTML/JSON

### 4. Tech stack
- Python 3.10+
- FastAPI, Uvicorn
- underthesea
- sentence-transformers
- FAISS

### 5. Installation
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 6. Run project
CLI:
```bash
python3 -m src.cli.compare_docs \
  --config configs/default.yaml \
  --hardware configs/hardware.gpu1650.yaml \
  --a /path/to/A.docx \
  --b /path/to/B.docx \
  --out outputs/run_A_B
```

WebUI:
```bash
uvicorn src.service.api:app --host 0.0.0.0 --port 8000
```
