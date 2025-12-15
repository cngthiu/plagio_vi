# =========================
# file: src/models/embedder.py
# =========================
from typing import List, Optional
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

class BiEncoder:
    def __init__(self, device_cfg: dict, max_seq_len: int = 256, backend: str = "torch", quantize: str = "fp16", model_name: Optional[str] = None):
        self.device = torch.device("cuda") if device_cfg["use_gpu"] else torch.device("cpu")
        self.max_seq_len = max_seq_len
        self.backend = backend
        
        # [TECH LEAD UPDATE] Thay model paraphrase bằng model chuyên tiếng Việt
        # Model này được train trên dữ liệu tiếng Việt, hiểu từ đồng nghĩa/ngữ cảnh tốt hơn nhiều.
        default_model = "bkai-foundation-models/vietnamese-bi-encoder" 
        self.model_name = model_name or default_model
        
        print(f"Loading Bi-Encoder: {self.model_name}...")
        self.model = SentenceTransformer(self.model_name, device=str(self.device))
        
        if device_cfg["use_gpu"] and device_cfg.get("torch_dtype") == "float16":
            self.model = self.model.half()
        
        # PhoBERT/Vi-models thường có max_seq_len là 256
        self.model.max_seq_length = self.max_seq_len

    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        # Normalize embeddings là bắt buộc cho Cosine Similarity
        emb = self.model.encode(texts, batch_size=batch_size, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
        return emb.astype("float32")