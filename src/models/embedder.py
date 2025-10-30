# =========================
# file: src/models/embedder.py
# =========================
from typing import List, Optional
import numpy as np
import torch

class BiEncoder:
    def __init__(self, device_cfg: dict, max_seq_len: int = 256, backend: str = "torch", quantize: str = "fp16", model_name: Optional[str] = None):
        self.device = torch.device("cuda") if device_cfg["use_gpu"] else torch.device("cpu")
        self.max_seq_len = max_seq_len
        self.backend = backend
        self.quantize = quantize
        self.model_name = model_name or "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        # Why: tránh phụ thuộc nặng, chọn model nhỏ đa ngôn ngữ.
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(self.model_name, device=str(self.device))
        if device_cfg["use_gpu"] and device_cfg.get("torch_dtype") == "float16":
            self.model = self.model.half()

    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        emb = self.model.encode(texts, batch_size=batch_size, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
        return emb.astype("float32")
