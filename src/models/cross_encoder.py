# =========================
# file: src/models/cross_encoder.py
# =========================
from typing import List, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class CrossEncoder:
    def __init__(self, device_cfg: dict, max_seq_len: int = 384, backend: str = "torch", quantize: str = "fp16", model_name: str = None):
        self.device = torch.device("cuda") if device_cfg["use_gpu"] else torch.device("cpu")
        self.max_seq_len = max_seq_len
        self.model_name = model_name or "cross-encoder/ms-marco-MiniLM-L-12-v2"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name).to(self.device)
        if device_cfg["use_gpu"] and device_cfg.get("torch_dtype") == "float16":
            self.model.half()
        self.model.eval()

    @torch.no_grad()
    def score_pairs(self, pairs: List[Tuple[str, str]], batch_size: int = 8) -> List[float]:
        scores = []
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i:i+batch_size]
            toks = self.tokenizer([p[0] for p in batch], [p[1] for p in batch],
                                  truncation=True, padding=True, max_length=self.max_seq_len, return_tensors="pt").to(self.device)
            out = self.model(**toks).logits.squeeze(-1)
            s = out.detach().float().cpu().tolist()
            if isinstance(s, float): s = [s]
            scores.extend(s)
        # map logits to (0,1) via sigmoid for consistency
        import math
        return [1/(1+math.exp(-x)) for x in scores]

