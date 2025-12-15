# =========================
# file: src/models/cross_encoder.py
# =========================
from typing import List, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class CrossEncoder:
    def __init__(self, device_cfg: dict, max_seq_len: int = 512, backend: str = "torch", quantize: str = "fp16", model_name: str = None):
        self.device = torch.device("cuda") if device_cfg["use_gpu"] else torch.device("cpu")
        self.max_seq_len = max_seq_len
        
        # Model multilingual tốt cho tiếng Việt
        self.model_name = model_name or "amberoad/bert-multilingual-passage-reranking-msmarco"
        
        print(f"Loading Cross-Encoder: {self.model_name}...")
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
            
            out = self.model(**toks).logits
            
            # [FIX LỖI BROADCAST SHAPE]
            # Nếu model trả về [Batch, 2] (Binary Classification), ta lấy cột index 1 (lớp "relevant")
            if out.shape[-1] == 2:
                # Lấy logit của lớp Positive (Relevant)
                relevant_logits = out[:, 1]
                s = torch.sigmoid(relevant_logits).detach().float().cpu().tolist()
            else:
                # Trường hợp model trả về 1 output duy nhất
                s = torch.sigmoid(out).squeeze(-1).detach().float().cpu().tolist()
            
            if isinstance(s, float): s = [s]
            scores.extend(s)
            
        return scores