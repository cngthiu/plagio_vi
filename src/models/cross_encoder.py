# =========================
# file: src/models/cross_encoder.py
# =========================
from typing import List, Tuple, Optional
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

try:
    import onnxruntime as ort
except Exception:
    ort = None

class CrossEncoder:
    def __init__(self, device_cfg: dict, max_seq_len: int = 384, backend: str = "torch", quantize: str = "fp16", model_name: Optional[str] = None):
        self.device_cfg = device_cfg
        self.device = torch.device("cuda") if device_cfg["use_gpu"] else torch.device("cpu")
        self.max_seq_len = max_seq_len
        self.backend = backend or "torch"
        self.quantize = quantize
        candidates = [model_name] if model_name else [
            "bkai-foundation-models/vietnamese-cross-encoder",
            "cross-encoder/ms-marco-MiniLM-L-12-v2",
        ]
        last_err = None
        self.model = None
        self.tokenizer = None
        self.ort_session = None

        if self.backend == "onnx":
            if ort is None:
                raise RuntimeError("onnxruntime not installed for ONNX backend")
            for name in candidates:
                if not name:
                    continue
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(name)
                    providers = ["CUDAExecutionProvider","CPUExecutionProvider"] if device_cfg["use_gpu"] else ["CPUExecutionProvider"]
                    self.ort_session = ort.InferenceSession(name, providers=providers)
                    self.model_name = name
                    break
                except Exception as e:
                    last_err = e
                    continue
            if self.ort_session is None or self.tokenizer is None:
                raise RuntimeError(f"Could not load ONNX cross-encoder from {candidates}") from last_err
        else:
            for name in candidates:
                if not name:
                    continue
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(name)
                    self.model = AutoModelForSequenceClassification.from_pretrained(name).to(self.device)
                    self.model_name = name
                    break
                except Exception as e:
                    last_err = e
                    continue
            if self.model is None or self.tokenizer is None:
                raise RuntimeError(f"Could not load torch cross-encoder from candidates: {candidates}") from last_err
            if device_cfg["use_gpu"] and device_cfg.get("torch_dtype") == "float16":
                self.model.half()
            self.model.eval()

    @torch.no_grad()
    def score_pairs(self, pairs: List[Tuple[str, str]], batch_size: int = 8) -> List[float]:
        scores = []
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i:i+batch_size]
            toks = self.tokenizer([p[0] for p in batch], [p[1] for p in batch],
                                  truncation=True, padding=True, max_length=self.max_seq_len, return_tensors="pt")
            if self.backend == "onnx":
                if self.ort_session is None:
                    raise RuntimeError("ONNX session not initialized")
                inputs = {k: v.cpu().numpy() for k, v in toks.items()}
                logits = self.ort_session.run(None, inputs)[0]
                s = logits.reshape(-1).tolist()
            else:
                toks = toks.to(self.device)
                out = self.model(**toks).logits.squeeze(-1)
                s = out.detach().float().cpu().tolist()
            if isinstance(s, float): s = [s]
            scores.extend(s)
        # map logits to (0,1) via sigmoid for consistency
        import math
        return [1/(1+math.exp(-x)) for x in scores]
