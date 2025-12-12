# =========================
# file: src/service/api.py
# =========================
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from functools import lru_cache

from src.utils.common import load_yaml  # Dùng hàm chung
from src.service.runner import run_compare
from src.models.embedder import BiEncoder
from src.models.cross_encoder import CrossEncoder
from src.utils.hardware import select_device_config

app = FastAPI(title="plagio_vi API", version="1.1.0")  # Bump version

@lru_cache(maxsize=4)
def get_model_bundle(config_path: str, hardware_path: str) -> tuple[BiEncoder, CrossEncoder]:
    """
    Load/cached models để tránh tải lại mỗi request.
    Cache key: (config_path, hardware_path).
    """
    # Thay thế _load_yaml cũ bằng load_yaml mới
    cfg = load_yaml(config_path)
    hw = load_yaml(hardware_path)
    
    device_cfg = select_device_config(hw)
    bi = BiEncoder(
        device_cfg=device_cfg,
        max_seq_len=cfg["chunking"]["max_seq_len_bi"],
        backend=hw["inference"]["bi_encoder"]["backend"],
        quantize=hw["inference"]["bi_encoder"]["quantize"]
    )
    ce = CrossEncoder(
        device_cfg=device_cfg,
        max_seq_len=cfg["chunking"]["max_seq_len_cross"],
        backend=hw["inference"]["cross_encoder"]["backend"],
        quantize=hw["inference"]["cross_encoder"]["quantize"]
    )
    return bi, ce

class CompareReq(BaseModel):
    a: str
    b: str
    out: str
    config: Optional[str] = "configs/default.yaml"
    hardware: Optional[str] = "configs/hardware.gpu1650.yaml"

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/compare")
def compare(req: CompareReq):
    # API này vẫn chạy đồng bộ (blocking), nhưng code đã gọn hơn
    bi, ce = get_model_bundle(req.config, req.hardware)
    res = run_compare(
        config_path=req.config,
        hardware_path=req.hardware,
        path_a=req.a,
        path_b=req.b,
        out_dir=req.out,
        bi_encoder=bi,
        cross_encoder=ce,
    )
    return res