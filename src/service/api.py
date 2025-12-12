# =========================
# file: src/service/api.py
# =========================
import logging
from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel

from src.utils.common import load_yaml
from src.service.runner import run_compare
from src.models.embedder import BiEncoder
from src.models.cross_encoder import CrossEncoder
from src.utils.hardware import select_device_config

# Setup logger
logger = logging.getLogger("api")
logger.setLevel(logging.INFO)

app = FastAPI(title="plagio_vi API", version="1.0.0-simple")

# --- Singleton Model Loader ---
# Vẫn giữ Class này để tránh việc load lại model gây tốn RAM/thời gian
# mỗi khi bạn F5 hoặc gọi API lần 2.
class ModelManager:
    _bi: Optional[BiEncoder] = None
    _ce: Optional[CrossEncoder] = None
    
    @classmethod
    def get_models(cls, config_path: str, hardware_path: str):
        if cls._bi is None or cls._ce is None:
            logger.info("Loading models... (This happens only once)")
            cfg = load_yaml(config_path)
            hw = load_yaml(hardware_path)
            device_cfg = select_device_config(hw)
            
            cls._bi = BiEncoder(
                device_cfg=device_cfg,
                max_seq_len=cfg["chunking"]["max_seq_len_bi"],
                backend=hw["inference"]["bi_encoder"]["backend"],
                quantize=hw["inference"]["bi_encoder"]["quantize"]
            )
            cls._ce = CrossEncoder(
                device_cfg=device_cfg,
                max_seq_len=cfg["chunking"]["max_seq_len_cross"],
                backend=hw["inference"]["cross_encoder"]["backend"],
                quantize=hw["inference"]["cross_encoder"]["quantize"]
            )
        return cls._bi, cls._ce

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
    """
    API chạy đồng bộ (Blocking).
    Client sẽ phải treo kết nối chờ cho đến khi xử lý xong.
    """
    logger.info(f"Processing comparison request for: {req.a} vs {req.b}")
    
    # 1. Lấy model từ Singleton (Nhanh nếu đã load trước đó)
    bi, ce = ModelManager.get_models(req.config, req.hardware)
    
    # 2. Chạy so sánh trực tiếp (Chặn luồng cho đến khi xong)
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