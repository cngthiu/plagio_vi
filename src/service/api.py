# =========================
# file: src/service/api.py
# =========================
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

from functools import lru_cache
import yaml

from src.service.runner import run_compare
from src.models.embedder import BiEncoder
from src.models.cross_encoder import CrossEncoder
from src.utils.hardware import select_device_config

app = FastAPI(title="plagio_vi API", version="1.0.0")


def _load_yaml(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@lru_cache(maxsize=4)
def get_model_bundle(config_path: str, hardware_path: str) -> tuple[BiEncoder, CrossEncoder]:
    """
    Load/cached models để tránh tải lại mỗi request.
    Cache key: (config_path, hardware_path).
    """
    cfg = _load_yaml(config_path)
    hw = _load_yaml(hardware_path)
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
