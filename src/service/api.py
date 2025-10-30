# =========================
# file: src/service/api.py
# =========================
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

from src.service.runner import run_compare

app = FastAPI(title="plagio_vi API", version="1.0.0")

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
    res = run_compare(config_path=req.config, hardware_path=req.hardware, path_a=req.a, path_b=req.b, out_dir=req.out)
    return res