# src/utils/common.py
import yaml
import numpy as np

def load_yaml(path: str):
    """Đọc file YAML an toàn."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def minmax_scale(a: np.ndarray) -> np.ndarray:
    """Chuẩn hóa vector về [0, 1]."""
    if a.size == 0:
        return a
    mn, mx = float(np.min(a)), float(np.max(a))
    if mx - mn < 1e-9:
        return np.zeros_like(a, dtype=np.float32)
    return ((a - mn) / (mx - mn)).astype(np.float32)

def html_escape(s: str) -> str:
    """Escape ký tự đặc biệt cho HTML."""
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")