# =========================
# file: src/utils/cache.py
# =========================
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np


class DiskCache:
    """
    Cache đơn giản dựa trên file trong outputs/cache/.
    - Namespace tạo thư mục con (vd: "embeddings", "simhash").
    - Key là md5 của payload (vd: text, list text).
    """
    def __init__(self, base_dir: str = "outputs/cache"):
        self.base = Path(base_dir)
        self.base.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _md5_bytes(data: bytes) -> str:
        return hashlib.md5(data).hexdigest()

    @staticmethod
    def _md5_text(s: str) -> str:
        return hashlib.md5(s.encode("utf-8")).hexdigest()

    @staticmethod
    def _md5_list_str(xs: Iterable[str]) -> str:
        h = hashlib.md5()
        for s in xs:
            h.update(b"\x1e")  # sep
            h.update(s.encode("utf-8"))
        return h.hexdigest()

    def _path(self, namespace: str, key: str, ext: str) -> Path:
        d = self.base / namespace
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{key}.{ext}"

    # ----- numpy -----
    def get_numpy(self, namespace: str, key: str) -> Optional[np.ndarray]:
        p = self._path(namespace, key, "npy")
        if not p.exists():
            return None
        try:
            return np.load(p, allow_pickle=False)
        except Exception:
            return None

    def set_numpy(self, namespace: str, key: str, arr: np.ndarray):
        p = self._path(namespace, key, "npy")
        np.save(p, arr)

    # ----- json -----
    def get_json(self, namespace: str, key: str) -> Optional[Any]:
        p = self._path(namespace, key, "json")
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None

    def set_json(self, namespace: str, key: str, obj: Any):
        p = self._path(namespace, key, "json")
        p.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")

    # ----- helpers for model/text -----
    def key_for_texts(self, texts: List[str], model_name: str) -> str:
        payload = f"model={model_name}||{self._md5_list_str(texts)}"
        return self._md5_text(payload)

    def key_for_strings(self, texts: List[str]) -> str:
        return self._md5_list_str(texts)