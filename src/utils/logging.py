# =========================
# file: src/utils/logging.py
# =========================
import json
from pathlib import Path
from typing import Dict, Any

class JsonLogger:
    def __init__(self, path: Path):
        self.path = path
        self.f = open(self.path, "a", encoding="utf-8")

    def log(self, obj: Dict[str, Any]):
        self.f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        self.f.flush()

    def __del__(self):
        try:
            self.f.close()
        except Exception:
            pass