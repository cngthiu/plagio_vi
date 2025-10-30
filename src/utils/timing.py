# =========================
# file: src/utils/timing.py
# =========================
import time
from contextlib import contextmanager

class Timer:
    def __init__(self, logger=None):
        self.logger = logger

    @contextmanager
    def section(self, name: str):
        t0 = time.time()
        try:
            yield
        finally:
            dt = time.time() - t0
            if self.logger:
                self.logger.log({"event": "timing", "section": name, "seconds": round(dt, 4)})