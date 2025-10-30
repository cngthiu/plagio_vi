# =========================
# file: src/utils/hardware.py
# =========================
import os

def set_num_threads(omp: int = 8, mkl: int = 8):
    os.environ["OMP_NUM_THREADS"] = str(omp)
    os.environ["MKL_NUM_THREADS"] = str(mkl)

def select_device_config(hw: dict):
    return {
        "use_gpu": hw["device"]["use_gpu"],
        "torch_dtype": hw["device"].get("torch_dtype","float32")
    }