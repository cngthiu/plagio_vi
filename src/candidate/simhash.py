# =========================
# file: src/candidate/simhash.py
# =========================
from typing import List
import hashlib
import math

def char_ngrams(s: str, n: int = 7) -> List[str]:
    s = f" {s} "
    return [s[i:i+n] for i in range(max(0, len(s)-n+1))]

def _hash64(x: str) -> int:
    return int(hashlib.md5(x.encode("utf-8")).hexdigest()[:16], 16)

def simhash(s: str, n: int = 7, bits: int = 64) -> int:
    v = [0]*bits
    for g in char_ngrams(s, n=n):
        h = _hash64(g)
        for i in range(bits):
            v[i] += 1 if (h >> i) & 1 else -1
    out = 0
    for i in range(bits):
        if v[i] >= 0:
            out |= (1 << i)
    return out

def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")

class SimHashIndex:
    def __init__(self, hashes: List[int]):
        self.hashes = hashes

    def topk_by_hamming(self, h: int, k: int = 50) -> List[int]:
        dists = [(i, hamming(h, x)) for i, x in enumerate(self.hashes)]
        dists.sort(key=lambda x: x[1])
        return [i for i, _d in dists[:k]]
