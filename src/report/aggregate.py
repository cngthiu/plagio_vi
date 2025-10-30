# =========================
# file: src/report/aggregate.py
# =========================
from typing import List, Dict, Any

def aggregate_stats(results: List[Dict[str, Any]], nA: int, nB: int, thresholds: dict) -> Dict[str, Any]:
    levels = {"very_high":0, "high":0, "medium":0, "low":0}
    for r in results:
        s = r["score"]
        if s >= thresholds.get("very_high", 0.87): levels["very_high"] += 1
        elif s >= thresholds.get("high", 0.82): levels["high"] += 1
        elif s >= thresholds.get("medium", 0.72): levels["medium"] += 1
        elif s >= thresholds.get("low", 0.62): levels["low"] += 1
    summary = {
        "chunks_A": nA, "chunks_B": nB, "pairs": len(results),
        "levels": levels
    }
    return {"summary": summary}

