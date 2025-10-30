# =========================
# file: src/ranking/rerank.py
# =========================
def combine_scores(s_cross: float, s_bi: float, s_lex: float, s_bm25: float,
                   alpha: float, beta: float, gamma: float, delta: float) -> float:
    return alpha*s_cross + beta*s_bi + gamma*s_lex + delta*s_bm25
