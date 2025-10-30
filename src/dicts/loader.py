# =========================
# file: src/dicts/loader.py
# =========================
from dataclasses import dataclass
from typing import Dict, Set

@dataclass
class Dicts:
    stop_phrases: Set[str]
    synonyms: Dict[str, str]
    abbreviations: Dict[str, str]
    gazetteer: Set[str]
    boilerplate: Set[str]

def _load_lines(p: str) -> Set[str]:
    try:
        return set([l.strip() for l in open(p, encoding="utf-8").read().splitlines() if l.strip()])
    except Exception:
        return set()

def _load_tsv_map(p: str) -> Dict[str, str]:
    m = {}
    try:
        for l in open(p, encoding="utf-8"):
            l = l.strip()
            if not l: continue
            a, b = l.split("\t")[:2]
            m[a.strip().lower()] = b.strip()
    except Exception:
        pass
    return m

def load_dictionaries(paths) -> Dicts:
    return Dicts(
        stop_phrases=_load_lines(paths.get("stop_phrases", "")),
        synonyms=_load_tsv_map(paths.get("synonyms", "")),
        abbreviations=_load_tsv_map(paths.get("abbreviations", "")),
        gazetteer=_load_lines(paths.get("gazetteer", "")),
        boilerplate=_load_lines(paths.get("boilerplate", "")),
    )

