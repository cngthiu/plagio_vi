# =========================
# file: src/dicts/loader.py
# =========================
from dataclasses import dataclass
from typing import Dict, Set
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class Dicts:
    stop_phrases: Set[str]
    synonyms: Dict[str, str]
    abbreviations: Dict[str, str]
    gazetteer: Set[str]
    boilerplate: Set[str]

def _load_lines(p: str, optional: bool = False) -> Set[str]:
    """Load danh sách dòng. Raise error nếu file không tồn tại (trừ khi optional=True)."""
    if not p:
        return set()
    path_obj = Path(p)
    
    if not path_obj.exists():
        if optional:
            logger.warning(f"Optional file not found: {p}")
            return set()
        # [UPDATE] Fail fast
        raise FileNotFoundError(f"Critical dictionary file not found: {p}")
        
    content = path_obj.read_text(encoding="utf-8")
    return set([l.strip() for l in content.splitlines() if l.strip()])

def _load_tsv_map(p: str, optional: bool = False) -> Dict[str, str]:
    if not p:
        return {}
    path_obj = Path(p)
    
    if not path_obj.exists():
        if optional:
            logger.warning(f"Optional file not found: {p}")
            return {}
        # [UPDATE] Fail fast
        raise FileNotFoundError(f"Critical dictionary file not found: {p}")

    m = {}
    with path_obj.open("r", encoding="utf-8") as f:
        for line_no, l in enumerate(f, 1):
            l = l.strip()
            if not l: continue
            parts = l.split("\t")
            if len(parts) >= 2:
                a, b = parts[:2]
                m[a.strip().lower()] = b.strip()
            else:
                logger.warning(f"Malformed line {line_no} in {p}: {l}")
    return m

def load_dictionaries(paths: Dict[str, str]) -> Dicts:
    logger.info("Loading dictionaries...")
    # Stop phrases và boilerplate là quan trọng -> không optional
    return Dicts(
        stop_phrases=_load_lines(paths.get("stop_phrases", ""), optional=False),
        synonyms=_load_tsv_map(paths.get("synonyms", ""), optional=True),
        abbreviations=_load_tsv_map(paths.get("abbreviations", ""), optional=True),
        gazetteer=_load_lines(paths.get("gazetteer", ""), optional=True),
        boilerplate=_load_lines(paths.get("boilerplate", ""), optional=False),
    )