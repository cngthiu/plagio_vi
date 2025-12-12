# =========================
# file: src/dicts/loader.py
# =========================
from dataclasses import dataclass
from typing import Dict, Set
import logging
from pathlib import Path

# Setup standard logger cho module này
logger = logging.getLogger(__name__)

@dataclass
class Dicts:
    stop_phrases: Set[str]
    synonyms: Dict[str, str]
    abbreviations: Dict[str, str]
    gazetteer: Set[str]
    boilerplate: Set[str]

def _load_lines(p: str) -> Set[str]:
    """Load danh sách dòng từ file text, bỏ qua dòng trống."""
    path_obj = Path(p)
    if not p or not path_obj.exists():
        if p:
            logger.warning(f"File not found: {p}. Returning empty set.")
        return set()
        
    try:
        # Dùng read_text để tự động handle đóng file
        content = path_obj.read_text(encoding="utf-8")
        return set([l.strip() for l in content.splitlines() if l.strip()])
    except Exception as e:
        logger.error(f"Failed to load lines from {p}: {e}")
        return set()

def _load_tsv_map(p: str) -> Dict[str, str]:
    """Load dictionary từ file TSV (tab-separated)."""
    path_obj = Path(p)
    if not p or not path_obj.exists():
        if p:
            logger.warning(f"File not found: {p}. Returning empty dict.")
        return {}

    m = {}
    try:
        with path_obj.open("r", encoding="utf-8") as f:
            for line_no, l in enumerate(f, 1):
                l = l.strip()
                if not l: continue
                parts = l.split("\t")
                if len(parts) >= 2:
                    a, b = parts[:2]
                    m[a.strip().lower()] = b.strip()
                else:
                    logger.debug(f"Skipping malformed line {line_no} in {p}: {l}")
    except Exception as e:
        logger.error(f"Failed to load TSV map from {p}: {e}")
    return m

def load_dictionaries(paths: Dict[str, str]) -> Dicts:
    logger.info("Loading dictionaries...")
    return Dicts(
        stop_phrases=_load_lines(paths.get("stop_phrases", "")),
        synonyms=_load_tsv_map(paths.get("synonyms", "")),
        abbreviations=_load_tsv_map(paths.get("abbreviations", "")),
        gazetteer=_load_lines(paths.get("gazetteer", "")),
        boilerplate=_load_lines(paths.get("boilerplate", "")),
    )