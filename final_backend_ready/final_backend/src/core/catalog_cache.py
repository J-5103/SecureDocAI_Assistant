# src/core/catalog_cache.py
from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.core.settings import get_settings

_settings = get_settings()

# In-memory cache: (timestamp, catalog_dict)
_CACHE: Optional[Tuple[float, Dict[str, Any]]] = None
_LOCK = threading.RLock()


def _now() -> float:
    return time.time()


def _ttl() -> int:
    """TTL for catalog cache (seconds)."""
    try:
        return max(30, int(_settings.CATALOG_TTL_SECS or 1200))
    except Exception:
        return 1200


def _catalog_path() -> Path:
    return Path(_settings.CATALOG_PATH).expanduser().resolve()


def _load_from_disk() -> Dict[str, Any]:
    """Load catalog JSON from disk; raise if missing/invalid."""
    p = _catalog_path()
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def _save_to_disk(catalog: Dict[str, Any]) -> Path:
    """Save catalog JSON to disk atomically (write temp then replace)."""
    p = _catalog_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(catalog, f, ensure_ascii=False, indent=2)
    tmp.replace(p)
    return p


def _is_stale(ts: float) -> bool:
    return (_now() - ts) > _ttl()


def get_catalog(*, force: bool = False) -> Dict[str, Any]:
    """
    Return the catalog dict from memory, reloading from disk when:
      - force=True, or
      - cache is empty, or
      - cache TTL expired.

    Raises:
        FileNotFoundError if the catalog file doesn't exist yet.
        json.JSONDecodeError for malformed files.
    """
    global _CACHE
    with _LOCK:
        if not force and _CACHE is not None and not _is_stale(_CACHE[0]):
            return _CACHE[1]

        data = _load_from_disk()  # may raise (good -> caller can decide to rebuild)
        _CACHE = (_now(), data)
        return data


def refresh_catalog() -> Dict[str, Any]:
    """
    Force refresh from disk and return fresh catalog.
    """
    return get_catalog(force=True)


def set_catalog(catalog: Dict[str, Any], *, persist: bool = False) -> Dict[str, Any]:
    """
    Inject a catalog dict into the in-memory cache.
    If persist=True, also write it to disk (overwriting existing).
    """
    global _CACHE
    with _LOCK:
        if persist:
            _save_to_disk(catalog)
        _CACHE = (_now(), catalog)
        return catalog


def ttl_remaining() -> float:
    """
    Seconds until the cached catalog will be considered stale.
    Returns 0 if no cache is present.
    """
    with _LOCK:
        if _CACHE is None:
            return 0.0
        age = _now() - _CACHE[0]
        return max(0.0, float(_ttl()) - age)


# ---------------------------
# Convenience lookup helpers
# ---------------------------

def list_tables() -> List[str]:
    """
    Return list of table names like 'dbo.Banks' from the cached catalog.
    """
    cat = get_catalog()
    return sorted((cat.get("tables") or {}).keys())


def get_table_info(table: str) -> Optional[Dict[str, Any]]:
    """
    Return table info dict for a given table name (case-insensitive), or None.
    """
    if not table:
        return None
    cat = get_catalog()
    tables: Dict[str, Any] = cat.get("tables") or {}

    # direct hit
    if table in tables:
        return tables[table]

    # case-insensitive / unqualified fallback
    t_lower = table.lower()
    for k, v in tables.items():
        if k.lower() == t_lower:
            return v
        # allow unqualified lookup: 'Banks' matches 'dbo.Banks'
        if "." in k and k.split(".", 1)[1].lower() == t_lower:
            return v
    return None


def tables_used_in_question_hint(question: str, top_k: int = 6) -> List[str]:
    """
    Very light lexical matcher: ranks tables by simple keyword overlap.
    (Good as a fallback if you don't have embeddings yet.)
    """
    q = (question or "").lower()
    toks = {t for t in q.replace(",", " ").replace(".", " ").split() if t}
    if not toks:
        return []

    tables = list_tables()
    scored: List[Tuple[int, str]] = []
    for full in tables:
        base = full.split(".", 1)[-1].lower()
        score = 0
        # table name hit
        if base in toks:
            score += 3
        # column hits
        info = get_table_info(full) or {}
        cols = (info.get("columns") or {}).keys()
        score += sum(1 for c in cols if c.lower() in toks)
        if score > 0:
            scored.append((score, full))
    scored.sort(key=lambda x: (-x[0], x[1]))
    return [t for _, t in scored[: max(1, top_k)]]


# ---------------------------
# CLI: quick peek
# ---------------------------
if __name__ == "__main__":
    try:
        cat = get_catalog()
        print(f"Loaded catalog with {len(cat.get('tables', {}))} tables.")
        print("Sample tables:", list_tables()[:10])
        print("TTL remaining (s):", round(ttl_remaining(), 1))
    except FileNotFoundError:
        print(f"Catalog not found at: {_catalog_path()}")
        print("Run: python -m src.core.catalog_builder")
