# src/core/catalog_cache.py
from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.core.settings import get_settings

# Try both core/services builder locations so you don't have to refactor imports
try:
    from src.core.catalog_builder import build_catalog  # must return dict
except Exception:  # pragma: no cover
    from src.services.catalog_builder import build_catalog  # type: ignore

# Engine to introspect live DB when (re)building
try:
    from src.core.db import get_engine
except Exception:  # pragma: no cover
    from src.app.db import get_engine  # type: ignore

_settings = get_settings()

# In-memory cache: (timestamp, catalog_dict)
_CACHE: Optional[Tuple[float, Dict[str, Any]]] = None
_LOCK = threading.RLock()
_BUILDING = False  # prevent thundering herd


def _now() -> float:
    return time.time()


def _ttl(default_secs: int = 1200) -> int:
    """TTL for catalog cache (seconds). Enforce sensible minimum."""
    try:
        val = int(getattr(_settings, "CATALOG_TTL_SECS", default_secs) or default_secs)
        return max(30, val)
    except Exception:
        return default_secs


def _catalog_path() -> Path:
    # Prefer resolved path helpers if available on settings, else fall back to CATALOG_PATH
    resolved = getattr(_settings, "CATALOG_PATH_RESOLVED", None)
    if resolved:
        return Path(str(resolved)).expanduser().resolve()
    return Path(getattr(_settings, "CATALOG_PATH", "./data/catalog.json")).expanduser().resolve()


def _read_json(p: Path) -> Optional[Dict[str, Any]]:
    if not p.exists() or p.stat().st_size == 0:
        return None
    with p.open("r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return None


def _write_json(p: Path, data: Dict[str, Any]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp.replace(p)


def _is_stale(ts: float, ttl: int) -> bool:
    return (_now() - ts) > ttl


def _build_and_store() -> Dict[str, Any]:
    """
    Build catalog from live DB and persist to disk.
    This is the ONLY place we call the builder + write file.
    """
    eng = get_engine()
    data = build_catalog(eng)  # <- must return a dict: {"tables": {...}} or similar
    if not isinstance(data, dict) or not data:
        raise RuntimeError("Catalog builder returned empty or non-dict data.")
    _write_json(_catalog_path(), data)
    return data


def get_catalog(
    *,
    force: bool = False,
    allow_build: bool = True,
    ttl_secs: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Return the catalog dict from memory; reload from disk if force/expired.
    If file is missing/invalid and allow_build=True, auto-build from DB and save.

    Raises:
        FileNotFoundError if missing and allow_build=False.
        RuntimeError if builder failed.
    """
    global _CACHE, _BUILDING
    ttl = int(ttl_secs if ttl_secs is not None else _ttl())

    with _LOCK:
        # serve from memory if fresh
        if not force and _CACHE is not None and not _is_stale(_CACHE[0], ttl):
            return _CACHE[1]

        # try disk
        disk = _read_json(_catalog_path())
        if disk and not force:
            _CACHE = (_now(), disk)
            return disk

        # disk missing/invalid or force requested
        if not allow_build:
            # behave like old code: surface FNFE for callers that want to handle rebuild
            raise FileNotFoundError(str(_catalog_path()))

        # avoid concurrent rebuilds
        if _BUILDING:
            # Wait briefly for the other thread to finish (simple backoff)
            for _ in range(50):
                time.sleep(0.05)
                if _CACHE is not None:
                    return _CACHE[1]
            # if still no cache, fall through and try to build anyway

        _BUILDING = True
        try:
            fresh = _build_and_store()
            _CACHE = (_now(), fresh)
            return fresh
        finally:
            _BUILDING = False


def refresh_catalog(*, force_rebuild: bool = False) -> Dict[str, Any]:
    """
    Reload from disk; if force_rebuild=True or file missing/invalid, rebuild from DB.
    """
    if force_rebuild:
        fresh = _build_and_store()
        with _LOCK:
            global _CACHE
            _CACHE = (_now(), fresh)
        return fresh

    try:
        return get_catalog(force=True, allow_build=False)
    except FileNotFoundError:
        return get_catalog(force=True, allow_build=True)


def set_catalog(catalog: Dict[str, Any], *, persist: bool = False) -> Dict[str, Any]:
    """
    Inject a catalog dict into the in-memory cache.
    If persist=True, also write it to disk (overwriting existing).
    """
    global _CACHE
    with _LOCK:
        if persist:
            _write_json(_catalog_path(), catalog)
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
    Return list of table names like 'dbo.Banks' (auto-builds if missing).
    """
    cat = get_catalog()
    # Accept both shapes: {"tables": {...}} or flat {"<table>": {...}}
    tables = cat.get("tables") if isinstance(cat.get("tables"), dict) else cat
    return sorted((tables or {}).keys())


def get_table_info(table: str) -> Optional[Dict[str, Any]]:
    """
    Return table info dict for a given table name (case-insensitive), or None.
    """
    if not table:
        return None
    cat = get_catalog()
    tables: Dict[str, Any] = cat.get("tables") if isinstance(cat.get("tables"), dict) else cat
    tables = tables or {}

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

    names = list_tables()
    scored: List[Tuple[int, str]] = []
    for full in names:
        base = full.split(".", 1)[-1].lower()
        score = 0
        # table name hit
        if base in toks:
            score += 3
        # column hits
        info = get_table_info(full) or {}
        cols = (info.get("columns") or {})
        # accept either list[{"name":..}] or dict{name: {...}}
        if isinstance(cols, dict):
            col_names = cols.keys()
        else:
            col_names = [c.get("name") for c in cols if isinstance(c, dict)]
        score += sum(1 for c in col_names if isinstance(c, str) and c.lower() in toks)
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
        tables = list_tables()
        print(f"Loaded catalog with {len(tables)} tables.")
        print("Sample tables:", tables[:10])
        print("TTL remaining (s):", round(ttl_remaining(), 1))
    except Exception as e:
        print(f"Catalog load/build failed at: {_catalog_path()}")
        print(f"Error: {type(e).__name__}: {e}")
        print("Tip: ensure DATABASE_URL is set and reachable.")
