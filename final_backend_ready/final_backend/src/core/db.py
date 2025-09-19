# src/core/db.py
from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.pool import QueuePool

from src.core.settings import get_settings

_settings = get_settings()

# ---------------------------
# Engine (singleton/lazy init)
# ---------------------------
_ENGINE: Engine | None = None


def get_engine() -> Engine:
    """Return a lazily-created SQLAlchemy Engine for MSSQL (pyodbc)."""
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = create_engine(
            _settings.DATABASE_URL,
            poolclass=QueuePool,
            pool_pre_ping=True,
            pool_size=_settings.POOL_SIZE,
            max_overflow=_settings.POOL_MAX_OVERFLOW,
            pool_recycle=_settings.POOL_RECYCLE_SECS,
            echo=_settings.SQL_ECHO,
            connect_args={
                # pyodbc login/connect timeout (seconds)
                "timeout": _settings.DB_CONNECT_TIMEOUT
            },
            future=True,
        )
    return _ENGINE


# ---------------------------
# SQL Safety Helpers
# ---------------------------

# Build a blocklist regex from settings (INSERT|UPDATE|... etc.)
_BLOCK_RE = re.compile(
    r"(?is)\b(" + r"|".join(re.escape(k) for k in _settings.SQL_BLOCKLIST) + r")\b"
)

# Simple pattern to capture schema-qualified refs: dbo.Table / [dbo].[Table]
_SCHEMA_TBL_RE = re.compile(
    r"(?is)(?:\[(?P<sch1>[^\]]+)\]|\b(?P<sch2>[A-Za-z_][\w]*)\b)\s*\.\s*(?:\[(?P<tb1>[^\]]+)\]|\b(?P<tb2>[A-Za-z_][\w]*)\b)"
)


def _ensure_single_statement(sql: str) -> str:
    # Reject multiple statements separated by ';' (basic guard)
    parts = [p.strip() for p in sql.strip().split(";") if p.strip()]
    if len(parts) > 1:
        raise ValueError("Only a single SELECT statement is allowed.")
    return parts[0]


def _ensure_select_only(sql: str) -> None:
    if not re.match(r"(?is)^\s*select\b", sql):
        raise ValueError("Only SELECT statements are allowed.")


def _enforce_blocklist(sql: str) -> None:
    if _BLOCK_RE.search(sql):
        raise ValueError("Disallowed SQL keyword detected.")


def _enforce_schema_allowlist(sql: str) -> None:
    """If ALLOW_SCHEMAS is set, ensure all schema refs are within allowlist."""
    allow = {s.lower() for s in _settings.ALLOW_SCHEMAS}
    if not allow:
        return
    for m in _SCHEMA_TBL_RE.finditer(sql):
        sch = (m.group("sch1") or m.group("sch2") or "").strip().strip("[]").lower()
        if sch and sch not in allow:
            raise ValueError(f"Disallowed schema referenced: {sch}")


def _inject_top_limit(sql: str, limit: int) -> str:
    """
    If no TOP/FETCH/LIMIT present, inject MSSQL TOP at the first SELECT.
    Safe for simple queries. For very complex SELECTs, prefer the model to include TOP.
    """
    if re.search(r"(?is)\btop\s+\d+\b", sql) or re.search(r"(?is)\bfetch\s+next\s+\d+\s+rows\b", sql):
        return sql
    # don't inject TOP inside subqueries here (best-effort). We only modify the first SELECT.
    return re.sub(r"(?is)^\s*select\s+", f"SELECT TOP {limit} ", sql, count=1)


def _prepare_sql(sql: str, row_limit: Optional[int]) -> str:
    s = _ensure_single_statement(sql)
    _ensure_select_only(s)
    _enforce_blocklist(s)
    _enforce_schema_allowlist(s)
    if row_limit and row_limit > 0:
        s = _inject_top_limit(s, row_limit)
    return s.strip() + ";"


# ---------------------------
# Public API
# ---------------------------

def run_select(
    sql: str,
    params: Optional[Dict[str, Any]] = None,
    *,
    timeout_secs: Optional[int] = None,
    row_limit: Optional[int] = None,
    enforce_readonly: Optional[bool] = None,
) -> List[Dict[str, Any]]:
    """
    Execute a guarded SELECT query and return list of dict rows.

    Args:
        sql: SQL text (may be produced by Text-to-SQL).
        params: Optional bound parameters for SQLAlchemy text().
        timeout_secs: Per-statement timeout (pyodbc). Defaults to STMT_TIMEOUT_SECS.
        row_limit: If provided, injects TOP {row_limit} when missing.
        enforce_readonly: If True, force SELECT-only (default = settings.READONLY_ENFORCED).
    """
    if enforce_readonly is None:
        enforce_readonly = _settings.READONLY_ENFORCED

    # Safety prep
    prepared_sql = _prepare_sql(sql, row_limit or _settings.SQL_ROW_LIMIT if enforce_readonly else None)

    eng = get_engine()
    with eng.connect() as conn:
        # Set per-statement timeout on raw pyodbc connection (seconds)
        try:
            raw = conn.connection  # type: ignore[attr-defined]
            raw.timeout = int(timeout_secs or _settings.STMT_TIMEOUT_SECS)
        except Exception:
            # Not fatal if driver doesn't expose .timeout
            pass

        result = conn.execute(text(prepared_sql), params or {})
        # Use .mappings() to get dict-like rows
        rows = result.mappings().all()
        return [dict(r) for r in rows]


def test_connection() -> Dict[str, Any]:
    """
    Quick connectivity probe returning @@VERSION and current DB name.
    """
    eng = get_engine()
    with eng.connect() as conn:
        try:
            ver = conn.exec_driver_sql("SELECT @@VERSION AS v").mappings().first()
            dbn = conn.exec_driver_sql("SELECT DB_NAME() AS db").mappings().first()
            return {
                "ok": True,
                "version": (ver or {}).get("v"),
                "database": (dbn or {}).get("db"),
            }
        except Exception as e:
            return {"ok": False, "error": f"{type(e).__name__}: {e}"}
