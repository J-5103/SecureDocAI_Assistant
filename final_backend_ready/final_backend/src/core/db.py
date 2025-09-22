# src/core/db.py
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.pool import QueuePool

# NOTE: keep working even if some settings aren’t present
try:
    from src.core.settings import get_settings  # your existing settings loader
except Exception:  # pragma: no cover
    def get_settings():
        class _S:
            DATABASE_URL = "sqlite:///./local.db"
            POOL_SIZE = 5
            POOL_MAX_OVERFLOW = 10
            POOL_RECYCLE_SECS = 1800
            SQL_ECHO = False
            DB_CONNECT_TIMEOUT = 15
            STMT_TIMEOUT_SECS = 60
            SQL_ROW_LIMIT = 1000
            READONLY_ENFORCED = True
            SQL_BLOCKLIST = ["insert", "update", "delete", "merge", "alter", "drop", "truncate", "create", "grant", "revoke"]
            ALLOW_SCHEMAS: list[str] = []  # empty means no restriction
        return _S()

_settings = get_settings()

# ---------------------------
# Engine (singleton/lazy init)
# ---------------------------
_ENGINE: Engine | None = None


def _db_family(url: str) -> str:
    u = (url or "").lower()
    if u.startswith("mssql"):
        return "mssql"
    if u.startswith("postgresql") or u.startswith("postgres"):
        return "postgres"
    if u.startswith("sqlite"):
        return "sqlite"
    # fallback to dialect name later
    return "unknown"


def get_engine() -> Engine:
    """Return a lazily-created SQLAlchemy Engine for configured DB."""
    global _ENGINE
    if _ENGINE is None:
        db_url = getattr(_settings, "DATABASE_URL", None)
        if not db_url:
            raise RuntimeError("DATABASE_URL is not configured in settings.")

        fam = _db_family(db_url)

        connect_args: Dict[str, Any] = {}
        if fam == "mssql":
            # pyodbc login/connect timeout (seconds)
            connect_args["timeout"] = int(getattr(_settings, "DB_CONNECT_TIMEOUT", 15))
        elif fam == "sqlite":
            connect_args["check_same_thread"] = False  # safe for most FastAPI apps

        _ENGINE = create_engine(
            db_url,
            poolclass=QueuePool,
            pool_pre_ping=True,
            pool_size=int(getattr(_settings, "POOL_SIZE", 5)),
            max_overflow=int(getattr(_settings, "POOL_MAX_OVERFLOW", 10)),
            pool_recycle=int(getattr(_settings, "POOL_RECYCLE_SECS", 1800)),
            echo=bool(getattr(_settings, "SQL_ECHO", False)),
            connect_args=connect_args,
            future=True,
        )
    return _ENGINE


def _current_family() -> str:
    eng = get_engine()
    try:
        # dialect.name is reliable: "mssql" | "postgresql" | "sqlite"
        name = eng.dialect.name.lower()
        if "mssql" in name:
            return "mssql"
        if "postgres" in name:
            return "postgres"
        if "sqlite" in name:
            return "sqlite"
    except Exception:
        pass
    return _db_family(getattr(_settings, "DATABASE_URL", ""))


# ---------------------------
# SQL Safety Helpers
# ---------------------------

# Build a blocklist regex from settings (INSERT|UPDATE|... etc.)
_BLOCK_RE = re.compile(
    r"(?is)\b(" + r"|".join(re.escape(k) for k in getattr(_settings, "SQL_BLOCKLIST", [
        "insert", "update", "delete", "merge", "alter", "drop", "truncate", "create", "grant", "revoke"
    ])) + r")\b"
)

# Simple pattern to capture schema-qualified refs: dbo.Table / [dbo].[Table] / public.table
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
    allow = {s.lower() for s in getattr(_settings, "ALLOW_SCHEMAS", [])}
    if not allow:
        return
    for m in _SCHEMA_TBL_RE.finditer(sql):
        sch = (m.group("sch1") or m.group("sch2") or "").strip().strip("[]").lower()
        if sch and sch not in allow:
            raise ValueError(f"Disallowed schema referenced: {sch}")


def _has_limit_or_fetch(sql: str) -> bool:
    return bool(
        re.search(r"(?is)\blimit\s+\d+\b", sql)
        or re.search(r"(?is)\boffset\s+\d+\b", sql)
        or re.search(r"(?is)\bfetch\s+next\s+\d+\s+rows\b", sql)
        or re.search(r"(?is)\btop\s+\d+\b", sql)
    )


def _inject_limit(sql: str, limit: int, family: str) -> str:
    """
    Inject a row limit if none present:
      - MSSQL: inject TOP at first SELECT
      - Postgres/SQLite: append LIMIT at the end
    """
    if _has_limit_or_fetch(sql) or limit <= 0:
        return sql

    if family == "mssql":
        return re.sub(r"(?is)^\s*select\s+", f"SELECT TOP {limit} ", sql, count=1)

    # For postgres/sqlite
    s = sql.rstrip().rstrip(";")
    return f"{s} LIMIT {limit}"


def _prepare_sql(sql: str, row_limit: Optional[int], family: str) -> str:
    s = _ensure_single_statement(sql)
    _ensure_select_only(s)
    _enforce_blocklist(s)
    _enforce_schema_allowlist(s)
    if row_limit and row_limit > 0:
        s = _inject_limit(s, row_limit, family)
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
        timeout_secs: Per-statement timeout (driver-dependent). Defaults to STMT_TIMEOUT_SECS.
        row_limit: If provided, injects LIMIT/TOP when missing.
        enforce_readonly: If True, force SELECT-only (default = settings.READONLY_ENFORCED).
    """
    if enforce_readonly is None:
        enforce_readonly = bool(getattr(_settings, "READONLY_ENFORCED", True))

    family = _current_family()

    # Safety prep
    default_limit = int(getattr(_settings, "SQL_ROW_LIMIT", 1000)) if enforce_readonly else None
    prepared_sql = _prepare_sql(sql, row_limit or default_limit, family)

    eng = get_engine()
    with eng.connect() as conn:
        # Set per-statement timeout if driver supports it (pyodbc/mssql)
        try:
            raw = conn.connection  # type: ignore[attr-defined]
            raw.timeout = int(timeout_secs or getattr(_settings, "STMT_TIMEOUT_SECS", 60))
        except Exception:
            pass  # not all drivers support this

        result = conn.execute(text(prepared_sql), params or {})
        rows = result.mappings().all()
        return [dict(r) for r in rows]


def test_connection() -> Dict[str, Any]:
    """
    Cross-DB connectivity probe returning version and current DB name (if available).
    """
    eng = get_engine()
    fam = _current_family()
    with eng.connect() as conn:
        try:
            conn.execute(text("SELECT 1"))
            if fam == "mssql":
                ver = conn.exec_driver_sql("SELECT @@VERSION AS v").mappings().first()
                dbn = conn.exec_driver_sql("SELECT DB_NAME() AS db").mappings().first()
            elif fam == "postgres":
                ver = conn.execute(text("SELECT version() AS v")).mappings().first()
                dbn = conn.execute(text("SELECT current_database() AS db")).mappings().first()
            elif fam == "sqlite":
                ver = conn.execute(text("SELECT sqlite_version() AS v")).mappings().first()
                dbn = {"db": None}
            else:
                ver = {"v": str(eng.dialect.name)}
                dbn = {"db": None}

            return {
                "ok": True,
                "family": fam,
                "version": (ver or {}).get("v"),
                "database": (dbn or {}).get("db"),
            }
        except Exception as e:
            return {"ok": False, "family": fam, "error": f"{type(e).__name__}: {e}"}
