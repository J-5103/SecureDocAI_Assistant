# src/core/schema_snapshot.py
from __future__ import annotations

import time
from typing import Dict, List, Tuple

from sqlalchemy import text

from src.core.db import get_engine
from src.core.settings import get_settings

_settings = get_settings()

# In-memory cache: (timestamp, schema_text)
_CACHE: Tuple[float, str] | None = None


def _get_allowed_schemas() -> List[str]:
    """Normalize allowed schemas list; default to ['dbo'] if unset."""
    allow = [s.strip() for s in (_settings.ALLOW_SCHEMAS or ["dbo"])]
    return [s for s in allow if s]


def _fetch_columns() -> List[Tuple[str, str, str, int]]:
    """
    Return rows: (schema, table, column, ordinal)
    Uses INFORMATION_SCHEMA.COLUMNS for portability.
    """
    allow = _get_allowed_schemas()
    placeholders = ", ".join([f":s{i}" for i in range(len(allow))]) or "('dbo')"

    sql = text(
        f"""
        SELECT
            c.TABLE_SCHEMA,
            c.TABLE_NAME,
            c.COLUMN_NAME,
            c.ORDINAL_POSITION
        FROM INFORMATION_SCHEMA.COLUMNS AS c
        WHERE c.TABLE_SCHEMA IN ({placeholders})
        ORDER BY c.TABLE_SCHEMA, c.TABLE_NAME, c.ORDINAL_POSITION
        """
    )
    params = {f"s{i}": sch for i, sch in enumerate(allow)}

    eng = get_engine()
    with eng.connect() as conn:
        rows = conn.execute(sql, params).all()

    # rows: List[Row[tuple]]; convert to plain tuples
    return [(r[0], r[1], r[2], int(r[3])) for r in rows]


def _fetch_column_types() -> Dict[Tuple[str, str, str], str]:
    """
    Map (schema, table, column) -> 'datatype(length/precision/scale)'
    """
    allow = _get_allowed_schemas()
    placeholders = ", ".join([f":s{i}" for i in range(len(allow))]) or "('dbo')"

    # Build a friendly type string similar to how people describe columns
    sql = text(
        f"""
        SELECT
            c.TABLE_SCHEMA,
            c.TABLE_NAME,
            c.COLUMN_NAME,
            CASE
                WHEN c.DATA_TYPE IN ('varchar','nvarchar','char','nchar','varbinary','binary')
                     AND c.CHARACTER_MAXIMUM_LENGTH IS NOT NULL
                THEN CONCAT(c.DATA_TYPE, '(', c.CHARACTER_MAXIMUM_LENGTH, ')')
                WHEN c.DATA_TYPE IN ('decimal','numeric')
                     AND c.NUMERIC_PRECISION IS NOT NULL
                THEN CONCAT(c.DATA_TYPE, '(', c.NUMERIC_PRECISION, ',', c.NUMERIC_SCALE, ')')
                ELSE c.DATA_TYPE
            END AS pretty_type
        FROM INFORMATION_SCHEMA.COLUMNS AS c
        WHERE c.TABLE_SCHEMA IN ({placeholders})
        """
    )
    params = {f"s{i}": sch for i, sch in enumerate(allow)}
    eng = get_engine()
    with eng.connect() as conn:
        rows = conn.execute(sql, params).all()

    out: Dict[Tuple[str, str, str], str] = {}
    for sch, tbl, col, ptype in rows:
        out[(str(sch), str(tbl), str(col))] = str(ptype)
    return out


def _fetch_foreign_keys() -> List[Tuple[str, str]]:
    """
    Return list of FK edges as tuples: ('schema.table', 'schema.table')
    Uses sys.foreign_keys + sys.foreign_key_columns for MSSQL.
    """
    allow = _get_allowed_schemas()
    placeholders = ", ".join([f":s{i}" for i in range(len(allow))]) or "('dbo')"

    sql = text(
        f"""
        SELECT
            QUOTENAME(OBJECT_SCHEMA_NAME(f.parent_object_id)) + '.' + QUOTENAME(OBJECT_NAME(f.parent_object_id)) AS src,
            QUOTENAME(OBJECT_SCHEMA_NAME(f.referenced_object_id)) + '.' + QUOTENAME(OBJECT_NAME(f.referenced_object_id)) AS dst
        FROM sys.foreign_keys AS f
        WHERE OBJECT_SCHEMA_NAME(f.parent_object_id) IN ({placeholders})
          AND OBJECT_SCHEMA_NAME(f.referenced_object_id) IN ({placeholders})
        GROUP BY
            QUOTENAME(OBJECT_SCHEMA_NAME(f.parent_object_id)) + '.' + QUOTENAME(OBJECT_NAME(f.parent_object_id)),
            QUOTENAME(OBJECT_SCHEMA_NAME(f.referenced_object_id)) + '.' + QUOTENAME(OBJECT_NAME(f.referenced_object_id))
        ORDER BY src, dst
        """
    )
    params = {f"s{i}": sch for i, sch in enumerate(allow)}
    eng = get_engine()
    with eng.connect() as conn:
        rows = conn.execute(sql, params).all()

    # Rows already quoted [schema].[table]; convert to plain schema.table
    edges: List[Tuple[str, str]] = []
    for src, dst in rows:
        edges.append((str(src).strip("[]").replace("].[", "."), str(dst).strip("[]").replace("].[", ".")))
    return edges


def build_schema_text(include_fks: bool = True) -> str:
    """
    Build a compact, LLM-friendly schema summary like:

    TABLE dbo.Banks:
      - BankId (int)
      - Name (nvarchar(200))
      - STDCode (varchar(10))

    TABLE dbo.Branches:
      - BranchId (int)
      - BankId (int)
      - City (nvarchar(100))

    Relationships:
      dbo.Branches -> dbo.Banks
    """
    cols = _fetch_columns()
    types = _fetch_column_types()

    # Group columns by table
    by_table: Dict[Tuple[str, str], List[Tuple[str, str]]] = {}
    for sch, tbl, col, _ord in cols:
        key = (sch, tbl)
        by_table.setdefault(key, []).append((col, types.get((sch, tbl, col), "")))

    lines: List[str] = []
    for (sch, tbl) in sorted(by_table.keys(), key=lambda k: (k[0].lower(), k[1].lower())):
        lines.append(f"TABLE {sch}.{tbl}:")
        cols_sorted = sorted(by_table[(sch, tbl)], key=lambda x: x[0].lower())
        for col, ptype in cols_sorted:
            if ptype:
                lines.append(f"  - {col} ({ptype})")
            else:
                lines.append(f"  - {col}")
        lines.append("")  # blank line between tables

    if include_fks:
        try:
            edges = _fetch_foreign_keys()
            if edges:
                lines.append("Relationships:")
                for src, dst in edges:
                    lines.append(f"  {src} -> {dst}")
                lines.append("")
        except Exception:
            # FK fetch isn't critical; ignore failures to keep snapshot robust
            pass

    # Trim trailing blank lines
    text_out = "\n".join(lines).rstrip()
    return text_out


def get_schema_text(force: bool = False, include_fks: bool = True) -> str:
    """
    Return cached schema text; rebuild if TTL expired or force=True.
    """
    global _CACHE
    now = time.time()
    ttl = max(30, int(_settings.SCHEMA_CACHE_TTL_SECS or 600))

    if force or _CACHE is None or (now - _CACHE[0]) > ttl:
        snap = build_schema_text(include_fks=include_fks)
        _CACHE = (now, snap)
    return _CACHE[1]


def refresh_schema_cache(include_fks: bool = True) -> str:
    """
    Force refresh and return the fresh schema snapshot text.
    """
    return get_schema_text(force=True, include_fks=include_fks)
