# src/core/catalog_builder.py
from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, List, Tuple

from sqlalchemy import text

from src.core.db import get_engine
from src.core.settings import get_settings

settings = get_settings()

# ---------------------------
# Config knobs (tune as needed)
# ---------------------------
MAX_SAMPLE_COLS = 8            # how many columns to sample per table (to keep it light)
SAMPLES_PER_COLUMN = 3         # distinct sample values per column
SAMPLE_CAST_NVARCHAR = 200     # CAST(.. AS NVARCHAR(N)) width to avoid giant payloads
SAMPLE_SKIP_TYPES = {
    "ntext", "text", "image", "xml", "geography", "geometry", "hierarchyid",
    "varbinary", "binary"
}


def _allowed_schemas() -> List[str]:
    allow = settings.ALLOW_SCHEMAS or ["dbo"]
    return [s for s in allow if s]


# ---------------------------
# INTROSPECTION QUERIES
# ---------------------------

def _fetch_columns_typed() -> List[Tuple[str, str, str, str, int]]:
    """
    Return: [(schema, table, column, pretty_type, ordinal), ...]
    Uses INFORMATION_SCHEMA for portability; pretty_type formats width/precision.
    """
    allow = _allowed_schemas()
    placeholders = ", ".join([f":s{i}" for i in range(len(allow))]) or "('dbo')"

    sql = text(f"""
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
            END AS pretty_type,
            c.ORDINAL_POSITION
        FROM INFORMATION_SCHEMA.COLUMNS c
        WHERE c.TABLE_SCHEMA IN ({placeholders})
        ORDER BY c.TABLE_SCHEMA, c.TABLE_NAME, c.ORDINAL_POSITION
    """)
    params = {f"s{i}": sch for i, sch in enumerate(allow)}

    eng = get_engine()
    with eng.connect() as conn:
        rows = conn.execute(sql, params).all()

    return [(str(r[0]), str(r[1]), str(r[2]), str(r[3]), int(r[4])) for r in rows]


def _fetch_primary_keys() -> List[Tuple[str, str, str, int]]:
    """
    Return: [(schema, table, column, key_ordinal), ...]
    """
    allow = _allowed_schemas()
    placeholders = ", ".join([f":s{i}" for i in range(len(allow))]) or "('dbo')"

    sql = text(f"""
        SELECT
          sch.name  AS schema_name,
          t.name    AS table_name,
          c.name    AS column_name,
          ic.key_ordinal
        FROM sys.tables t
        JOIN sys.schemas sch ON sch.schema_id = t.schema_id
        JOIN sys.indexes i ON i.object_id = t.object_id AND i.is_primary_key = 1
        JOIN sys.index_columns ic ON ic.object_id = t.object_id AND ic.index_id = i.index_id
        JOIN sys.columns c ON c.object_id = t.object_id AND c.column_id = ic.column_id
        WHERE sch.name IN ({placeholders})
        ORDER BY sch.name, t.name, ic.key_ordinal
    """)
    params = {f"s{i}": sch for i, sch in enumerate(allow)}

    eng = get_engine()
    with eng.connect() as conn:
        rows = conn.execute(sql, params).all()

    return [(str(r[0]), str(r[1]), str(r[2]), int(r[3])) for r in rows]


def _fetch_foreign_keys() -> List[Tuple[str, str, str, str, str, str, str, int]]:
    """
    Return FK edges with column pairs:
    [(src_schema, src_table, src_col, dst_schema, dst_table, dst_col, fk_name, ordinal), ...]
    """
    allow = _allowed_schemas()
    placeholders = ", ".join([f":s{i}" for i in range(len(allow))]) or "('dbo')"

    sql = text(f"""
        SELECT
          sch_p.name AS src_schema,
          tp.name    AS src_table,
          cp.name    AS src_column,
          sch_r.name AS dst_schema,
          tr.name    AS dst_table,
          cr.name    AS dst_column,
          fk.name    AS fk_name,
          k.constraint_column_id AS ord
        FROM sys.foreign_keys fk
        JOIN sys.foreign_key_columns k ON k.constraint_object_id = fk.object_id
        JOIN sys.tables tp ON tp.object_id = fk.parent_object_id
        JOIN sys.schemas sch_p ON sch_p.schema_id = tp.schema_id
        JOIN sys.columns cp ON cp.object_id = tp.object_id AND cp.column_id = k.parent_column_id
        JOIN sys.tables tr ON tr.object_id = fk.referenced_object_id
        JOIN sys.schemas sch_r ON sch_r.schema_id = tr.schema_id
        JOIN sys.columns cr ON cr.object_id = tr.object_id AND cr.column_id = k.referenced_column_id
        WHERE sch_p.name IN ({placeholders}) AND sch_r.name IN ({placeholders})
        ORDER BY fk.name, k.constraint_column_id
    """)
    params = {f"s{i}": sch for i, sch in enumerate(allow)}

    eng = get_engine()
    with eng.connect() as conn:
        rows = conn.execute(sql, params).all()

    return [(str(r[0]), str(r[1]), str(r[2]), str(r[3]), str(r[4]), str(r[5]), str(r[6]), int(r[7])) for r in rows]


def _fetch_row_counts() -> Dict[Tuple[str, str], int]:
    """
    Fast row counts using DMVs (approx but good for planning):
    {(schema, table): row_count}
    """
    allow = _allowed_schemas()
    placeholders = ", ".join([f":s{i}" for i in range(len(allow))]) or "('dbo')"

    sql = text(f"""
        SELECT
          sch.name AS schema_name,
          t.name   AS table_name,
          SUM(ps.rows) AS row_count
        FROM sys.tables t
        JOIN sys.schemas sch ON sch.schema_id = t.schema_id
        JOIN sys.dm_db_partition_stats ps ON ps.object_id = t.object_id AND ps.index_id IN (0,1)
        WHERE sch.name IN ({placeholders})
        GROUP BY sch.name, t.name
    """)
    params = {f"s{i}": sch for i, sch in enumerate(allow)}

    eng = get_engine()
    with eng.connect() as conn:
        rows = conn.execute(sql, params).all()

    out: Dict[Tuple[str, str], int] = {}
    for sch, tbl, cnt in rows:
        out[(str(sch), str(tbl))] = int(cnt or 0)
    return out


def _pick_sample_columns(columns: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """
    Choose a subset of columns for sampling (prioritize dimension-like/text-ish columns).
    Input: [(col_name, pretty_type)]
    """
    def score(ptype: str) -> int:
        t = ptype.lower()
        if any(x in t for x in ("char", "text", "date", "time")):
            return 3
        if any(x in t for x in ("int", "bigint", "smallint", "bit", "decimal", "numeric")):
            return 2
        return 1

    ranked = sorted(columns, key=lambda x: score(x[1]), reverse=True)
    return ranked[:MAX_SAMPLE_COLS]


def _fetch_samples_for_table(schema: str, table: str, cols: List[Tuple[str, str]]) -> Dict[str, List[str]]:
    """
    For each selected column, grab up to SAMPLES_PER_COLUMN distinct sample values (as strings).
    Skips large/binary types.
    """
    eng = get_engine()
    samples: Dict[str, List[str]] = {}

    for col, ptype in cols:
        base_type = ptype.split("(", 1)[0].lower()
        if base_type in SAMPLE_SKIP_TYPES:
            continue

        # Random-ish sampling: ORDER BY NEWID(), cast to NVARCHAR to keep payload small
        # NOTE: building dynamic SQL with quoted identifiers to avoid injection.
        sql = f"""
            SELECT TOP {SAMPLES_PER_COLUMN}
                   CAST([{col}] AS NVARCHAR({SAMPLE_CAST_NVARCHAR})) AS v
            FROM [{schema}].[{table}]
            WHERE [{col}] IS NOT NULL
            GROUP BY [{col}]
            ORDER BY NEWID();
        """
        try:
            with eng.connect() as conn:
                rows = conn.exec_driver_sql(sql).fetchall()
            vals = [str(r[0]) for r in rows if r[0] is not None]
            if vals:
                samples[col] = vals
        except Exception:
            # Sampling is best-effort; ignore failures (e.g., permissions, views, huge tables)
            continue

    return samples


# ---------------------------
# BUILD CATALOG
# ---------------------------

def build_catalog() -> Dict[str, Any]:
    """
    Build a rich JSON catalog:
    {
      "generatedAt": "...Z",
      "db": {"name": "DBNAME", "version": "..."},
      "tables": {
        "dbo.Banks": {
          "rowCount": 5421,
          "columns": {"BankId":"int","Name":"nvarchar(200)","STDCode":"varchar(10)"},
          "pk": ["BankId"],
          "samples": {"STDCode":["079","022"]}
        },
        ...
      },
      "fks": [
        {"name":"FK_Branches_Banks",
         "from":"dbo.Branches", "to":"dbo.Banks",
         "pairs":[["BankId","BankId"]]}
      ]
    }
    """
    eng = get_engine()
    with eng.connect() as conn:
        db_name = conn.exec_driver_sql("SELECT DB_NAME()").scalar()
        db_ver = conn.exec_driver_sql("SELECT @@VERSION").scalar()

    cols_typed = _fetch_columns_typed()
    pks = _fetch_primary_keys()
    fks = _fetch_foreign_keys()
    counts = _fetch_row_counts()

    # Organize column info per table
    tables: Dict[str, Dict[str, Any]] = {}
    by_table_cols: Dict[Tuple[str, str], List[Tuple[str, str]]] = {}

    for sch, tbl, col, ptype, _ord in cols_typed:
        key = (sch, tbl)
        by_table_cols.setdefault(key, []).append((col, ptype))

    # Primary keys per table
    pk_map: Dict[Tuple[str, str], List[str]] = {}
    for sch, tbl, col, ord_ in pks:
        pk_map.setdefault((sch, tbl), []).append((ord_, col))
    # sort by ordinal and flatten
    for k, lst in pk_map.items():
        lst.sort(key=lambda x: x[0])
        pk_map[k] = [c for _, c in lst]

    # Samples per table (best-effort)
    for (sch, tbl), cols in by_table_cols.items():
        tkey = f"{sch}.{tbl}"
        col_map = {c: t for c, t in cols}
        tinfo: Dict[str, Any] = {
            "rowCount": counts.get((sch, tbl), 0),
            "columns": col_map,
        }
        if (sch, tbl) in pk_map:
            tinfo["pk"] = pk_map[(sch, tbl)]
        # samples (small & safe subset)
        sample_cols = _pick_sample_columns(cols)
        sample_dict = _fetch_samples_for_table(sch, tbl, sample_cols)
        if sample_dict:
            tinfo["samples"] = sample_dict

        tables[tkey] = tinfo

    # FK top-level list with column pairs grouped by FK name
    fk_map: Dict[str, Dict[str, Any]] = {}
    for src_s, src_t, src_c, dst_s, dst_t, dst_c, fk_name, ord_ in fks:
        key = fk_name
        entry = fk_map.setdefault(key, {
            "name": fk_name,
            "from": f"{src_s}.{src_t}",
            "to": f"{dst_s}.{dst_t}",
            "pairs": []
        })
        entry["pairs"].append([src_c, dst_c])

    catalog: Dict[str, Any] = {
        "generatedAt": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "db": {"name": db_name, "version": db_ver},
        "tables": tables,
        "fks": list(fk_map.values()),
        "schemas": _allowed_schemas(),
    }
    return catalog


# ---------------------------
# SAVE / LOAD HELPERS
# ---------------------------

def save_catalog(catalog: Dict[str, Any], path: str | None = None) -> str:
    target = path or settings.CATALOG_PATH
    with open(target, "w", encoding="utf-8") as f:
        json.dump(catalog, f, ensure_ascii=False, indent=2)
    return target


def load_catalog(path: str | None = None) -> Dict[str, Any]:
    target = path or settings.CATALOG_PATH
    with open(target, "r", encoding="utf-8") as f:
        return json.load(f)


def build_and_save(path: str | None = None) -> str:
    catalog = build_catalog()
    return save_catalog(catalog, path)


# ---------------------------
# CLI entry (manual run)
# ---------------------------
if __name__ == "__main__":
    p = build_and_save()
    print(f"Catalog built and saved to: {p}")
