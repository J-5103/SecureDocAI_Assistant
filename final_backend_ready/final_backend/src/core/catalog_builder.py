# src/core/catalog_builder.py
from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, List, Tuple, Optional

from sqlalchemy import text
from sqlalchemy.engine import Engine

from src.core.db import get_engine
from src.core.settings import get_settings

settings = get_settings()

# ---------------------------
# Config knobs (tune as needed)
# ---------------------------
MAX_SAMPLE_COLS = 8            # how many columns to sample per table (to keep it light)
SAMPLES_PER_COLUMN = 3         # distinct sample values per column
SAMPLE_CAST_NVARCHAR = 200     # MSSQL cast width to avoid giant payloads
SAMPLE_SKIP_TYPES = {
    "ntext", "text", "image", "xml", "geography", "geometry", "hierarchyid",
    "varbinary", "binary", "bytea", "json", "jsonb", "tsvector", "blob"
}

# ---------------------------
# Helpers
# ---------------------------

def _family(engine: Engine) -> str:
    n = (engine.dialect.name or "").lower()
    if "mssql" in n:
        return "mssql"
    if "postgres" in n:
        return "postgres"
    if "sqlite" in n:
        return "sqlite"
    return n or "unknown"

def _allowed_schemas(fam: str) -> List[str]:
    allow = list(getattr(settings, "ALLOW_SCHEMAS", []) or [])
    if allow:
        return [s for s in allow if s]
    # sensible defaults
    if fam == "mssql":
        return ["dbo"]
    if fam == "postgres":
        return ["public"]
    if fam == "sqlite":
        return ["main"]  # alias for UI keys; SQLite doesn’t really use schemas
    return []

def _q_mssql(ident: str) -> str:
    return f"[{ident.replace(']', ']]')}]"

def _q_pg(ident: str) -> str:
    return f"\"{ident.replace('\"', '\"\"')}\""

def _q_sqlite(ident: str) -> str:
    return f"\"{ident.replace('\"', '\"\"')}\""

# ---------------------------
# INTROSPECTION (per dialect)
# ---------------------------

def _fetch_columns_typed(engine: Engine, fam: str, allow: List[str]) -> List[Tuple[str, str, str, str, int]]:
    """
    Return: [(schema, table, column, pretty_type, ordinal), ...]
    """
    if fam in ("mssql", "postgres"):
        placeholders = ", ".join([f":s{i}" for i in range(len(allow))]) or (":s0" if fam == "mssql" else ":s0")
        if not allow:
            allow = _allowed_schemas(fam)
        sql = text(f"""
            SELECT
                c.TABLE_SCHEMA,
                c.TABLE_NAME,
                c.COLUMN_NAME,
                CASE
                  WHEN c.DATA_TYPE IN ('varchar','nvarchar','char','nchar','varbinary','binary','character varying','character','varbit','bit')
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
        params = {f"s{i}": sch for i, sch in enumerate(allow)} or {"s0": "dbo" if fam == "mssql" else "public"}
        with engine.connect() as conn:
            rows = conn.execute(sql, params).all()
        return [(str(r[0]), str(r[1]), str(r[2]), str(r[3]), int(r[4])) for r in rows]

    # SQLite
    out: List[Tuple[str, str, str, str, int]] = []
    with engine.connect() as conn:
        # list tables (exclude sqlite internal)
        trows = conn.exec_driver_sql("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'").fetchall()
        for (tbl,) in trows:
            pragma = conn.exec_driver_sql(f"PRAGMA table_info({_q_sqlite(tbl)})").fetchall()
            # pragma columns: cid, name, type, notnull, dflt_value, pk
            for cid, name, typ, _nn, _def, _pk in pragma:
                ptype = str(typ or "").strip() or "text"
                out.append(("main", str(tbl), str(name), ptype, int(cid) + 1))
    return out

def _fetch_primary_keys(engine: Engine, fam: str, allow: List[str]) -> List[Tuple[str, str, str, int]]:
    """Return: [(schema, table, column, key_ordinal), ...]"""
    if fam == "mssql":
        placeholders = ", ".join([f":s{i}" for i in range(len(allow))]) or ":s0"
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
        params = {f"s{i}": sch for i, sch in enumerate(allow)} or {"s0": "dbo"}
        with engine.connect() as conn:
            rows = conn.execute(sql, params).all()
        return [(str(r[0]), str(r[1]), str(r[2]), int(r[3])) for r in rows]

    if fam == "postgres":
        placeholders = ", ".join([f":s{i}" for i in range(len(allow))]) or ":s0"
        sql = text(f"""
            SELECT
              tc.table_schema, tc.table_name, kcu.column_name, kcu.ordinal_position
            FROM information_schema.table_constraints tc
            JOIN information_schema.key_column_usage kcu
              ON tc.constraint_name = kcu.constraint_name
             AND tc.table_schema = kcu.table_schema
            WHERE tc.constraint_type = 'PRIMARY KEY'
              AND tc.table_schema IN ({placeholders})
            ORDER BY tc.table_schema, tc.table_name, kcu.ordinal_position
        """)
        params = {f"s{i}": sch for i, sch in enumerate(allow)} or {"s0": "public"}
        with engine.connect() as conn:
            rows = conn.execute(sql, params).all()
        return [(str(r[0]), str(r[1]), str(r[2]), int(r[3])) for r in rows]

    # SQLite
    out: List[Tuple[str, str, str, int]] = []
    with engine.connect() as conn:
        trows = conn.exec_driver_sql("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'").fetchall()
        for (tbl,) in trows:
            pragma = conn.exec_driver_sql(f"PRAGMA table_info({_q_sqlite(tbl)})").fetchall()
            ord_ = 0
            for _cid, name, _type, _nn, _def, pk in pragma:
                if int(pk or 0) > 0:
                    ord_ += 1
                    out.append(("main", str(tbl), str(name), ord_))
    return out

def _fetch_foreign_keys(engine: Engine, fam: str, allow: List[str]) -> List[Tuple[str, str, str, str, str, str, str, int]]:
    """
    Return FK edges with column pairs:
    [(src_schema, src_table, src_col, dst_schema, dst_table, dst_col, fk_name, ordinal), ...]
    """
    if fam == "mssql":
        placeholders = ", ".join([f":s{i}" for i in range(len(allow))]) or ":s0"
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
        params = {f"s{i}": sch for i, sch in enumerate(allow)} or {"s0": "dbo"}
        with engine.connect() as conn:
            rows = conn.execute(sql, params).all()
        return [(str(r[0]), str(r[1]), str(r[2]), str(r[3]), str(r[4]), str(r[5]), str(r[6]), int(r[7])) for r in rows]

    if fam == "postgres":
        placeholders = ", ".join([f":s{i}" for i in range(len(allow))]) or ":s0"
        sql = text(f"""
            SELECT
              tc.table_schema AS src_schema,
              tc.table_name   AS src_table,
              kcu.column_name AS src_column,
              ccu.table_schema AS dst_schema,
              ccu.table_name   AS dst_table,
              ccu.column_name  AS dst_column,
              tc.constraint_name AS fk_name,
              kcu.ordinal_position AS ord
            FROM information_schema.table_constraints AS tc
            JOIN information_schema.key_column_usage AS kcu
              ON tc.constraint_name = kcu.constraint_name
             AND tc.table_schema = kcu.table_schema
            JOIN information_schema.constraint_column_usage AS ccu
              ON ccu.constraint_name = tc.constraint_name
             AND ccu.table_schema = tc.table_schema
            WHERE tc.constraint_type = 'FOREIGN KEY'
              AND tc.table_schema IN ({placeholders})
            ORDER BY tc.constraint_name, kcu.ordinal_position
        """)
        params = {f"s{i}": sch for i, sch in enumerate(allow)} or {"s0": "public"}
        with engine.connect() as conn:
            rows = conn.execute(sql, params).all()
        return [(str(r[0]), str(r[1]), str(r[2]), str(r[3]), str(r[4]), str(r[5]), str(r[6]), int(r[7])) for r in rows]

    # SQLite
    out: List[Tuple[str, str, str, str, str, str, str, int]] = []
    with engine.connect() as conn:
        trows = conn.exec_driver_sql("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'").fetchall()
        for (tbl,) in trows:
            fkrows = conn.exec_driver_sql(f"PRAGMA foreign_key_list({_q_sqlite(tbl)})").fetchall()
            # columns: id, seq, table, from, to, on_update, on_delete, match
            for r in fkrows:
                _id, seq, dst_table, src_col, dst_col, *_rest = r
                fk_name = f"fk_{tbl}_{dst_table}"
                out.append((
                    "main", str(tbl), str(src_col),
                    "main", str(dst_table), str(dst_col),
                    fk_name, int(seq)
                ))
    return out

def _fetch_row_counts(engine: Engine, fam: str, allow: List[str]) -> Dict[Tuple[str, str], int]:
    """
    {(schema, table): row_count}
    Use system views where available; avoid heavy full COUNT(*) on large DBs.
    """
    if fam == "mssql":
        placeholders = ", ".join([f":s{i}" for i in range(len(allow))]) or ":s0"
        # ✅ Use sys.partitions (p.rows) for broad compatibility
        sql = text(f"""
            SELECT
              sch.name AS schema_name,
              t.name   AS table_name,
              SUM(p.rows) AS row_count
            FROM sys.tables t
            JOIN sys.schemas sch ON sch.schema_id = t.schema_id
            JOIN sys.partitions p ON p.object_id = t.object_id AND p.index_id IN (0,1)
            WHERE sch.name IN ({placeholders})
            GROUP BY sch.name, t.name
        """)
        params = {f"s{i}": sch for i, sch in enumerate(allow)} or {"s0": "dbo"}
        with engine.connect() as conn:
            rows = conn.execute(sql, params).all()
        return {(str(s), str(t)): int(cnt or 0) for s, t, cnt in rows}

    if fam == "postgres":
        placeholders = ", ".join([f":s{i}" for i in range(len(allow))]) or ":s0"
        sql = text(f"""
            SELECT n.nspname AS schema_name, c.relname AS table_name,
                   COALESCE(c.reltuples::bigint, 0) AS row_count
            FROM pg_class c
            JOIN pg_namespace n ON n.oid = c.relnamespace
            WHERE c.relkind = 'r' AND n.nspname IN ({placeholders})
        """)
        params = {f"s{i}": sch for i, sch in enumerate(allow)} or {"s0": "public"}
        with engine.connect() as conn:
            rows = conn.execute(sql, params).all()
        return {(str(s), str(t)): int(cnt or 0) for s, t, cnt in rows}

    # SQLite: avoid heavy counts; return 0 (UI can still use samples)
    out: Dict[Tuple[str, str], int] = {}
    with engine.connect() as conn:
        trows = conn.exec_driver_sql("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'").fetchall()
        for (tbl,) in trows:
            out[("main", str(tbl))] = 0
    return out

def _pick_sample_columns(columns: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """
    Choose a subset of columns for sampling (prioritize dimension-like/text-ish columns).
    Input: [(col_name, pretty_type)]
    """
    def score(ptype: str) -> int:
        t = (ptype or "").lower()
        if any(x in t for x in ("char", "text", "date", "time")):
            return 3
        if any(x in t for x in ("int", "bigint", "smallint", "bit", "decimal", "numeric")):
            return 2
        return 1

    ranked = sorted(columns, key=lambda x: score(x[1]), reverse=True)
    return ranked[:MAX_SAMPLE_COLS]

def _fetch_samples_for_table(engine: Engine, fam: str, schema: str, table: str, cols: List[Tuple[str, str]]) -> Dict[str, List[str]]:
    """
    For each selected column, grab up to SAMPLES_PER_COLUMN distinct sample values.
    """
    samples: Dict[str, List[str]] = {}
    for col, ptype in cols:
        base_type = (ptype or "").split("(", 1)[0].lower()
        if base_type in SAMPLE_SKIP_TYPES:
            continue

        try:
            if fam == "mssql":
                sql = f"""
                    SELECT TOP {SAMPLES_PER_COLUMN}
                           CAST({_q_mssql(col)} AS NVARCHAR({SAMPLE_CAST_NVARCHAR})) AS v
                    FROM {_q_mssql(schema)}.{_q_mssql(table)}
                    WHERE {_q_mssql(col)} IS NOT NULL
                    GROUP BY {_q_mssql(col)}
                    ORDER BY NEWID();
                """
                with engine.connect() as conn:
                    rows = conn.exec_driver_sql(sql).fetchall()
            elif fam == "postgres":
                sql = f"""
                    SELECT CAST({_q_pg(col)} AS TEXT) AS v
                    FROM {_q_pg(schema)}.{_q_pg(table)}
                    WHERE {_q_pg(col)} IS NOT NULL
                    GROUP BY {_q_pg(col)}
                    ORDER BY random()
                    LIMIT {SAMPLES_PER_COLUMN};
                """
                with engine.connect() as conn:
                    rows = conn.exec_driver_sql(sql).fetchall()
            else:  # sqlite
                sql = f"""
                    SELECT CAST({_q_sqlite(col)} AS TEXT) AS v
                    FROM {_q_sqlite(table)}
                    WHERE {_q_sqlite(col)} IS NOT NULL
                    GROUP BY {_q_sqlite(col)}
                    ORDER BY RANDOM()
                    LIMIT {SAMPLES_PER_COLUMN};
                """
                with engine.connect() as conn:
                    rows = conn.exec_driver_sql(sql).fetchall()

            vals = [str(r[0]) for r in rows if r[0] is not None]
            if vals:
                samples[col] = vals
        except Exception:
            # Sampling is best-effort; ignore failures (permissions/huge tables/etc.)
            continue
    return samples

# ---------------------------
# BUILD CATALOG
# ---------------------------

def build_catalog(engine: Optional[Engine] = None) -> Dict[str, Any]:
    """
    Build a rich JSON catalog (see structure in docstring).
    """
    engine = engine or get_engine()
    fam = _family(engine)
    allow = _allowed_schemas(fam)

    # DB info
    with engine.connect() as conn:
        if fam == "mssql":
            db_name = conn.exec_driver_sql("SELECT DB_NAME()").scalar()
            db_ver = conn.exec_driver_sql("SELECT @@VERSION").scalar()
        elif fam == "postgres":
            db_name = conn.execute(text("SELECT current_database()")).scalar()
            db_ver = conn.execute(text("SELECT version()")).scalar()
        else:  # sqlite
            db_name = "sqlite"
            db_ver = conn.execute(text("SELECT sqlite_version()")).scalar()

    cols_typed = _fetch_columns_typed(engine, fam, allow)
    pks = _fetch_primary_keys(engine, fam, allow)
    fks = _fetch_foreign_keys(engine, fam, allow)
    counts = _fetch_row_counts(engine, fam, allow)

    # Organize column info per table
    tables: Dict[str, Dict[str, Any]] = {}
    by_table_cols: Dict[Tuple[str, str], List[Tuple[str, str]]] = {}

    for sch, tbl, col, ptype, _ord in cols_typed:
        key = (sch, tbl)
        by_table_cols.setdefault(key, []).append((col, ptype))

    # Primary keys per table
    pk_map: Dict[Tuple[str, str], List[Tuple[int, str]]] = {}
    for sch, tbl, col, ord_ in pks:
        pk_map.setdefault((sch, tbl), []).append((ord_, col))
    for k, lst in pk_map.items():
        lst.sort(key=lambda x: x[0])
        pk_map[k] = [(o, c) for o, c in lst]  # keep ord—flatten later

    # Build table entries
    for (sch, tbl), cols in by_table_cols.items():
        tkey = f"{sch}.{tbl}"
        col_map = {c: t for c, t in cols}
        tinfo: Dict[str, Any] = {
            "rowCount": counts.get((sch, tbl), 0),
            "columns": col_map,
        }
        if (sch, tbl) in pk_map:
            tinfo["pk"] = [c for _o, c in pk_map[(sch, tbl)]]

        # samples (small & safe subset)
        sample_cols = _pick_sample_columns(cols)
        sample_dict = _fetch_samples_for_table(engine, fam, sch, tbl, sample_cols)
        if sample_dict:
            tinfo["samples"] = sample_dict

        tables[tkey] = tinfo

    # FK top-level list with column pairs grouped by FK name
    fk_map: Dict[str, Dict[str, Any]] = {}
    for src_s, src_t, src_c, dst_s, dst_t, dst_c, fk_name, ord_ in fks:
        entry = fk_map.setdefault(fk_name, {
            "name": fk_name,
            "from": f"{src_s}.{src_t}",
            "to": f"{dst_s}.{dst_t}",
            "pairs": []
        })
        entry["pairs"].append([src_c, dst_c])

    catalog: Dict[str, Any] = {
        "generatedAt": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "db": {"name": db_name, "version": db_ver, "family": fam},
        "tables": tables,
        "fks": list(fk_map.values()),
        "schemas": allow,
    }
    return catalog

# ---------------------------
# SAVE / LOAD HELPERS
# ---------------------------

def _resolved_catalog_path() -> str:
    # Prefer resolved helper if present
    p = getattr(settings, "CATALOG_PATH_RESOLVED", None)
    if p:
        return str(p)
    return getattr(settings, "CATALOG_PATH", "./data/catalog.json")

def save_catalog(catalog: Dict[str, Any], path: str | None = None) -> str:
    target = path or _resolved_catalog_path()
    with open(target, "w", encoding="utf-8") as f:
        json.dump(catalog, f, ensure_ascii=False, indent=2)
    return target

def load_catalog(path: str | None = None) -> Dict[str, Any]:
    target = path or _resolved_catalog_path()
    with open(target, "r", encoding="utf-8") as f:
        return json.load(f)

def build_and_save(path: str | None = None, engine: Optional[Engine] = None) -> str:
    catalog = build_catalog(engine=engine)
    return save_catalog(catalog, path)

# ---------------------------
# CLI entry (manual run)
# ---------------------------
if __name__ == "__main__":
    p = build_and_save()
    print(f"Catalog built and saved to: {p}")
