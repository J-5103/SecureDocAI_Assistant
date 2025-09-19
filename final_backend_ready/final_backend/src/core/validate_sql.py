# src/core/validate_sql.py
from __future__ import annotations

import re
from typing import Iterable, List, Optional, Set, Tuple

from src.core.settings import get_settings

_settings = get_settings()

# -----------------------------
# Regexes & helpers
# -----------------------------

# Code fences (just in case)
_FENCE_RE = re.compile(r"```(?:sql)?|```", re.IGNORECASE)

# Strip line comments -- ... and block comments /* ... */
_LINE_COMMENT_RE = re.compile(r"--[^\n]*")
_BLOCK_COMMENT_RE = re.compile(r"/\*.*?\*/", re.DOTALL)

# First word (SELECT / WITH) to allow WITH CTEs that end in SELECT
_WITH_OR_SELECT_RE = re.compile(r"(?is)^\s*(?:with\b.*?\)\s*)*select\b")

# Blocklist: build from settings (INSERT|UPDATE|DELETE|DROP|ALTER|TRUNCATE|MERGE|EXEC|xp_|...)
_BLOCK_RE = re.compile(
    r"(?is)\b(" + r"|".join(re.escape(k) for k in _settings.SQL_BLOCKLIST) + r")\b"
)

# Find table references after FROM/JOIN (captures schema-qualified or bare names)
# Examples matched:
#   FROM dbo.Banks b
#   JOIN [dbo].[Branches] AS br
#   FROM Banks
_FROM_JOIN_TARGET_RE = re.compile(
    r"(?is)\b(from|join)\s+((?:\[[^\]]+\]|\w+)(?:\s*\.\s*(?:\[[^\]]+\]|\w+))?)"
)

# Extract schema.table patterns anywhere (for conservative allowlist checks)
_SCHEMA_TBL_RE = re.compile(
    r"(?is)(?:\[(?P<sch1>[^\]]+)\]|\b(?P<sch2>[A-Za-z_][\w]*)\b)\s*\.\s*(?:\[(?P<tb1>[^\]]+)\]|\b(?P<tb2>[A-Za-z_][\w]*)\b)"
)


def _strip_fences(sql: str) -> str:
    return _FENCE_RE.sub("", sql or "").strip()


def _remove_comments(sql: str) -> str:
    # remove block comments first, then line comments
    s = _BLOCK_COMMENT_RE.sub(" ", sql)
    s = _LINE_COMMENT_RE.sub(" ", s)
    return s


def _collapse_ws(sql: str) -> str:
    return re.sub(r"\s+", " ", sql or "").strip()


def _split_statements(sql: str) -> List[str]:
    """
    Split on semicolons not inside single/double brackets or quotes (best-effort).
    """
    parts: List[str] = []
    buf: List[str] = []
    in_single = False
    in_double = False
    in_bracket = False  # [identifier]
    prev = ""
    for ch in sql:
        if ch == "'" and not in_double and not in_bracket:
            in_single = not in_single
        elif ch == '"' and not in_single and not in_bracket:
            in_double = not in_double
        elif ch == "[" and not in_single and not in_double:
            in_bracket = True
        elif ch == "]" and in_bracket:
            in_bracket = False

        if ch == ";" and not (in_single or in_double or in_bracket):
            parts.append("".join(buf).strip())
            buf = []
        else:
            buf.append(ch)
        prev = ch
    tail = "".join(buf).strip()
    if tail:
        parts.append(tail)
    return [p for p in parts if p]


def _ensure_single_select(sql: str) -> str:
    stmts = _split_statements(sql.strip())
    if len(stmts) != 1:
        raise ValueError("Only a single SELECT statement is allowed.")
    s = stmts[0].strip()
    if not _WITH_OR_SELECT_RE.match(s):
        raise ValueError("Only SELECT (optionally WITH ... SELECT) is allowed.")
    return s


def _ensure_no_blocklist(sql: str) -> None:
    if _BLOCK_RE.search(sql):
        raise ValueError("Disallowed SQL keyword detected.")


def _normalize_ident(name: str) -> str:
    # strip brackets and whitespace; return lowercase
    n = name.strip().strip("[]").strip()
    return n.lower()


def _extract_from_join_targets(sql: str) -> List[str]:
    """
    Extract table targets after FROM/JOIN. Returns list of:
    - 'schema.table' if qualified
    - 'table' if unqualified
    """
    targets: List[str] = []
    for m in _FROM_JOIN_TARGET_RE.finditer(sql):
        raw = m.group(2).strip()
        # raw could be dbo.Table or [dbo].[Table] or Table
        if "." in raw:
            parts = [p.strip() for p in re.split(r"\.", raw, maxsplit=1)]
            sch = _normalize_ident(parts[0])
            tb = _normalize_ident(parts[1])
            targets.append(f"{sch}.{tb}")
        else:
            targets.append(_normalize_ident(raw))
    return targets


def _enforce_schema_allowlist(sql: str) -> None:
    """
    Ensure all schema-qualified references use allowed schemas (from settings).
    Bare table names are allowed here (treated as default schema); you can make
    this strict if you want to force schema qualification.
    """
    allow = {s.lower() for s in (_settings.ALLOW_SCHEMAS or ["dbo"])}
    for m in _SCHEMA_TBL_RE.finditer(sql):
        sch = _normalize_ident(m.group("sch1") or m.group("sch2") or "")
        if sch and sch not in allow:
            raise ValueError(f"Disallowed schema referenced: {sch}")


def _inject_top_if_missing(sql: str, limit: int) -> str:
    """
    Inject MSSQL TOP at first SELECT if no TOP/FETCH appears. Avoids changing
    semantics for COUNT/AGG + GROUP BY by still allowing a TOP on outer SELECT,
    which SQL Server accepts (it limits group rows).
    """
    has_top = re.search(r"(?is)\bselect\s+top\s+\d+", sql) is not None
    has_fetch = re.search(r"(?is)\bfetch\s+next\s+\d+\s+rows", sql) is not None
    if has_top or has_fetch:
        return sql
    return re.sub(r"(?is)^\s*select\s+", f"SELECT TOP {limit} ", sql, count=1)


# -----------------------------
# Public API
# -----------------------------

def extract_tables(sql: str) -> Set[str]:
    """
    Return a set of referenced table names:
      - schema-qualified 'schema.table' when present
      - bare 'table' for unqualified references
    """
    s = _collapse_ws(_remove_comments(_strip_fences(sql or "")))
    return set(_extract_from_join_targets(s))


def validate_sql(
    sql: str,
    *,
    allowed_tables: Optional[Iterable[str]] = None,
    enforce_row_limit: bool = True,
    row_limit: Optional[int] = None,
    require_schema_qualified: bool = False,
) -> str:
    """
    Validate and normalize an LLM-generated SQL string for **MSSQL SELECT-only** usage.

    - Ensures a single statement and that it starts with SELECT (or WITH ... SELECT).
    - Applies a keyword blocklist (from settings.SQL_BLOCKLIST).
    - Enforces schema allowlist (settings.ALLOW_SCHEMAS) for any qualified refs.
    - Optionally enforces that all referenced tables ∈ allowed_tables.
    - Optionally injects TOP {ROW_LIMIT} into the outer SELECT when missing.
    - Returns the cleaned SQL with a trailing semicolon.

    Args:
      sql: raw SQL text (possibly with code fences/comments).
      allowed_tables: iterable of allowed table names; can be 'dbo.Table' or bare 'Table'.
      enforce_row_limit: whether to inject TOP limit when not present.
      row_limit: override limit; defaults to settings.SQL_ROW_LIMIT.
      require_schema_qualified: if True, reject any bare table references.

    Raises:
      ValueError on any validation failure.
    """
    if not sql or not sql.strip():
        raise ValueError("Empty SQL.")

    # 1) Clean up
    s = _strip_fences(sql)
    s = _remove_comments(s)
    s = s.strip()

    # 2) Single SELECT (WITH allowed)
    s = _ensure_single_select(s)

    # 3) Safety checks
    _ensure_no_blocklist(s)
    _enforce_schema_allowlist(s)

    # 4) Allowed tables enforcement
    if allowed_tables is not None:
        allow_norm: Set[str] = set()
        for t in allowed_tables:
            t = (t or "").strip()
            if not t:
                continue
            if "." in t:
                sch, tb = t.split(".", 1)
                allow_norm.add(f"{_normalize_ident(sch)}.{_normalize_ident(tb)}")
            else:
                allow_norm.add(_normalize_ident(t))

        refs = _extract_from_join_targets(s)
        if require_schema_qualified:
            # reject any bare refs
            for r in refs:
                if "." not in r:
                    raise ValueError(f"Unqualified table reference not allowed: {r}")

        for r in refs:
            if "." in r:
                # schema-qualified must match fully
                if r not in allow_norm:
                    raise ValueError(f"Disallowed table referenced: {r}")
            else:
                # bare ref must match an allowed bare or any allowed fully-qualified with default schema dbo.<r>
                if (r not in allow_norm) and (f"dbo.{r}" not in allow_norm):
                    raise ValueError(f"Disallowed table referenced: {r}")

    # 5) Inject TOP if needed
    if enforce_row_limit:
        lim = int(row_limit or _settings.SQL_ROW_LIMIT or 100)
        s = _inject_top_if_missing(s, lim)

    # 6) Ensure trailing semicolon
    s = s.strip()
    if not s.endswith(";"):
        s += ";"
    return s


# -----------------------------
# Self-test
# -----------------------------
if __name__ == "__main__":
    demo = """
    -- count banks:
    SELECT COUNT(*) AS c
    FROM dbo.Banks b
    JOIN dbo.Branches br ON br.BankId = b.BankId
    WHERE b.STDCode = '079'
    """
    print("Tables:", extract_tables(demo))
    print(validate_sql(demo, allowed_tables=["dbo.Banks","dbo.Branches"]))
