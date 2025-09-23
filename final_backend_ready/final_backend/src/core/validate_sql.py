# src/core/validate_sql.py
from __future__ import annotations

import re
from typing import Iterable, List, Optional, Set

from src.core.settings import get_settings

_settings = get_settings()

# -----------------------------
# Regexes & helpers
# -----------------------------

# Code fences
_FENCE_RE = re.compile(r"```(?:sql)?|```", re.IGNORECASE)

# Strip line comments -- ... and block comments /* ... */
_LINE_COMMENT_RE = re.compile(r"--[^\n]*")
_BLOCK_COMMENT_RE = re.compile(r"/\*.*?\*/", re.DOTALL)

# String literals: N'...' or '...'
_STR_LITERAL_RE = re.compile(r"(?is)N?'(?:''|[^'])*'")

# First word (SELECT / WITH) to allow WITH CTEs that end in SELECT
_WITH_OR_SELECT_RE = re.compile(r"(?is)^\s*(?:with\b.*?\)\s*)*select\b")

# Blocklist: build from settings (INSERT|UPDATE|DELETE|DROP|ALTER|TRUNCATE|MERGE|EXEC|xp_|...)
_BLOCK_RE = re.compile(
    r"(?is)\b(" + r"|".join(re.escape(k) for k in _settings.SQL_BLOCKLIST) + r")\b"
)

# Find table targets after FROM/JOIN (captures schema-qualified or bare names)
_FROM_JOIN_TARGET_RE = re.compile(
    r"(?is)\b(from|join|apply|outer\s+apply|cross\s+apply)\s+((?:\[[^\]]+\]|\w+)(?:\s*\.\s*(?:\[[^\]]+\]|\w+))?)"
)

# Same, but also try to capture an alias that follows the target
# e.g. FROM dbo.Customers c   |   JOIN [dbo].[Leads] AS l
_FROM_JOIN_WITH_ALIAS_RE = re.compile(
    r"""(?is)
    \b(from|join|apply|outer\s+apply|cross\s+apply)\s+
    (?P<target>(?:\[[^\]]+\]|\w+)(?:\s*\.\s*(?:\[[^\]]+\]|\w+))?)
    (?:\s+(?:as\s+)?(?P<alias>\w+))?
    """
)

# Extract schema.table patterns anywhere (used for conservative allowlist checks)
_SCHEMA_TBL_RE = re.compile(
    r"(?is)(?:\[(?P<sch1>[^\]]+)\]|\b(?P<sch2>[A-Za-z_][\w]*)\b)\s*\.\s*(?:\[(?P<tb1>[^\]]+)\]|\b(?P<tb2>[A-Za-z_][\w]*)\b)"
)


def _strip_fences(sql: str) -> str:
    return _FENCE_RE.sub("", sql or "").strip()


def _remove_comments(sql: str) -> str:
    s = _BLOCK_COMMENT_RE.sub(" ", sql)
    s = _LINE_COMMENT_RE.sub(" ", s)
    return s


def _strip_string_literals(sql: str) -> str:
    # replace string contents with '' to protect downstream regexes
    return _STR_LITERAL_RE.sub("''", sql or "")


def _collapse_ws(sql: str) -> str:
    return re.sub(r"\s+", " ", sql or "").strip()


def _split_statements(sql: str) -> List[str]:
    """
    Split on semicolons not inside single/double quotes or [brackets].
    """
    parts: List[str] = []
    buf: List[str] = []
    in_single = False
    in_double = False
    in_bracket = False
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
    n = (name or "").strip().strip("[]").strip()
    return n.lower()


def _extract_from_join_targets(sql: str) -> List[str]:
    """
    Extract table targets after FROM/JOIN/APPLY.
    Returns:
      - 'schema.table' if qualified
      - 'table' if unqualified
    """
    targets: List[str] = []
    for m in _FROM_JOIN_TARGET_RE.finditer(sql):
        raw = (m.group(2) or "").strip()
        if "." in raw:
            parts = [p.strip() for p in re.split(r"\.", raw, maxsplit=1)]
            sch = _normalize_ident(parts[0])
            tb = _normalize_ident(parts[1])
            targets.append(f"{sch}.{tb}")
        else:
            targets.append(_normalize_ident(raw))
    return targets


def _extract_aliases(sql: str) -> Set[str]:
    """
    Collect likely table aliases appearing right after a FROM/JOIN/APPLY target.
    """
    aliases: Set[str] = set()
    # words that should not be treated as aliases even if they follow the target
    reserved = {
        "on", "where", "group", "order", "having", "union", "except", "intersect",
        "inner", "left", "right", "full", "cross", "outer", "apply", "join"
    }
    for m in _FROM_JOIN_WITH_ALIAS_RE.finditer(sql):
        alias = (m.group("alias") or "").strip()
        if alias and alias.lower() not in reserved:
            aliases.add(_normalize_ident(alias))
    return aliases


def _enforce_schema_allowlist(sql: str) -> None:
    """
    Ensure schema-qualified references use allowed schemas.
    Ignore alias.column like 'c.FirstName' by detecting known aliases.
    Also ignore anything inside string literals.
    """
    allow = {s.lower() for s in (_settings.ALLOW_SCHEMAS or ["dbo"])}
    # Work on a copy with string literals stripped so we don't match 'dbo.x' inside quotes
    s = _strip_string_literals(sql)
    aliases = _extract_aliases(s)

    for m in _SCHEMA_TBL_RE.finditer(s):
        sch = _normalize_ident(m.group("sch1") or m.group("sch2") or "")
        if not sch:
            continue
        # If the left token is a known alias, this is alias.column → skip
        if sch in aliases:
            continue
        if sch not in allow:
            raise ValueError(f"Disallowed schema referenced: {sch}")


def _inject_top_if_missing(sql: str, limit: int) -> str:
    has_top = re.search(r"(?is)\bselect\s+top\s+\d+", sql) is not None
    has_fetch = re.search(r"(?is)\bfetch\s+next\s+\d+\s+rows", sql) is not None
    if has_top or has_fetch:
        return sql
    return re.sub(r"(?is)^\s*select\s+", f"SELECT TOP {limit} ", sql, count=1)


# -----------------------------
# Public API
# -----------------------------

def extract_tables(sql: str) -> Set[str]:
    # normalize, strip comments & strings before hunting for FROM/JOIN targets
    s = _strip_string_literals(_collapse_ws(_remove_comments(_strip_fences(sql or ""))))
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

    # 4) Allowed tables (soft enforcement to avoid false blocks on multi-table/union cases)
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

        # strip strings before extracting refs to avoid matching 'dbo.x' inside quotes
        refs = _extract_from_join_targets(_strip_string_literals(s))

        if require_schema_qualified:
            # reject any bare refs
            for r in refs:
                if "." not in r:
                    raise ValueError(f"Unqualified table reference not allowed: {r}")

        # Soft mode: if a ref isn't in the allow list, don't fail hard.
        # (Schema allowlist + keyword blocklist above still protect us.)
        for r in refs:
            if "." in r:
                if r not in allow_norm:
                    continue
            else:
                if (r not in allow_norm) and (f"dbo.{r}" not in allow_norm):
                    continue

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
    WHERE b.STDCode = '079' AND CAST('dbo.fake' AS NVARCHAR(50)) IS NOT NULL
    """
    print("Tables:", extract_tables(demo))
    print(validate_sql(demo, allowed_tables=["dbo.Banks","dbo.Branches"]))
