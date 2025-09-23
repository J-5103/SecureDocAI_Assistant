# src/core/text2sql.py
from __future__ import annotations

import re
from typing import List, Optional, Set, Tuple

import requests

from src.core.settings import get_settings
from src.core.schema_snapshot import get_schema_text

_settings = get_settings()

# -----------------------------
# System prompt & few-shot data
# -----------------------------

_SYS_PROMPT = """You are a senior SQL engineer for Microsoft SQL Server (T-SQL).
Return ONLY a single valid SELECT statement. No prose. No explanations. No code fences.

HARD RULES:
- Read-only: SELECT statements only. Never modify data (no INSERT/UPDATE/DELETE/MERGE).
- Use schema-qualified names such as dbo.Table.
- Prefer explicit column lists over SELECT * when practical.
- If no explicit limit is provided, cap rows using TOP {ROW_LIMIT}.
- Use ANSI-compliant syntax accepted by SQL Server.
- Use only tables and columns that exist in the provided schema.
- When joining tables, use obvious foreign keys or name-matching columns.
- All identifiers are case-insensitive unless quoted; avoid quoting unless necessary.
"""

# Keep these concise and domain-flavored; adjust to your tables later.
_FEWSHOTS: List[tuple[str, str]] = [
    (
        "Count banks with STD code 079",
        "SELECT COUNT(*) AS bank_count FROM dbo.Banks WHERE STDCode = '079';",
    ),
    (
        "List top 10 branches by total transaction amount in 2024",
        "SELECT TOP 10 b.BranchName, SUM(t.Amount) AS total_amount "
        "FROM dbo.Transactions t "
        "JOIN dbo.Branches b ON b.BranchId = t.BranchId "
        "WHERE t.TransactionDate >= '2024-01-01' AND t.TransactionDate < '2025-01-01' "
        "GROUP BY b.BranchName "
        "ORDER BY total_amount DESC;",
    ),
    (
        "Average loan amount per city",
        "SELECT TOP 100 c.City, AVG(l.Amount) AS avg_amount "
        "FROM dbo.Loans l "
        "JOIN dbo.Customers c ON c.CustomerId = l.CustomerId "
        "GROUP BY c.City "
        "ORDER BY avg_amount DESC;",
    ),
]


def _fewshots_block() -> str:
    if not _settings.ENABLE_FEWSHOTS or not _FEWSHOTS:
        return ""
    parts = []
    for q, a in _FEWSHOTS:
        parts.append(f"Q: {q}\nA: {a}")
    return "\n\n".join(parts)


def _build_prompt(question: str, schema_text: str, *, forbid: Optional[Set[str]] = None) -> str:
    base = _SYS_PROMPT.format(ROW_LIMIT=_settings.SQL_ROW_LIMIT)
    few = _fewshots_block()
    # Keep the prompt bounded if schema is huge
    schema_trimmed = schema_text
    max_chars = 12000  # safe budget for most local models
    if len(schema_trimmed) > max_chars:
        schema_trimmed = schema_trimmed[:max_chars] + "\n-- (schema truncated) --"

    prompt = (
        f"{base}\n\n"
        f"Relevant database schema (tables, columns, relationships):\n"
        f"{schema_trimmed}\n\n"
    )
    if forbid:
        prompt += (
            "Do NOT use these tables (they are invalid or not present in the schema): "
            + ", ".join(sorted(forbid))
            + ".\n\n"
        )
    if few:
        prompt += f"{few}\n\n"
    prompt += f"User question: {question}\nSQL:"
    return prompt


# -----------------------------
# Ollama call & output cleanup
# -----------------------------

_CODE_FENCE_RE = re.compile(r"```(?:sql)?|```", re.IGNORECASE)
_SELECT_START_RE = re.compile(r"(?is)\bselect\b")


def _strip_fences(text: str) -> str:
    return _CODE_FENCE_RE.sub("", text or "").strip()


def _first_statement(sql: str) -> str:
    """Return first semicolon-terminated statement; if none, return the first line heuristically."""
    s = sql.strip()
    if ";" in s:
        s = s.split(";")[0]
    return s.strip()


def _normalize_sql(text: str) -> str:
    """
    Normalize LLM output to a single SELECT statement:
    - remove code fences/prose
    - find first SELECT ... [;]
    - ensure trailing semicolon
    """
    raw = _strip_fences(text)
    # try to find the first SELECT
    m = _SELECT_START_RE.search(raw)
    if not m:
        # sometimes the model prepends chit-chat; try to salvage a SELECT with a broad pattern
        candidates = re.findall(r"(?is)\bselect\b.+", raw)
        if candidates:
            raw = candidates[0]
        else:
            return ""

    sql = raw[m.start():]
    sql = _first_statement(sql)
    sql = sql.rstrip()
    if not sql.endswith(";"):
        sql += ";"
    return sql


def _post_ollama(prompt: str) -> str:
    """
    Call Ollama's /api/generate with given prompt.
    """
    url = f"{_settings.OLLAMA_URL}/api/generate"
    payload = {
        "model": _settings.OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        # Conservative decoding for determinism on SQL
        "options": {
            "temperature": 0.1,
            "num_predict": 256,
            "top_p": 0.9,
        },
    }
    r = requests.post(url, json=payload, timeout=90)
    r.raise_for_status()
    data = r.json()
    return data.get("response", "")


# -----------------------------
# Light validation & row capping
# -----------------------------

# capture schema-qualified table tokens like dbo.Table or [dbo].[Table]
_QNAME_RE = re.compile(r"(?i)(?:\[(?P<s1>[^\]]+)\]|\b(?P<s2>[A-Za-z_]\w*))\s*\.\s*(?:\[(?P<t1>[^\]]+)\]|\b(?P<t2>[A-Za-z_]\w*))")

def _extract_tables_from_text(txt: str) -> Set[str]:
    out: Set[str] = set()
    for m in _QNAME_RE.finditer(txt or ""):
        schema = m.group("s1") or m.group("s2")
        table = m.group("t1") or m.group("t2")
        if schema and table:
            out.add(f"{schema}.{table}")
    return out

_LIMIT_HINT_RE = re.compile(r"(?is)\b(top\s+\d+|offset\s+\d+\s+rows|fetch\s+next\s+\d+\s+rows\s+only)\b")
_COUNT_AGG_RE = re.compile(r"(?is)\bcount\s*\(", re.IGNORECASE)

def _ensure_top_limit(sql: str, row_limit: int) -> str:
    """If the query is not obviously limited, inject TOP {row_limit} after the first SELECT (careful of DISTINCT)."""
    s = sql.strip()
    if not s.lower().startswith("select"):
        return s
    if _LIMIT_HINT_RE.search(s):
        return s
    # simple COUNT(*) or other aggregate-only often returns 1 row; skip TOP
    if _COUNT_AGG_RE.search(s) and "group by" not in s.lower():
        return s

    # Insert TOP n after SELECT or SELECT DISTINCT
    m = re.match(r"(?is)select\s+distinct", s)
    if m:
        return s[:m.end()] + f" TOP {row_limit}" + s[m.end():]
    m2 = re.match(r"(?is)select", s)
    if m2:
        return s[:m2.end()] + f" TOP {row_limit}" + s[m2.end():]
    return s


def _validate_tables(sql: str, schema_text: str) -> Tuple[Set[str], Set[str]]:
    """Return (used_tables, unknown_tables) comparing tables in SQL vs schema snapshot text."""
    used = _extract_tables_from_text(sql)
    known = _extract_tables_from_text(schema_text)
    unknown = {t for t in used if t not in known}
    return used, unknown


# -----------------------------
# Public API
# -----------------------------

def generate_sql(question: str, *, schema_text: Optional[str] = None) -> str:
    """
    Generate a single, read-only SQL SELECT statement for SQL Server.

    Args:
        question: Natural language question from the user.
        schema_text: Optional prebuilt schema snapshot; if omitted, uses cached snapshot.

    Returns:
        A single SQL statement string ending with ';'. Empty string if generation failed.
    """
    q = (question or "").strip()
    if not q:
        return ""

    schema = schema_text or get_schema_text()  # cached snapshot

    # ---- Primary attempt
    prompt = _build_prompt(q, schema)
    out = _post_ollama(prompt)
    sql = _normalize_sql(out)

    # If failed to get SELECT, try a smaller prompt (no few-shots, shorter schema)
    if not _SELECT_START_RE.match(sql or ""):
        tiny_schema = "\n".join(schema.splitlines()[:120])  # ~ first 120 lines
        tiny_prompt = _build_prompt(q, tiny_schema if tiny_schema else schema)
        out2 = _post_ollama(tiny_prompt)
        sql = _normalize_sql(out2)

    if not sql:
        return ""

    # ---- Light validation against schema to avoid hallucinated tables
    _, unknown = _validate_tables(sql, schema)
    if unknown:
        # Ask again, explicitly forbidding the hallucinated tables once
        forbid = set(unknown)
        retry_prompt = _build_prompt(q, schema, forbid=forbid)
        out3 = _post_ollama(retry_prompt)
        sql2 = _normalize_sql(out3)
        if sql2:
            sql = sql2  # use retry result

    # ---- Enforce row cap when appropriate
    sql = _ensure_top_limit(sql, _settings.SQL_ROW_LIMIT)

    # ensure single trailing semicolon
    if not sql.endswith(";"):
        sql += ";"

    return sql


# Convenience for quick manual tests
if __name__ == "__main__":
    print(generate_sql("Count banks with STD code 079"))
