# src/core/prompt_sql.py
from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

from src.core.settings import get_settings

_settings = get_settings()

# -----------------------------
# System rules (MSSQL / T-SQL)
# -----------------------------
_SYSTEM_RULES = """You are a senior SQL engineer for Microsoft SQL Server (T-SQL).
Return ONLY a single valid SELECT statement. No explanations. No comments. No code fences.

Hard rules:
- Read-only: SELECT statements only. Never modify data.
- Use schema-qualified names such as dbo.Table.
- Prefer explicit column lists over SELECT *.
- If no limit is provided, cap rows using TOP {ROW_LIMIT}.
- Use SQL that is accepted by Microsoft SQL Server.
- Use obvious foreign key relationships when joining tables.
- Do not reference tables or columns that are not present in the provided schema snippet.
- If multiple interpretations exist, choose the most likely based on column names and relationships.
"""

# -----------------------------
# Default few-shots (edit to your domain)
# -----------------------------
_DEFAULT_FEWSHOTS: List[Tuple[str, str]] = [
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

# -----------------------------
# Helpers
# -----------------------------

def _format_fewshots(
    shots: Optional[Sequence[Tuple[str, str]]],
    enable: bool,
) -> str:
    if not enable or not shots:
        return ""
    parts = []
    for q, a in shots:
        q = (q or "").strip()
        a = (a or "").strip().rstrip(";") + ";"
        if q and a:
            parts.append(f"Q: {q}\nA: {a}")
    return "\n\n".join(parts)


def _trim_schema(schema_text: str, max_chars: Optional[int]) -> str:
    if not schema_text:
        return ""
    if max_chars and max_chars > 0 and len(schema_text) > max_chars:
        return schema_text[:max_chars] + "\n-- (schema truncated) --"
    return schema_text


# -----------------------------
# Public API
# -----------------------------

def build_prompt(
    question: str,
    schema_snippet: str,
    *,
    fewshots: Optional[Sequence[Tuple[str, str]]] = None,
    extra_instructions: Optional[str] = None,
    max_schema_chars: Optional[int] = 12000,
) -> str:
    """
    Construct a strict Text-to-SQL prompt for MSSQL.

    Args:
        question: User's natural-language question.
        schema_snippet: Compact schema text (tables/columns/relationships). Keep it focused.
        fewshots: Optional list of (question, sql) examples. Defaults to _DEFAULT_FEWSHOTS if ENABLE_FEWSHOTS=true.
        extra_instructions: Optional extra guardrails (e.g., allowed tables).
        max_schema_chars: Trim schema to this many characters to stay within model context.

    Returns:
        Full prompt string ready for the LLM.
    """
    q = (question or "").strip()
    schema = _trim_schema(schema_snippet or "", max_schema_chars)

    system = _SYSTEM_RULES.format(ROW_LIMIT=_settings.SQL_ROW_LIMIT)

    # Choose few-shots
    use_shots: Optional[Sequence[Tuple[str, str]]] = fewshots
    if use_shots is None and _settings.ENABLE_FEWSHOTS:
        use_shots = _DEFAULT_FEWSHOTS

    shots_block = _format_fewshots(use_shots, enable=_settings.ENABLE_FEWSHOTS)

    prompt_parts: List[str] = [
        system,
        "",
        "Relevant database schema (tables, columns, relationships):",
        schema or "(no schema provided)",
        "",
    ]

    if extra_instructions:
        prompt_parts.extend(["Additional constraints:", extra_instructions.strip(), ""])

    if shots_block:
        prompt_parts.extend([shots_block, ""])

    prompt_parts.append(f"User question: {q}\nSQL:")

    return "\n".join(prompt_parts)


# Convenience default that uses the built-in few-shots
def build_default_prompt(question: str, schema_snippet: str) -> str:
    return build_prompt(question, schema_snippet, fewshots=_DEFAULT_FEWSHOTS)


# if __name__ == "__main__":
#     # quick self-check
#     demo_schema = """TABLE dbo.Banks:
#   - BankId (int)
#   - Name (nvarchar(200))
#   - STDCode (varchar(10))

# TABLE dbo.Branches:
#   - BranchId (int)
#   - BankId (int)
#   - City (nvarchar(100))

# Relationships:
#   dbo.Branches -> dbo.Banks
# """
#     demo = build_default_prompt("Count banks with STD code 079", demo_schema)
#     print(demo)
