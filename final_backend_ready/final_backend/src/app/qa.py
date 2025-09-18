from typing import Optional, Tuple
import re
from .db import reflect_schema, run_select
from .sqlgen import generate_sql
from .sqlguard import is_safe_select, single_statement

def _short_table(rows: list[dict]) -> str:
    if not rows:
        return "No matching rows found."
    cols = list(rows[0].keys())
    head = rows[:10]
    header = " | ".join(cols)
    sep = "-" * (sum(len(c) for c in cols) + 3*(len(cols)-1))
    lines = [header, sep]
    for r in head:
        lines.append(" | ".join("" if r[c] is None else str(r[c]) for c in cols))
    if len(rows) > len(head):
        lines.append(f"... and {len(rows)-len(head)} more rows.")
    return "\n".join(lines)

# very light rule for COUNT
def _rule_based(question: str, schema: dict[str, list[str]]) -> Optional[Tuple[str, dict]]:
    q = question.lower()
    m = re.search(r"(how many|count|kitne)\s+([a-z0-9_]+)", q)
    if m:
        noun = m.group(2)  # try to map noun to a table
        for t in schema:
            if noun in (t.lower(), t.lower().rstrip('s'), t.lower()+'s'):
                return f"SELECT COUNT(*) AS count FROM {t}", {}
    return None

async def answer_from_db(question: str) -> dict:
    schema = await reflect_schema()
    allowed_tables = set(schema.keys())

    # 1) tiny rule-based first
    rb = _rule_based(question, schema)
    sql, params = (rb if rb else (None, None))

    # 2) LLM provider (optional)
    if not sql:
        sql = await generate_sql(question, schema)
        params = {}

    if not sql:
        return {"ok": True, "source": "db", "reply": "Sorry, I can't answer this from the database schema."}

    sql = sql.strip()
    if not single_statement(sql) or not is_safe_select(sql, allowed_tables):
        return {"ok": True, "source": "db", "reply": "Generated SQL blocked by safety rules."}

    rows = await run_select(sql, params or {})
    # smart presentation
    if rows and "count" in rows[0] and len(rows[0]) == 1:
        return {"ok": True, "source": "db", "reply": f"Count: {rows[0]['count']}", "rows": rows, "sql": sql}
    return {"ok": True, "source": "db", "reply": _short_table(rows), "rows": rows, "sql": sql}
