import re

BAD_TOKENS = [
    ";", "--", "/*", "*/",
    " drop ", " delete ", " update ", " insert ",
    " alter ", " create ", " grant ", " revoke ", " truncate ",
]

def is_safe_select(sql: str, allowed_tables: set[str]) -> bool:
    s = sql.strip().lower()
    if not s.startswith("select"):
        return False
    for b in BAD_TOKENS:
        if b in s:
            return False
    # naive table extraction from FROM/JOIN
    parts = re.findall(r'(?:from|join)\s+([a-zA-Z0-9_".]+)', s)
    used = {p.replace('"','').split()[0] for p in parts}
    return used.issubset({t.lower() for t in allowed_tables})

def single_statement(sql: str) -> bool:
    # ban multiple statements quickly
    return sql.count(";") == 0
