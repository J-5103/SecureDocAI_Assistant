from .config import settings
from typing import Optional
import httpx

SYSTEM_PROMPT = """You are a Text-to-SQL generator.
Return ONLY one SQL SELECT statement to answer the user's question.
Rules:
- Use ONLY the provided schema (tables and columns).
- Read-only: SELECT only. No comments, semicolons, DDL or DML.
- Prefer equality/ILIKE/LIKE and safe joins.
- If impossible from schema, answer exactly: NONE
"""

def build_schema_text(schema: dict[str, list[str]], allow_tables: set[str] | None = None) -> str:
    lines = []
    for t, cols in schema.items():
        if allow_tables and t not in allow_tables:
            continue
        lines.append(f"TABLE {t} (")
        for c in cols:
            lines.append(f"  {c}")
        lines.append(")")
    return "\n".join(lines)

def keyword_candidates(question: str, schema: dict[str, list[str]], top_k: int = 6) -> set[str]:
    q = question.lower()
    scores = []
    for t, cols in schema.items():
        score = 0
        if t.lower() in q:
            score += 3
        for c in cols:
            if c.lower() in q:
                score += 1
        scores.append((score, t))
    scores.sort(reverse=True)
    chosen = [t for s, t in scores if s > 0][:top_k]
    return set(chosen) if chosen else set(schema.keys())  # fallback all schema (small)

async def gen_sql_from_openai(question: str, schema_text: str) -> Optional[str]:
    from openai import OpenAI
    if not settings.OPENAI_API_KEY:
        return None
    client = OpenAI(api_key=settings.OPENAI_API_KEY)
    user = f"Schema:\n{schema_text}\n\nQuestion: {question}\n\nSQL:"
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[{"role":"system","content":SYSTEM_PROMPT},
                  {"role":"user","content":user}]
    )
    sql = resp.choices[0].message.content.strip()
    if sql.upper() == "NONE":
        return None
    return sql

async def gen_sql_from_ollama(question: str, schema_text: str) -> Optional[str]:
    prompt = f"{SYSTEM_PROMPT}\n\nSchema:\n{schema_text}\n\nQuestion: {question}\n\nSQL:"
    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(
            f"{settings.OLLAMA_HOST}/api/generate",
            json={"model": settings.OLLAMA_MODEL, "prompt": prompt, "stream": False, "options":{"temperature":0}},
        )
    r.raise_for_status()
    text = r.json().get("response", "").strip()
    if text.upper() == "NONE":
        return None
    # remove code fences if any
    text = text.replace("```sql","").replace("```","").strip()
    return text

async def generate_sql(question: str, schema: dict[str, list[str]]) -> Optional[str]:
    allow = keyword_candidates(question, schema)
    schema_text = build_schema_text(schema, allow)
    provider = (settings.TEXT_TO_SQL_PROVIDER or "none").lower()

    if provider == "openai":
        return await gen_sql_from_openai(question, schema_text)
    if provider == "ollama":
        return await gen_sql_from_ollama(question, schema_text)
    # provider none => no LLM
    return None
