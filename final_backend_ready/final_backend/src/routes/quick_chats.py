# src/routes/quick_chats.py
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Body
from pydantic import BaseModel
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import uuid

# ---- Core T2SQL plumbing (optional; comment out if not using yet) ----
from src.core.select_candidates import pick_candidates, make_schema_snippet
from src.core.prompt_sql import build_default_prompt
from src.core.text2sql import _post_ollama as ollama_generate
from src.core.validate_sql import validate_sql
from src.core.db import run_select
from src.core.settings import get_settings
from src.core.logging_utils import (
    Timer,
    log_text2sql_event,
    log_sql_execution,
    new_correlation_id,
)

settings = get_settings()

router = APIRouter(prefix="/api/quick-chats", tags=["quick-chats"])

# --- In-memory store (process lifetime only) ---
CHATS: Dict[str, Dict] = {}            # id -> {id,title,createdAt}
MESSAGES: Dict[str, List[Dict]] = {}    # id -> [{id,text,sender,at}]

DEFAULT_GREETING = "Hello! I’m ready to help. Ask anything—counts, lists, or quick insights."

def now_iso() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"

# ---- Models ----
class ChatCreate(BaseModel):
    seedPrompt: Optional[str] = None
    # We always insert a greeting; seedGreeting kept for backward compat.
    seedGreeting: Optional[bool] = True
    greetingText: Optional[str] = None

class ChatPatch(BaseModel):
    title: str

class SendMessage(BaseModel):
    # Make optional so empty body doesn't 422
    text: Optional[str] = None

# ---- Helpers ----
def _mk_id(prefix: str = "m") -> str:
    return f"{prefix}_{int(datetime.utcnow().timestamp() * 1000)}"

def _format_markdown_table(rows: List[Dict], max_rows: int = 20) -> Tuple[str, int]:
    if not rows:
        return "_No rows found._", 0
    headers = list(rows[0].keys())
    md = []
    md.append("| " + " | ".join(headers) + " |")
    md.append("|" + "|".join(["---"] * len(headers)) + "|")
    for r in rows[:max_rows]:
        md.append("| " + " | ".join(str(r.get(h, "")) for h in headers) + " |")
    extra = len(rows) - min(len(rows), max_rows)
    if extra > 0:
        md.append(f"\n…showing {max_rows} of {len(rows)} rows.")
    return "\n".join(md), len(rows)

def _looks_like_data_question(t: str) -> bool:
    q = (t or "").lower()
    triggers = [
        "count", "list", "top", "sum", "average", "avg", "min", "max",
        "where", "group", "order", "between", "transactions", "branches",
        "customers", "banks", "loans", "report", "trend", "total", "per ",
        "contact", "contacts", "phone", "mobile", "number", "numbers"
    ]
    return any(x in q for x in triggers)

def _is_sql_mode(t: str) -> bool:
    q = (t or "").strip().lower()
    return q.startswith("sql:") or q.startswith("db:") or q.startswith("data:")

# ---- Shared turn executor ----
def _run_chat_turn(chat_id: str, text: Optional[str]) -> str:
    """
    Executes a chat turn. If `text` is None/empty, we do a noop-style reply
    (so the API works with 'chatId only' calls).
    When text looks like a data question, we run the text-to-SQL pipeline.
    Returns reply string (markdown allowed).
    """
    if chat_id not in CHATS:
        raise HTTPException(status_code=404, detail="Chat not found")

    t = (text or "").strip()  # <-- FIX: Python uses .strip(), not .trim()

    # correlation id (best-effort)
    try:
        new_correlation_id()
    except Exception:
        pass

    # If user sent text, store it
    if t:
        ts = now_iso()
        MESSAGES[chat_id].append({"id": _mk_id("u"), "text": t, "sender": "user", "at": ts})

    # If no text, produce a lightweight reply and exit (supports chatId-only POST)
    if not t:
        reply = "Ready. Send a question (e.g., *Count banks with STD code 079*)."
        MESSAGES[chat_id].append({"id": _mk_id("a"), "text": reply, "sender": "ai", "at": now_iso()})
        return reply

    # Decide mode
    force_sql = _is_sql_mode(t)
    text_for_sql = t.split(":", 1)[1].strip() if force_sql and ":" in t else t
    run_sql = force_sql or _looks_like_data_question(t)

    if not run_sql:
        reply = "I can run data queries if you start with `sql:` or ask a question about your tables."
        MESSAGES[chat_id].append({"id": _mk_id("a"), "text": reply, "sender": "ai", "at": now_iso()})
        return reply

    # ---- Text-to-SQL pipeline (optional; requires DB) ----
    reply_markdown = ""
    timer = Timer()
    try:
        tables, _debug = pick_candidates(text_for_sql, top_k=6, expand_fks=True, neighbors_per_table=2)
        snippet = make_schema_snippet(tables)

        prompt = build_default_prompt(text_for_sql, snippet)
        gen_out = ollama_generate(prompt)  # full text
        gen_ms = timer.ms

        safe_sql = validate_sql(gen_out, allowed_tables=tables, enforce_row_limit=True)

        timer = Timer()
        rows = run_select(safe_sql)
        exec_ms = timer.ms

        table_md, row_count = _format_markdown_table(rows, max_rows=20)
        reply_markdown = f"**SQL executed:**\n```sql\n{safe_sql}\n```\n\n**Result:**\n{table_md}"

        try:
            log_text2sql_event(text_for_sql, picked_tables=tables, sql=safe_sql, ok=True, duration_ms=gen_ms)
            log_sql_execution(safe_sql, row_count=row_count, duration_ms=exec_ms, tables=tables, ok=True)
        except Exception:
            pass

    except Exception as e:
        try:
            log_text2sql_event(text_for_sql, picked_tables=[], sql="", ok=False, error=str(e))
        except Exception:
            pass
        reply_markdown = f"Failed to run query. {type(e).__name__}: {e}"

    MESSAGES[chat_id].append({"id": _mk_id("a"), "text": reply_markdown, "sender": "ai", "at": now_iso()})
    return reply_markdown

# ---- Endpoints ----
@router.get("")
def list_quick_chats():
    items = sorted(CHATS.values(), key=lambda c: c["createdAt"], reverse=True)
    enriched = [{**it, "chatId": it["id"]} for it in items]
    return {"items": enriched}

@router.get("/ids")
def list_chat_ids():
    return {"ids": list(CHATS.keys())}

@router.get("/{chat_id}")
def get_quick_chat(chat_id: str):
    chat = CHATS.get(chat_id)
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")
    return {**chat, "chatId": chat_id}

@router.post("", status_code=201)
def create_quick_chat(body: ChatCreate):
    cid = str(uuid.uuid4())
    created = now_iso()
    title = (body.seedPrompt or "New quick chat")[:64]

    CHATS[cid] = {"id": cid, "title": title, "createdAt": created}
    MESSAGES[cid] = []

    # Always insert a welcome message
    greeting = (body.greetingText or DEFAULT_GREETING).strip()
    if greeting:
        MESSAGES[cid].append({
            "id": _mk_id("a"),
            "text": greeting,
            "sender": "ai",
            "at": created
        })

    # Optional first user prompt
    if body.seedPrompt:
        MESSAGES[cid].append({
            "id": _mk_id("u"),
            "text": body.seedPrompt,
            "sender": "user",
            "at": created
        })

    return {"id": cid, "chatId": cid, "title": title, "createdAt": created}

@router.patch("/{chat_id}")
def rename_quick_chat(chat_id: str, patch: ChatPatch):
    chat = CHATS.get(chat_id)
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")
    new_title = (patch.title or "").strip()
    if new_title:
        chat["title"] = new_title[:200]
    return {"ok": True, "chatId": chat_id, "title": chat["title"]}

@router.get("/{chat_id}/messages")
def get_messages(chat_id: str):
    if chat_id not in CHATS:
        raise HTTPException(status_code=404, detail="Chat not found")
    return {
        "chatId": chat_id,
        "title": CHATS[chat_id]["title"],
        "createdAt": CHATS[chat_id]["createdAt"],
        "items": MESSAGES.get(chat_id, []),
    }

# NEW: Support chat turn with ONLY chatId (empty body) → fixes 405
@router.post("/{chat_id}")
def run_chat_turn(chat_id: str):
    reply = _run_chat_turn(chat_id, text=None)
    return {"ok": True, "chatId": chat_id, "reply": reply}

# LEGACY/FALLBACK: tolerate empty body → fixes 422
@router.post("/{chat_id}/messages")
def send_message(chat_id: str, payload: Optional[SendMessage] = Body(default=None)):
    text = payload.text if payload else None
    reply = _run_chat_turn(chat_id, text=text)
    return {"ok": True, "chatId": chat_id, "reply": reply}

@router.delete("/{chat_id}")
def delete_chat(chat_id: str):
    if chat_id not in CHATS:
        raise HTTPException(status_code=404, detail="Chat not found")
    CHATS.pop(chat_id, None)
    MESSAGES.pop(chat_id, None)
    return {"ok": True, "chatId": chat_id}
