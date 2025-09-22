# src/routes/quick_chats.py
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Body
from pydantic import BaseModel
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
from pathlib import Path
import threading
import uuid
import os
import json
import re
import time
import logging

# ---- Catalog & DB ----
from src.core.catalog_cache import (
    get_catalog,
    tables_used_in_question_hint,
    get_table_info,
)
from src.core.db import run_select
try:
    from src.core.db import test_connection  # optional
except Exception:  # pragma: no cover
    test_connection = None  # type: ignore

# ---- T2SQL plumbing ----
from src.core.select_candidates import pick_candidates, make_schema_snippet
from src.core.prompt_sql import build_default_prompt
from src.core.validate_sql import validate_sql
from src.core.settings import get_settings
from src.core.logging_utils import (
    Timer,
    log_text2sql_event,
    log_sql_execution,
    new_correlation_id,
)

# Providers (tolerant imports)
try:
    from src.core.text2sql import _post_ollama as ollama_generate  # type: ignore
except Exception:  # pragma: no cover
    ollama_generate = None  # type: ignore
try:
    from src.core.text2sql import _post_openai as openai_generate  # type: ignore
except Exception:  # pragma: no cover
    openai_generate = None  # type: ignore

settings = get_settings()
logger = logging.getLogger("app:t2sql")

router = APIRouter(prefix="/api/quick-chats", tags=["quick-chats"])

# =========================
# PERSISTENT JSON STORAGE
# =========================
_STORE_LOCK = threading.RLock()
_BASE_UPLOADS = Path(os.environ.get("UPLOADS_DIR", "uploads")).resolve()
_QC_BASE = _BASE_UPLOADS / "quick_chats"
_QC_MSG_DIR = _QC_BASE / "messages"
_QC_CHATS_PATH = _QC_BASE / "chats.json"
_QC_BASE.mkdir(parents=True, exist_ok=True)
_QC_MSG_DIR.mkdir(parents=True, exist_ok=True)

# Per-chat execution locks (prevents double model runs)
_CHAT_LOCKS: Dict[str, threading.Lock] = {}

def _chat_lock(chat_id: str) -> threading.Lock:
    with _STORE_LOCK:
        lock = _CHAT_LOCKS.get(chat_id)
        if lock is None:
            lock = threading.Lock()
            _CHAT_LOCKS[chat_id] = lock
        return lock

def _now_iso() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"

def _now_ms() -> int:
    return int(time.time() * 1000)

def _load_chats_from_disk() -> Dict[str, Dict]:
    if not _QC_CHATS_PATH.exists():
        return {}
    try:
        data = json.loads(_QC_CHATS_PATH.read_text("utf-8"))
        items = data.get("items") or []
        return {c["id"]: c for c in items if isinstance(c, dict) and "id" in c}
    except Exception:
        return {}

def _save_chats_to_disk(chats: Dict[str, Dict]) -> None:
    payload = {"items": list(chats.values())}
    tmp = _QC_CHATS_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(_QC_CHATS_PATH)

def _msg_path(chat_id: str) -> Path:
    return _QC_MSG_DIR / f"{chat_id}.json"

def _load_messages_from_disk(chat_id: str) -> List[Dict]:
    p = _msg_path(chat_id)
    if not p.exists():
        return []
    try:
        data = json.loads(p.read_text("utf-8"))
        return data.get("items") or []
    except Exception:
        return []

def _save_messages_to_disk(chat_id: str, msgs: List[Dict]) -> None:
    p = _msg_path(chat_id)
    payload = {"items": msgs}
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(p)

# --- In-memory cache (synced with disk) ---
CHATS: Dict[str, Dict] = _load_chats_from_disk()        # id -> {id,title,createdAt}
MESSAGES: Dict[str, List[Dict]] = {}                     # lazy-load per chat

DEFAULT_GREETING = (
    "Hello! I’m ready to help you for your queries.\n\n"
    "I can run data queries if you start with `sql:` or ask a question about your tables."
)

# ====== Per-chat recent turn state for idempotency ======
# _TURN_STATE[chat_id] = {"text_norm": str, "ts": int, "thinking_id": str}
_TURN_STATE: Dict[str, Dict[str, Any]] = {}
_DEDUPE_WINDOW_MS = 8000  # 8 seconds

# =========================
# MODELS
# =========================
class ChatCreate(BaseModel):
    seedPrompt: Optional[str] = None
    seedGreeting: Optional[bool] = True
    greetingText: Optional[str] = None

class ChatPatch(BaseModel):
    title: str

class SendMessage(BaseModel):
    text: Optional[str] = None
    # If the UI can send this (e.g., crypto.randomUUID()), backend dedup is perfect.
    clientMsgId: Optional[str] = None

# =========================
# Small helpers
# =========================
def _mk_id(prefix: str = "m") -> str:
    return f"{prefix}_{int(time.time() * 1000)}"

def _ensure_chat_loaded(chat_id: str, *, create_if_missing: bool = True) -> None:
    with _STORE_LOCK:
        if chat_id in CHATS:
            if chat_id not in MESSAGES:
                MESSAGES[chat_id] = _load_messages_from_disk(chat_id)
            return

        p = _msg_path(chat_id)
        if p.exists():
            created = datetime.utcfromtimestamp(p.stat().st_mtime).isoformat(timespec="seconds") + "Z"
            CHATS[chat_id] = {"id": chat_id, "title": "Recovered chat", "createdAt": created}
            _save_chats_to_disk(CHATS)
            MESSAGES[chat_id] = _load_messages_from_disk(chat_id)
            return

        if create_if_missing:
            created = _now_iso()
            CHATS[chat_id] = {"id": chat_id, "title": "New quick chat", "createdAt": created}
            _save_chats_to_disk(CHATS)
            MESSAGES[chat_id] = []
            _save_messages_to_disk(chat_id, [])

def _norm_text(t: str) -> str:
    return re.sub(r"\s+", " ", (t or "").strip().lower())

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
        "contact", "contacts", "phone", "mobile", "number", "numbers",
        "email", "mail", "whatsapp"
    ]
    return any(x in q for x in triggers)

def _is_sql_mode(t: str) -> bool:
    q = (t or "").strip().lower()
    return q.startswith("sql:") or q.startswith("db:") or q.startswith("data:")

# ---------- SQL extraction & fallback ----------
_SQL_FENCE = re.compile(r"```(?:sql)?\s*([\s\S]*?)```", re.I)

def _extract_sql_from_text(s: str) -> str:
    if not s:
        raise ValueError("Empty SQL from model")
    m = _SQL_FENCE.search(s)
    if m:
        s = m.group(1)
    s = re.sub(r"^\s*(?:sql|query)\s*:\s*", "", s, flags=re.I)
    lines = [ln for ln in s.splitlines() if not ln.strip().startswith("--")]
    s = "\n".join(lines).strip()
    m_with = re.search(r"(?is)^\s*with\b", s)
    m_sel  = re.search(r"(?is)\bselect\b", s)
    if m_with and (not m_sel or m_with.start() <= m_sel.start()):
        start = m_with.start()
    elif m_sel:
        start = m_sel.start()
    else:
        raise ValueError("No SELECT found in model output")
    stmt = s[start:].strip()
    semi = stmt.find(";")
    if semi != -1:
        stmt = stmt[:semi]
    return stmt.strip()

def _q_tbl_mssql(tbl: str) -> str:
    parts = [p for p in (tbl or "").split(".") if p]
    return ".".join(f"[{p.replace(']', ']]')}]" for p in parts) if parts else tbl

def _q_col_mssql(col: str) -> str:
    return f"[{(col or '').replace(']', ']]')}]"

# ---------- contact-aware helpers / union fallback ----------
_CONTACT_HINTS = {
    "contact", "contacts", "phone", "mobile", "mobile no", "mobile number",
    "phone no", "phone number", "whatsapp", "email", "e-mail", "mail",
    "number", "numbers", "call"
}
_NAME_HINTS = {"name", "full name", "fullname", "first name", "last name"}

def _needs_contact_columns(q: str) -> bool:
    ql = (q or "").lower()
    return any(h in ql for h in _CONTACT_HINTS | _NAME_HINTS)

def _sql_mentions_contact_cols(sql: str) -> bool:
    s = (sql or "").lower()
    return any(w in s for w in ("name", "phone", "mobile", "email"))

def _lower_keys(d: Dict[str, Any]) -> Dict[str, Any]:
    return {str(k).lower(): v for k, v in (d or {}).items()}

def _first_present_expr(cols_lc: Dict[str, Any], candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c.lower() in cols_lc:
            return _q_col_mssql(c)
    return None

def _best_name_expr(cols: Dict[str, Any]) -> Optional[str]:
    cols_lc = _lower_keys(cols)
    # FirstName + LastName
    if "firstname" in cols_lc and "lastname" in cols_lc:
        return "LTRIM(RTRIM(CONCAT([FirstName],' ',[LastName])))"
    # FullName / full_name
    for c in ("FullName", "full_name", "Name"):
        if c.lower() in cols_lc:
            return f"CAST({_q_col_mssql(c)} AS NVARCHAR(200))"
    # Single fallback to any 'name'-ish column
    for k in cols:
        if "name" in k.lower():
            return f"CAST({_q_col_mssql(k)} AS NVARCHAR(200))"
    return None

def _nonnull_condition(cols: Dict[str, Any], candidates: List[str]) -> str:
    parts = []
    for c in candidates:
        if c in cols:
            parts.append(f"{_q_col_mssql(c)} IS NOT NULL")
    return " OR ".join(parts) or "1=1"

def _fallback_sql_contacts_union(question: str) -> Optional[str]:
    """
    Build a UNION ALL across multiple tables with contact-like fields.
    Returns either:
      - SELECT SUM(cnt) ... (if 'count' in question), or
      - SELECT TOP 200 Name, Phone, Email, SourceTable ... UNION ALL ...
    """
    try:
        catalog = get_catalog()
    except Exception:
        return None

    # Pick candidate tables (expand a bit)
    cand_tables = tables_used_in_question_hint(question, top_k=12) or []
    # If hinting is empty, scan all known tables, but cap to avoid huge unions
    all_tables = list((catalog.get("tables") or {}).keys())
    if not cand_tables:
        cand_tables = all_tables[:20]

    selects: List[str] = []
    count_parts: List[str] = []

    for tbl in cand_tables:
        info = get_table_info(tbl) or {}
        cols_map: Dict[str, Any] = (info.get("columns") or {})
        if not cols_map:
            continue
        cols_lc = _lower_keys(cols_map)

        # Find contact-ish columns
        phone_col = _first_present_expr(cols_lc, ["Phone", "PhoneNumber", "Mobile", "MobileNo", "ContactNo", "ContactNumber"])
        email_col = _first_present_expr(cols_lc, ["Email", "EmailAddress", "Mail", "MailId", "E-mail"])
        name_expr = _best_name_expr(cols_map)

        # Skip tables that have none of the useful fields
        if not (phone_col or email_col or name_expr):
            continue

        name_sql = name_expr or "CAST(NULL AS NVARCHAR(200))"
        phone_sql = f"CAST({phone_col} AS NVARCHAR(200))" if phone_col else "CAST(NULL AS NVARCHAR(200))"
        email_sql = f"CAST({email_col} AS NVARCHAR(200))" if email_col else "CAST(NULL AS NVARCHAR(200))"

        q_tbl = _q_tbl_mssql(tbl)
        selects.append(
            f"SELECT TOP 50 {name_sql} AS Name, {phone_sql} AS Phone, {email_sql} AS Email, "
            f"CAST('{tbl}' AS NVARCHAR(128)) AS SourceTable FROM {q_tbl}"
        )

        # For counts, try to count only rows that look like contacts
        nnc = _nonnull_condition({k: True for k in cols_map.keys()},  # presence only
                                 [c for c in ["Phone","PhoneNumber","Mobile","MobileNo","Email","EmailAddress","Mail","MailId","E-mail","FullName","full_name","Name","FirstName","LastName"] if c in cols_map])
        count_parts.append(f"SELECT COUNT(1) AS cnt FROM {q_tbl} WHERE {nnc}")

    if not selects and not count_parts:
        return None

    ql = (question or "").lower()
    if "count" in ql:
        # Sum counts across tables
        return "SELECT SUM(cnt) AS TotalContacts FROM (\n  " + "\n  UNION ALL\n  ".join(count_parts) + "\n) x"

    # Otherwise list contacts across tables
    union = "\nUNION ALL\n".join(selects)
    return (
        "SELECT TOP 200 * FROM (\n"
        f"{union}\n"
        ") u"
    )

def _fallback_sql(question: str) -> Optional[str]:
    """
    Fallback builder:
      - If the question is contact-like → multi-table UNION
      - Otherwise → single-table light preview (previous behavior)
    """
    if _needs_contact_columns(question):
        return _fallback_sql_contacts_union(question)

    try:
        _ = get_catalog()
    except Exception:
        return None
    candidates = tables_used_in_question_hint(question, top_k=3)
    if not candidates:
        return None
    tbl = candidates[0]
    info = get_table_info(tbl) or {}
    cols_map = (info.get("columns") or {})
    cols = list(cols_map.keys())
    ql = (question or "").lower()
    if "count" in ql:
        return f"SELECT COUNT(*) AS count FROM {_q_tbl_mssql(tbl)}"
    prefer_names = ["FullName", "full_name", "Name", "FirstName", "LastName"]
    prefer_contact = ["Phone", "Mobile", "MobileNo", "PhoneNumber", "Email", "EmailAddress"]
    chosen: List[str] = []
    for p in prefer_names + prefer_contact:
        for c in cols:
            if c.lower() == p.lower() and c not in chosen:
                chosen.append(c)
    if not chosen:
        chosen = cols[: min(4, len(cols))] if cols else []
    if not chosen:
        return None
    cols_sql = ", ".join(_q_col_mssql(c) for c in chosen)
    return f"SELECT TOP 100 {cols_sql} FROM {_q_tbl_mssql(tbl)}"

def _require_db_and_catalog() -> None:
    try:
        if callable(test_connection):
            tc = test_connection()  # type: ignore
            if isinstance(tc, dict) and not tc.get("ok", True):
                raise RuntimeError(tc.get("error", "DB connectivity failed"))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database not reachable: {e}")
    try:
        get_catalog()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Catalog not ready: {e}")

def _generate_sql_via_model(user_text: str, schema_snippet: str) -> str:
    prompt = build_default_prompt(user_text, schema_snippet)
    provider = (getattr(settings, "TEXT_TO_SQL_PROVIDER", "ollama") or "ollama").lower()
    logger.info("t2sql.generate: provider=%s", provider)
    if provider == "openai":
        if not openai_generate:
            raise RuntimeError("OpenAI provider not available in this build.")
        return openai_generate(prompt)
    if provider == "ollama":
        if not ollama_generate:
            raise RuntimeError("Ollama provider not available in this build.")
        return ollama_generate(prompt)
    if ollama_generate:
        return ollama_generate(prompt)
    if openai_generate:
        return openai_generate(prompt)
    raise RuntimeError("No Text-to-SQL provider configured.")

# ---------- idempotency helpers ----------
def _existing_exchange_indices(
    msgs: List[Dict], text_norm: str, client_id: Optional[str]
) -> Tuple[Optional[int], Optional[int]]:
    """
    Find the last user message that matches this turn, and the first AI message after it.
    Matching priority:
      1) clientMsgId (if provided)
      2) normalized text
    Returns (user_idx, ai_idx_after_or_None)
    """
    user_idx = None
    # scan backwards for user match
    for i in range(len(msgs) - 1, -1, -1):
        m = msgs[i]
        if m.get("sender") != "user":
            continue
        if client_id and m.get("clientMsgId") == client_id:
            user_idx = i
            break
        if not client_id and _norm_text(m.get("text") or "") == text_norm:
            user_idx = i
            break
    if user_idx is None:
        return None, None
    ai_idx = None
    for j in range(user_idx + 1, len(msgs)):
        if msgs[j].get("sender") == "ai":
            ai_idx = j
            break
    return user_idx, ai_idx

# =========================
# Chat turn
# =========================
def _run_chat_turn_locked(chat_id: str, text: str, client_msg_id: Optional[str]) -> str:
    """Must be called with _chat_lock(chat_id) held."""
    t_norm = _norm_text(text)
    now = _now_ms()

    # Hard idempotency based on message list itself (works across refreshes)
    with _STORE_LOCK:
        msgs = MESSAGES.setdefault(chat_id, [])
        u_idx, ai_idx = _existing_exchange_indices(msgs, t_norm, client_msg_id)
        if u_idx is not None:
            # Already posted this message
            if ai_idx is not None:
                # Already answered → return that existing AI text
                return msgs[ai_idx].get("text") or ""
            else:
                # Already queued and has "Thinking…" → do not append another
                return "Still thinking…"

    # Soft idempotency (short window) to catch very fast double posts
    state = _TURN_STATE.get(chat_id) or {}
    if state and state.get("text_norm") == t_norm and (now - int(state.get("ts", 0))) < _DEDUPE_WINDOW_MS:
        # Duplicate call within window. If an AI answer already exists, return it.
        with _STORE_LOCK:
            msgs = MESSAGES.get(chat_id, [])
            for i in range(len(msgs) - 1, -1, -1):
                if msgs[i].get("sender") == "ai" and not msgs[i].get("thinking"):
                    return msgs[i].get("text") or ""
        return "Still thinking…"

    # Append user message
    with _STORE_LOCK:
        msgs = MESSAGES.setdefault(chat_id, [])
        msgs.append({
            "id": _mk_id("u"),
            "text": text,
            "sender": "user",
            "clientMsgId": client_msg_id,
            "at": _now_iso(),
            "atMs": now
        })
        _save_messages_to_disk(chat_id, msgs)

        # Append thinking marker as an AI message at the END
        thinking_id = _mk_id("a")
        msgs.append({
            "id": thinking_id,
            "text": "🧠 Thinking…",
            "sender": "ai",
            "thinking": True,
            "at": _now_iso(),
            "atMs": _now_ms()
        })
        _save_messages_to_disk(chat_id, msgs)

        # Update dedupe state
        _TURN_STATE[chat_id] = {"text_norm": t_norm, "ts": now, "thinking_id": thinking_id}

    # Generate answer
    reply_markdown = ""
    gen_ms = exec_ms = 0
    try:
        _require_db_and_catalog()
        tables, _debug = pick_candidates(text, top_k=6, expand_fks=True, neighbors_per_table=2)
        snippet = make_schema_snippet(tables)

        timer = Timer()
        model_out = _generate_sql_via_model(text, snippet)
        gen_ms = timer.ms
        logger.info("t2sql.model_out.first_line: %s",
                    (model_out or "").strip().splitlines()[0] if model_out else "EMPTY")

        try:
            sql_for_validation = _extract_sql_from_text(model_out)
        except Exception:
            sql_for_validation = _fallback_sql(text) or ""
            if not sql_for_validation:
                raise

        safe_sql = validate_sql(sql_for_validation, allowed_tables=tables, enforce_row_limit=True)

        # If the user asked for contacts/phones but the model omitted those cols,
        # force the deterministic multi-table fallback.
        if _needs_contact_columns(text) and not _sql_mentions_contact_cols(safe_sql):
            raise ValueError("Model missed contact columns; triggering multi-table fallback")

        logger.info("t2sql.sql.validated: %s", safe_sql.replace("\n", " "))

        timer = Timer()
        rows = run_select(safe_sql)
        exec_ms = timer.ms

        table_md, row_count = _format_markdown_table(rows, max_rows=20)
        reply_markdown = f"**SQL executed:**\n```sql\n{safe_sql}\n```\n\n**Result:**\n{table_md}"

        try:
            log_text2sql_event(text, picked_tables=tables, sql=safe_sql, ok=True, duration_ms=gen_ms)
            log_sql_execution(safe_sql, row_count=row_count, duration_ms=exec_ms, tables=tables, ok=True)
        except Exception:
            pass

    except Exception as e:
        try:
            fb_sql = _fallback_sql(text)
            if fb_sql:
                safe_sql = validate_sql(fb_sql, allowed_tables=None, enforce_row_limit=True)
                rows = run_select(safe_sql)
                table_md, _ = _format_markdown_table(rows, max_rows=20)
                reply_markdown = f"**SQL executed (fallback):**\n```sql\n{safe_sql}\n```\n\n**Result:**\n{table_md}"
            else:
                raise
        except Exception:
            reply_markdown = f"Failed to run query. {type(e).__name__}: {e}"
        try:
            log_text2sql_event(text, picked_tables=[], sql="", ok=False, error=str(e))
        except Exception:
            pass

    # Replace thinking with final message (in-place)
    with _STORE_LOCK:
        msgs = MESSAGES.setdefault(chat_id, [])
        th_id = _TURN_STATE.get(chat_id, {}).get("thinking_id")
        replaced = False
        if th_id:
            for i in range(len(msgs) - 1, -1, -1):
                if msgs[i].get("id") == th_id:
                    msgs[i] = {
                        "id": th_id,
                        "text": reply_markdown,
                        "sender": "ai",
                        "thinking": False,
                        "at": _now_iso(),
                        "atMs": _now_ms()
                    }
                    replaced = True
                    break
        if not replaced:
            msgs.append({
                "id": _mk_id("a"),
                "text": reply_markdown,
                "sender": "ai",
                "thinking": False,
                "at": _now_iso(),
                "atMs": _now_ms()
            })
        _save_messages_to_disk(chat_id, msgs)

    return reply_markdown

def _run_chat_turn(chat_id: str, text: Optional[str], client_msg_id: Optional[str]) -> str:
    _ensure_chat_loaded(chat_id, create_if_missing=True)
    t = (text or "").strip()
    if not t:
        return ""  # no-op turn; do NOT mutate messages
    try:
        new_correlation_id()
    except Exception:
        pass
    lock = _chat_lock(chat_id)
    with lock:
        return _run_chat_turn_locked(chat_id, t, client_msg_id)

# =========================
# Endpoints
# =========================
@router.get("")
def list_quick_chats():
    with _STORE_LOCK:
        chats = _load_chats_from_disk()
        CHATS.clear()
        CHATS.update(chats)
        items = sorted(CHATS.values(), key=lambda c: c["createdAt"], reverse=True)
        enriched = [{**it, "chatId": it["id"]} for it in items]
        return {"items": enriched}

@router.get("/ids")
def list_chat_ids():
    with _STORE_LOCK:
        return {"ids": list(CHATS.keys())}

@router.get("/{chat_id}")
def get_quick_chat(chat_id: str):
    with _STORE_LOCK:
        _ensure_chat_loaded(chat_id, create_if_missing=True)
        chat = CHATS[chat_id]
        return {**chat, "chatId": chat_id}

@router.post("", status_code=201)
def create_quick_chat(body: ChatCreate):
    cid = str(uuid.uuid4())
    created = _now_iso()
    title = (body.seedPrompt or "New quick chat")[:64]
    greeting = (body.greetingText or DEFAULT_GREETING).strip()

    with _STORE_LOCK:
        CHATS[cid] = {"id": cid, "title": title, "createdAt": created}
        _save_chats_to_disk(CHATS)

        msgs: List[Dict] = []
        if greeting:
            msgs.append({
                "id": _mk_id("a"),
                "text": greeting,
                "sender": "ai",
                "thinking": False,
                "at": created,
                "atMs": _now_ms()
            })
        if body.seedPrompt:
            msgs.append({
                "id": _mk_id("u"),
                "text": body.seedPrompt,
                "sender": "user",
                "at": created,
                "atMs": _now_ms()
            })
        MESSAGES[cid] = msgs
        _save_messages_to_disk(cid, msgs)

    return {"id": cid, "chatId": cid, "title": title, "createdAt": created}

@router.patch("/{chat_id}")
def rename_quick_chat(chat_id: str, patch: ChatPatch):
    with _STORE_LOCK:
        _ensure_chat_loaded(chat_id, create_if_missing=True)
        chat = CHATS[chat_id]
        new_title = (patch.title or "").strip()
        if new_title:
            chat["title"] = new_title[:200]
            _save_chats_to_disk(CHATS)
        return {"ok": True, "chatId": chat_id, "title": chat["title"]}

@router.get("/{chat_id}/messages}")
def _bad_path_guard(chat_id: str):  # just in case someone hits wrong route
    return get_messages(chat_id)

@router.get("/{chat_id}/messages")
def get_messages(chat_id: str):
    with _STORE_LOCK:
        _ensure_chat_loaded(chat_id, create_if_missing=True)
        msgs = MESSAGES.get(chat_id)
        if msgs is None:
            msgs = _load_messages_from_disk(chat_id)
            MESSAGES[chat_id] = msgs
        # Return a stable sorted view (no mutation), prevents UI from showing re-ordered duplicates
        items = sorted(msgs, key=lambda m: (int(m.get("atMs") or 0), str(m.get("id") or "")))
        return {
            "chatId": chat_id,
            "title": CHATS[chat_id]["title"],
            "createdAt": CHATS[chat_id]["createdAt"],
            "items": items,
        }

@router.post("/{chat_id}")
def run_chat_turn_endpoint(chat_id: str):
    # IMPORTANT: No-op on refresh/mount. Does not mutate messages.
    _ensure_chat_loaded(chat_id, create_if_missing=True)
    return {"ok": True, "chatId": chat_id, "reply": ""}

@router.post("/{chat_id}/messages")
def send_message(chat_id: str, payload: Optional[SendMessage] = Body(default=None)):
    text = (payload.text if payload else None) or ""
    client_id = payload.clientMsgId if payload else None
    reply = _run_chat_turn(chat_id, text=text, client_msg_id=client_id)
    return {"ok": True, "chatId": chat_id, "reply": reply}

@router.delete("/{chat_id}")
def delete_chat(chat_id: str):
    with _STORE_LOCK:
        try:
            _msg_path(chat_id).unlink(missing_ok=True)  # py>=3.8
        except Exception:
            pass
        if chat_id in CHATS:
            CHATS.pop(chat_id, None)
            _save_chats_to_disk(CHATS)
        MESSAGES.pop(chat_id, None)
        _TURN_STATE.pop(chat_id, None)
    return {"ok": True, "chatId": chat_id}
