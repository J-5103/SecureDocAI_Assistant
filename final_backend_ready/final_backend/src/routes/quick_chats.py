# src/routes/quick_chats.py
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Body
from pydantic import BaseModel
from typing import Dict, List, Tuple, Optional, Any, Set
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
from src.core.validate_sql import validate_sql, extract_tables
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

def _clean_cell(val: Any, *, limit: int = 160) -> str:
    s = "" if val is None else str(val)
    s = s.replace("\r", " ").replace("\n", " ").replace("\t", " ").strip()
    if len(s) > limit:
        s = s[:limit - 1] + "…"
    return s

def _format_markdown_table(rows: List[Dict], max_rows: int = 60) -> Tuple[str, int]:
    if not rows:
        return "_No rows found._", 0
    headers = list(rows[0].keys())
    md = []
    md.append("| " + " | ".join(headers) + " |")
    md.append("|" + "|".join(["---"] * len(headers)) + "|")
    for r in rows[:max_rows]:
        md.append("| " + " | ".join(_clean_cell(r.get(h, "")) for h in headers) + " |")
    extra = len(rows) - min(len(rows), max_rows)
    if extra > 0:
        md.append(f"\n…showing {max_rows} of {len(rows)} rows.")
    return "\n".join(md), len(rows)

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

# --- existence check to avoid "Invalid object name ..." ---
def _table_exists(tbl: str) -> bool:
    try:
        run_select(f"SELECT TOP 0 1 FROM {_q_tbl_mssql(tbl)}")
        return True
    except Exception:
        return False

# ---------- helpers for “contact-ish” union ----------
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

def _format_grouped_by_source(
    rows: List[Dict],
    value_col: str,
    source_table_col: str = "SourceTable",
    source_col_col: str = "SourceColumn",
    max_rows_per_group: int = 60,
) -> Tuple[str, int]:
    if not rows:
        return "_No rows found._", 0
    groups: Dict[str, List[Dict]] = {}
    for r in rows:
        tbl = str(r.get(source_table_col, "") or "")
        groups.setdefault(tbl, []).append(r)

    parts: List[str] = []
    total = 0
    for tbl in sorted(groups.keys()):
        part_rows = groups[tbl]
        total += len(part_rows)
        header = f"**{tbl}**"
        parts.append(header)
        md, _ = _format_markdown_table(
            [{value_col: r.get(value_col), source_col_col: r.get(source_col_col)} for r in part_rows],
            max_rows=max_rows_per_group,
        )
        parts.append(md)
        parts.append("")  # spacing
    return "\n\n".join(parts).strip(), total

def _best_name_expr(cols: Dict[str, Any]) -> Optional[str]:
    cols_lc = _lower_keys(cols)
    for c in ("Name", "FullName", "full_name"):
        if c.lower() in cols_lc:
            return f"NULLIF(LTRIM(RTRIM(CAST({_q_col_mssql(c)} AS NVARCHAR(200)))),'')"
    if "firstname" in cols_lc and "lastname" in cols_lc:
        return "NULLIF(LTRIM(RTRIM(CONCAT([FirstName],' ',[LastName]))), '')"
    for k in cols:
        if "name" in k.lower():
            return f"NULLIF(LTRIM(RTRIM(CAST({_q_col_mssql(k)} AS NVARCHAR(200)))),'')"
    return None

def _nonnull_condition(cols: Dict[str, Any], candidates: List[str]) -> str:
    parts = []
    for c in candidates:
        if c in cols:
            parts.append(f"{_q_col_mssql(c)} IS NOT NULL")
    return " OR ".join(parts) or "1=1"

# ---------- fuzzy matchers ----------
def _normkey(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())

def _col_matches_phone(col: str) -> bool:
    k = _normkey(col)
    if "contact" in k:
        return True  # contactnumber / contactno / contact etc.
    return (
        "phone" in k or "mobil" in k or "whatsapp" in k or
        k.endswith("phoneno") or k.endswith("mobileno") or
        k in {"phone", "phoneno", "phone1", "phone2", "mobile", "mobile1", "mobile2", "contactnumber"}
    )

def _col_matches_email(col: str) -> bool:
    k = _normkey(col)
    if "body" in k:
        return False
    return ("email" in k) or ("mail" in k)

# ---------- metric & date helpers (NEW) ----------
_METRIC_SYNONYMS = {
    "revenue","sale","sales","amount","amt","price","cost","charge","fee",
    "value","subtotal","total","grandtotal","net","gross","tax","discount",
    "qty","quantity","volume"
}

def _col_matches_revenueish(col: str, needle: Optional[str]) -> bool:
    k = _normkey(col)
    if needle:
        n = _normkey(needle)
        if n and n in k:
            return True
    # avoid email/phone columns
    if _col_matches_phone(col) or _col_matches_email(col):
        return False
    return any(s in k for s in _METRIC_SYNONYMS)

_DATE_CANDIDATES = (
    "date","dt","created","createdat","created_on","orderdate","invoicedate",
    "timestamp","ts","postdate","posted","entrydate","updated","modified"
)

def _pick_date_column(cols_map: Dict[str, Any]) -> Optional[str]:
    for c in cols_map.keys():
        k = _normkey(c)
        if any(tok in k for tok in _DATE_CANDIDATES):
            return c
    return None

# Parse coarse time windows from free-text
_DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")
def _parse_time_window(q: str) -> Dict[str, Any]:
    s = (q or "").lower()
    # explicit range
    dts = _DATE_RE.findall(s)
    if len(dts) >= 2:
        return {"type": "between", "start": dts[0], "end": dts[1]}
    if "from" in s and "to" in s and len(dts) == 2:
        return {"type": "between", "start": dts[0], "end": dts[1]}
    if "since" in s and len(dts) >= 1:
        return {"type": "since", "start": dts[0]}
    # relative windows
    if "today" in s:
        return {"type": "today"}
    if "yesterday" in s:
        return {"type": "yesterday"}
    if "this month" in s or "current month" in s:
        return {"type": "this_month"}
    if "last month" in s or "previous month" in s:
        return {"type": "last_month"}
    if "this year" in s or "current year" in s:
        return {"type": "this_year"}
    if "last year" in s or "previous year" in s:
        return {"type": "last_year"}
    return {}

def _date_predicate_for(date_col: Optional[str], window: Dict[str, Any]) -> str:
    if not date_col or not window:
        return ""
    c = _q_col_mssql(date_col)
    t = window.get("type")
    if t == "between":
        return f"{c} >= CAST('{window['start']}' AS date) AND {c} < DATEADD(day, 1, CAST('{window['end']}' AS date))"
    if t == "since":
        return f"{c} >= CAST('{window['start']}' AS date)"
    if t == "today":
        return f"{c} >= CONVERT(date, GETDATE()) AND {c} < DATEADD(day, 1, CONVERT(date, GETDATE()))"
    if t == "yesterday":
        return f"{c} >= DATEADD(day, -1, CONVERT(date, GETDATE())) AND {c} < CONVERT(date, GETDATE())"
    if t == "this_month":
        return ("{c} >= DATEFROMPARTS(YEAR(GETDATE()), MONTH(GETDATE()), 1) "
                "AND {c} < DATEADD(month, 1, DATEFROMPARTS(YEAR(GETDATE()), MONTH(GETDATE()), 1))").format(c=c)
    if t == "last_month":
        return ("{c} >= DATEADD(month, -1, DATEFROMPARTS(YEAR(GETDATE()), MONTH(GETDATE()), 1)) "
                "AND {c} < DATEFROMPARTS(YEAR(GETDATE()), MONTH(GETDATE()), 1)").format(c=c)
    if t == "this_year":
        return ("{c} >= DATEFROMPARTS(YEAR(GETDATE()), 1, 1) "
                "AND {c} < DATEADD(year, 1, DATEFROMPARTS(YEAR(GETDATE()), 1, 1))").format(c=c)
    if t == "last_year":
        return ("{c} >= DATEADD(year, -1, DATEFROMPARTS(YEAR(GETDATE()), 1, 1)) "
                "AND {c} < DATEFROMPARTS(YEAR(GETDATE()), 1, 1)").format(c=c)
    return ""

# Parse metric intent (sum/avg/min/max/count + metric word)
def _parse_metric_request(q: str) -> Optional[Dict[str, Any]]:
    s = (q or "").lower()
    kind = None
    if any(k in s for k in ("avg ", "average", "mean ")):
        kind = "avg"
    elif any(k in s for k in ("min ", "minimum", "lowest", "smallest")):
        kind = "min"
    elif any(k in s for k in ("max ", "maximum", "highest", "largest")):
        kind = "max"
    elif "count" in s and "distinct" not in s:
        kind = "count"
    elif "sum" in s or "total" in s:
        kind = "sum"
    # Try to pick a needle (metric name) e.g. revenue/amount/price
    m = re.search(r"(revenue|amount|price|sales?|subtotal|grand\s*total|net|gross|value|cost|fee|charge|qty|quantity)", s)
    needle = m.group(1) if m else ""
    if not kind and needle:
        kind = "sum"
    if not kind:
        return None
    return {"kind": kind, "needle": needle, "window": _parse_time_window(q)}

# Single-number metric across all tables (sum/avg/min/max)
def _fallback_metric_union(question: str) -> Optional[str]:
    req = _parse_metric_request(question)
    if not req or req["kind"] == "count":
        return None  # 'count' ke liye niche _fallback_count_union use hota hai
    try:
        catalog = get_catalog()
    except Exception:
        return None

    all_tables = list((catalog.get("tables") or {}).keys())
    pieces: List[str] = []
    pieces_avg: List[str] = []
    window = req.get("window") or {}
    needle = req.get("needle") or ""

    for tbl in all_tables:
        if not _table_exists(tbl):
            continue
        info = get_table_info(tbl) or {}
        cols_map: Dict[str, Any] = (info.get("columns") or {})
        if not cols_map:
            continue

        # pick metric cols
        metric_cols = [c for c in cols_map.keys() if _col_matches_revenueish(c, needle if needle else None)]
        if not metric_cols:
            continue

        date_col = _pick_date_column(cols_map)
        pred = _date_predicate_for(date_col, window)
        where_clause = f"WHERE {pred}" if pred else ""
        t = _q_tbl_mssql(tbl)

        for mc in metric_cols:
            qmc = _q_col_mssql(mc)
            if req["kind"] == "sum":
                pieces.append(f"SELECT SUM(TRY_CONVERT(decimal(38,6), {qmc})) AS v FROM {t} {where_clause}")
            elif req["kind"] == "min":
                pieces.append(f"SELECT MIN(TRY_CONVERT(decimal(38,6), {qmc})) AS v FROM {t} {where_clause}")
            elif req["kind"] == "max":
                pieces.append(f"SELECT MAX(TRY_CONVERT(decimal(38,6), {qmc})) AS v FROM {t} {where_clause}")
            elif req["kind"] == "avg":
                pieces_avg.append(
                    "SELECT SUM(TRY_CONVERT(decimal(38,6), {c})) AS s, "
                    "SUM(CASE WHEN TRY_CONVERT(decimal(38,6), {c}) IS NOT NULL THEN 1 ELSE 0 END) AS n "
                    "FROM {t} {w}".format(c=qmc, t=t, w=where_clause)
                )

    if req["kind"] == "avg" and pieces_avg:
        inner = " \n  UNION ALL\n  ".join(pieces_avg)
        return "SELECT CAST(SUM(s)/NULLIF(SUM(n),0) AS decimal(38,2)) AS Average FROM (\n  " + inner + "\n) x"
    if pieces and req["kind"] == "sum":
        inner = " \n  UNION ALL\n  ".join(pieces)
        return "SELECT CAST(SUM(v) AS decimal(38,2)) AS Total FROM (\n  " + inner + "\n) x"
    if pieces and req["kind"] == "min":
        inner = " \n  UNION ALL\n  ".join(pieces)
        return "SELECT MIN(v) AS MinValue FROM (\n  " + inner + "\n) x"
    if pieces and req["kind"] == "max":
        inner = " \n  UNION ALL\n  ".join(pieces)
        return "SELECT MAX(v) AS MaxValue FROM (\n  " + inner + "\n) x"
    return None

# ---------- Grouped totals: "… by region/state/product/country" (NEW) ----------
_BY_TOKEN_RE = re.compile(r"\bby\s+([a-z0-9_ ,\-]+)", re.I)
_WISE_RE = re.compile(r"\b([a-z0-9_]+)\s*[- ]?wise\b", re.I)

def _col_matches_dimension(col: str, needle: str) -> bool:
    k = _normkey(col); n = _normkey(needle)
    if not n:
        return False
    if n in k:
        return True
    groups = {
        "region":  {"region","zone","territory","area"},
        "state":   {"state","province","prov","regionstate"},
        "country": {"country","nation","cntry"},
        "city":    {"city","town","district","location"},
        "product": {"product","productname","item","sku","model","article","variant"},
        "brand":   {"brand","make"},
        "category":{"category","subcategory","dept","department","segment","class"},
        "branch":  {"branch","branchname","outlet","store"},
    }
    if n in groups:
        return any(s in k for s in groups[n])
    return False

def _parse_grouped_metric_request(q: str) -> Optional[Dict[str, Any]]:
    mr = _parse_metric_request(q)
    if not mr:
        # if user only said "sales by region", default metric=sum + try needle from text
        mr = {"kind": "sum", "needle": "", "window": _parse_time_window(q)}
    dim = ""
    m = _BY_TOKEN_RE.search(q or "")
    if m:
        dim = (m.group(1) or "").split(",")[0].strip()
    else:
        w = _WISE_RE.search(q or "")
        if w:
            dim = (w.group(1) or "").strip()
    dim = dim.lower()
    if not dim:
        return None
    mr["dimension"] = dim
    return mr

def _fallback_grouped_metric_union(question: str) -> Optional[str]:
    req = _parse_grouped_metric_request(question)
    if not req:
        return None
    try:
        catalog = get_catalog()
    except Exception:
        return None

    all_tables = list((catalog.get("tables") or {}).keys())
    dim_token = req["dimension"]
    needle = req.get("needle") or ""
    window = req.get("window") or {}

    pieces_sum: List[str] = []
    pieces_avg: List[str] = []   # Bucket, s, n
    pieces_min: List[str] = []
    pieces_max: List[str] = []
    pieces_cnt: List[str] = []   # row counts per bucket

    for tbl in all_tables:
        if not _table_exists(tbl):
            continue
        tinfo = get_table_info(tbl) or {}
        cols_map: Dict[str, Any] = (tinfo.get("columns") or {})
        if not cols_map:
            continue

        # choose dimension column
        dim_col = None
        for c in cols_map.keys():
            if _col_matches_dimension(c, dim_token):
                dim_col = c; break
        if not dim_col:
            continue

        # choose metric columns
        metric_cols: List[str] = []
        if req["kind"] in ("sum","avg","min","max"):
            if needle:
                metric_cols = [c for c in cols_map.keys() if _col_matches_revenueish(c, needle)]
            if not metric_cols:
                metric_cols = [c for c in cols_map.keys() if _col_matches_revenueish(c, None)]
        if req["kind"] not in ("sum","avg","min","max") and not metric_cols:
            metric_cols = ["*"]  # count by bucket

        if not metric_cols:
            continue

        date_col = _pick_date_column(cols_map)
        pred = _date_predicate_for(date_col, window)
        where_clause = f"WHERE {pred}" if pred else ""
        t = _q_tbl_mssql(tbl)
        b = f"CAST({_q_col_mssql(dim_col)} AS NVARCHAR(200))"

        for mc in metric_cols:
            if mc == "*":
                pieces_cnt.append(f"SELECT {b} AS Bucket, COUNT(1) AS v FROM {t} {where_clause} GROUP BY {b}")
                continue
            qmc = _q_col_mssql(mc)
            if req["kind"] == "sum":
                pieces_sum.append(f"SELECT {b} AS Bucket, SUM(TRY_CONVERT(decimal(38,6), {qmc})) AS v FROM {t} {where_clause} GROUP BY {b}")
            elif req["kind"] == "avg":
                pieces_avg.append(
                    "SELECT {b} AS Bucket, "
                    "SUM(TRY_CONVERT(decimal(38,6), {c})) AS s, "
                    "SUM(CASE WHEN TRY_CONVERT(decimal(38,6), {c}) IS NOT NULL THEN 1 ELSE 0 END) AS n "
                    "FROM {t} {w} GROUP BY {b}".format(b=b, c=qmc, t=t, w=where_clause)
                )
            elif req["kind"] == "min":
                pieces_min.append(f"SELECT {b} AS Bucket, MIN(TRY_CONVERT(decimal(38,6), {qmc})) AS v FROM {t} {where_clause} GROUP BY {b}")
            elif req["kind"] == "max":
                pieces_max.append(f"SELECT {b} AS Bucket, MAX(TRY_CONVERT(decimal(38,6), {qmc})) AS v FROM {t} {where_clause} GROUP BY {b}")

    if req["kind"] == "sum" and pieces_sum:
        inner = " \n  UNION ALL\n  ".join(pieces_sum)
        return "SELECT Bucket, CAST(SUM(v) AS decimal(38,2)) AS Total FROM (\n  " + inner + "\n) x GROUP BY Bucket ORDER BY Total DESC"
    if req["kind"] == "avg" and pieces_avg:
        inner = " \n  UNION ALL\n  ".join(pieces_avg)
        return "SELECT Bucket, CAST(SUM(s)/NULLIF(SUM(n),0) AS decimal(38,2)) AS Average FROM (\n  " + inner + "\n) x GROUP BY Bucket ORDER BY Average DESC"
    if req["kind"] == "min" and pieces_min:
        inner = " \n  UNION ALL\n  ".join(pieces_min)
        return "SELECT Bucket, MIN(v) AS MinValue FROM (\n  " + inner + "\n) x GROUP BY Bucket ORDER BY MinValue ASC"
    if req["kind"] == "max" and pieces_max:
        inner = " \n  UNION ALL\n  ".join(pieces_max)
        return "SELECT Bucket, MAX(v) AS MaxValue FROM (\n  " + inner + "\n) x GROUP BY Bucket ORDER BY MaxValue DESC"
    if pieces_cnt and req["kind"] not in ("sum","avg","min","max"):
        inner = " \n  UNION ALL\n  ".join(pieces_cnt)
        return "SELECT Bucket, SUM(v) AS Count FROM (\n  " + inner + "\n) x GROUP BY Bucket ORDER BY Count DESC"
    return None

# ---------- “select all <word>” parser + union builder ----------
_SELECT_ALL_RE = re.compile(r"\bselect\s+all\s+([a-z0-9_ \-]+)\b", re.I)

def _parse_select_all_target(q: str) -> Optional[Dict[str, Any]]:
    m = _SELECT_ALL_RE.search(q or "")
    if not m:
        return None
    target = (m.group(1) or "").strip().lower()
    if not target:
        return None
    nk = _normkey(target)
    if any(k in nk for k in ("email", "mail", "emailid")):
        return {"kind": "email", "alias": "Email"}
    if any(k in nk for k in ("phone", "mobile", "whatsapp", "number", "phoneno", "mobileno")):
        return {"kind": "phone", "alias": "Phone"}
    if "name" in nk:
        return {"kind": "name", "alias": "Name"}
    if "contact" in nk or "contacts" in nk:
        return {"kind": "contact", "alias": "Contact"}  # email + phone family
    return {"kind": "generic", "alias": re.sub(r"\s+", "_", target).title(), "needle": nk}

def _fallback_select_all_union(question: str) -> Optional[str]:
    info = _parse_select_all_target(question)
    if not info:
        return None
    try:
        catalog = get_catalog()
    except Exception:
        return None

    all_tables = list((catalog.get("tables") or {}).keys())  # scan all
    cand_tables = tables_used_in_question_hint(question, top_k=80) or all_tables

    # merge & keep order with preference to hinted tables
    preferred = list(dict.fromkeys(cand_tables + all_tables))

    selects: List[str] = []
    alias = info["alias"]

    for tbl in preferred:
        if not _table_exists(tbl):
            continue

        tinfo = get_table_info(tbl) or {}
        cols_map: Dict[str, Any] = (tinfo.get("columns") or {})
        if not cols_map:
            continue
        cols_lc = _lower_keys(cols_map)
        exprs: List[Tuple[str, str]] = []
        seen: Set[str] = set()

        email_cands = [
            "Email", "EmailAddress", "EmailID", "EmailId", "Mail", "MailId", "E-mail",
            "to_email", "from_email"
        ]
        phone_cands = [
            "Phone", "PhoneNo", "PhoneNumber", "Phone_Number", "Phone1", "Phone2",
            "phone_number_1", "phone_number_2",
            "Mobile", "MobileNo", "Mobile_No", "MobileNumber", "Mobile1", "Mobile2",
            "ContactNo", "ContactNumber",
            "WhatsApp", "WhatsAppNo", "WhatsAppNumber", "WhatsappNumber", "WhatsappNo",
            "contact_number"
        ]

        if info["kind"] in ("email", "contact"):
            for cand in email_cands:
                lc = cand.lower()
                if lc in cols_lc and lc not in seen:
                    exprs.append((f"CAST({_q_col_mssql(cand)} AS NVARCHAR(500))", cand))
                    seen.add(lc)
            for c in cols_map.keys():
                lc = c.lower()
                if lc not in seen and _col_matches_email(c):
                    exprs.append((f"CAST({_q_col_mssql(c)} AS NVARCHAR(500))", c))
                    seen.add(lc)

        if info["kind"] in ("phone", "contact"):
            for cand in phone_cands:
                lc = cand.lower()
                if lc in cols_lc and lc not in seen:
                    exprs.append((f"CAST({_q_col_mssql(cand)} AS NVARCHAR(200))", cand))
                    seen.add(lc)
            for c in cols_map.keys():
                lc = c.lower()
                if lc not in seen and _col_matches_phone(c):
                    exprs.append((f"CAST({_q_col_mssql(c)} AS NVARCHAR(200))", c))
                    seen.add(lc)

        if info["kind"] == "name":
            nx = _best_name_expr(cols_map)
            if nx:
                exprs.append((nx, "Name"))

        if info["kind"] == "generic":
            needle = info.get("needle") or ""
            for c in cols_map.keys():
                if needle and needle in _normkey(c):
                    exprs.append((f"CAST({_q_col_mssql(c)} AS NVARCHAR(500))", c))

        if not exprs:
            continue

        qtbl = _q_tbl_mssql(tbl)
        for ex, src_col in exprs:
            selects.append(
                "SELECT TOP 60 "
                f"NULLIF(LTRIM(RTRIM({ex})), '') AS {alias}, "
                f"CAST('{tbl}' AS NVARCHAR(128)) AS SourceTable, "
                f"CAST('{src_col}' AS NVARCHAR(128)) AS SourceColumn "
                f"FROM {qtbl} WHERE {ex} IS NOT NULL"
            )

    if not selects:
        return None

    union = "\nUNION ALL\n".join(selects)
    return f"SELECT {alias}, SourceTable, SourceColumn FROM (\n{union}\n) u WHERE {alias} <> ''"

# ---------- legacy contact-aware union ----------
def _fallback_sql_contacts_union(question: str) -> Optional[str]:
    try:
        catalog = get_catalog()
    except Exception:
        return None

    all_tables = list((catalog.get("tables") or {}).keys())
    cand_tables = tables_used_in_question_hint(question, top_k=40) or all_tables

    selects: List[str] = []
    count_parts: List[str] = []

    want_name = "name" in (question or "").lower()

    for tbl in dict.fromkeys(cand_tables + all_tables):
        if not _table_exists(tbl):
            continue

        info = get_table_info(tbl) or {}
        cols_map: Dict[str, Any] = (info.get("columns") or {})
        if not cols_map:
            continue
        cols_lc = _lower_keys(cols_map)

        phone_col = _first_present_expr(cols_lc, ["Phone", "PhoneNumber", "Mobile", "MobileNo", "ContactNo", "ContactNumber", "contact_number"])
        email_col = _first_present_expr(cols_lc, ["Email", "EmailAddress", "Mail", "MailId", "E-mail", "to_email", "from_email"])
        name_expr = _best_name_expr(cols_map)

        if not (phone_col or email_col or name_expr):
            # fuzzy fallback
            for c in cols_map:
                if not phone_col and _col_matches_phone(c):
                    phone_col = _q_col_mssql(c)
                if not email_col and _col_matches_email(c):
                    email_col = _q_col_mssql(c)
            if not (phone_col or email_col or name_expr):
                continue

        name_sql  = name_expr or "CAST(NULL AS NVARCHAR(200))"
        phone_sql = f"CAST({phone_col} AS NVARCHAR(200))" if phone_col else "CAST(NULL AS NVARCHAR(200))"
        email_sql = f"CAST({email_col} AS NVARCHAR(200))" if email_col else "CAST(NULL AS NVARCHAR(200))"

        where_pred = ""
        preds = []
        if name_expr:
            preds.append(f"{name_expr} IS NOT NULL")
        if phone_col:
            preds.append(f"{phone_col} IS NOT NULL")
        if email_col:
            preds.append(f"{email_col} IS NOT NULL")
        if want_name and name_expr:
            where_pred = f"WHERE {name_expr} IS NOT NULL"
        elif preds:
            where_pred = f"WHERE {' OR '.join(preds)}"

        q_tbl = _q_tbl_mssql(tbl)
        selects.append(
            "SELECT TOP 60 "
            f"{name_sql} AS Name, {phone_sql} AS Phone, {email_sql} AS Email, "
            f"CAST('{tbl}' AS NVARCHAR(128)) AS SourceTable "
            f"FROM {q_tbl} {where_pred}"
        )

        nnc = _nonnull_condition({k: True for k in cols_map.keys()},
                                 [c for c in ["Phone","PhoneNumber","Mobile","MobileNo","Email","EmailAddress","Mail","MailId","E-mail","FullName","full_name","Name","FirstName","LastName","to_email","from_email","contact_number"] if c in cols_map])
        if nnc:
            count_parts.append(f"SELECT COUNT(1) AS cnt FROM {q_tbl} WHERE {nnc}")

    if not selects and not count_parts:
        return None

    ql = (question or "").lower()
    if "count" in ql:
        return "SELECT SUM(cnt) AS TotalContacts FROM (\n  " + "\n  UNION ALL\n  ".join(count_parts) + "\n) x"

    union = "\nUNION ALL\n".join(selects)
    return "SELECT TOP 60 * FROM (\n" + union + "\n) u"

# ---------- NEW: multi-table COUNT for any token ----------
_COUNT_ANY_RE = re.compile(r"\bcount(?:\s+all)?\s+([a-z0-9_ \-]+)\b", re.I)

def _parse_count_target(q: str) -> Optional[Dict[str, str]]:
    m = _COUNT_ANY_RE.search(q or "")
    if not m:
        return None
    token = (m.group(1) or "").strip().lower()
    if not token:
        return None
    nk = _normkey(token)
    if any(k in nk for k in ("email", "mail", "emailid")):
        return {"kind": "email"}
    if any(k in nk for k in ("phone", "mobile", "whatsapp", "number", "phoneno", "mobileno", "contact")):
        return {"kind": "phone"}  # includes generic "contact"
    return {"kind": "generic", "needle": nk}

def _fallback_count_union(question: str) -> Optional[str]:
    info = _parse_count_target(question)
    if not info:
        return None
    try:
        catalog = get_catalog()
    except Exception:
        return None

    all_tables = list((catalog.get("tables") or {}).keys())
    parts: List[str] = []

    for tbl in all_tables:
        if not _table_exists(tbl):
            continue
        tinfo = get_table_info(tbl) or {}
        cols_map: Dict[str, Any] = (tinfo.get("columns") or {})
        if not cols_map:
            continue

        preds: List[str] = []
        if info["kind"] == "email":
            for c in cols_map.keys():
                if _col_matches_email(c):
                    preds.append(f"{_q_col_mssql(c)} IS NOT NULL")
        elif info["kind"] == "phone":
            for c in cols_map.keys():
                if _col_matches_phone(c):
                    preds.append(f"{_q_col_mssql(c)} IS NOT NULL")
        else:
            needle = info.get("needle") or ""
            for c in cols_map.keys():
                if needle and needle in _normkey(c):
                    preds.append(f"{_q_col_mssql(c)} IS NOT NULL")

        if not preds:
            continue

        where_pred = " OR ".join(preds)
        parts.append(f"SELECT COUNT(1) AS cnt FROM {_q_tbl_mssql(tbl)} WHERE {where_pred}")

    if not parts:
        return None
    return "SELECT SUM(cnt) AS Total FROM (\n  " + "\n  UNION ALL\n  ".join(parts) + "\n) t"

# ---------- General fallback ----------
def _fallback_sql(question: str) -> Optional[str]:
    # 1) Grouped metrics like "total revenue by region/state/product ..."
    gm = _fallback_grouped_metric_union(question)
    if gm:
        return gm

    # 2) Single-number metrics like "total revenue", "average price last month"
    mu = _fallback_metric_union(question)
    if mu:
        return mu

    # 3) Multi-table counts like "count phone numbers" / "count all emails"
    ca = _fallback_count_union(question)
    if ca:
        return ca

    # 4) Deterministic "select all <word>" (e.g., phones/emails/names)
    sa = _fallback_select_all_union(question)
    if sa:
        return sa

    # 5) Contact triple (Name/Phone/Email) union when contact words present
    if _needs_contact_columns(question):
        return _fallback_sql_contacts_union(question)

    # 6) Last-resort: pick one table & preview a few columns
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
    prefer_contact = ["Phone", "Mobile", "MobileNo", "PhoneNumber", "Email", "EmailAddress", "to_email", "from_email"]
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


# ---------- direct user SQL detection ----------
_DIRECT_SQL_RE = re.compile(r"(?is)^\s*(with\b.*?select|select)\b.*\bfrom\b")

def _looks_like_user_sql(text: str) -> bool:
    if _is_sql_mode(text):
        return True
    if _SQL_FENCE.search(text or ""):
        return True
    # don't treat "select all ..." as SQL
    if re.match(r"(?is)^\s*select\s+all\b", text or ""):
        return False
    return _DIRECT_SQL_RE.search(text or "") is not None

def _ensure_db_and_catalog() -> None:
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

def _generate_sql_via_model(user_text: str, schema_snippet: str) -> Tuple[str, str]:
    prompt = build_default_prompt(user_text, schema_snippet)
    provider = (getattr(settings, "TEXT_TO_SQL_PROVIDER", "ollama") or "ollama").lower()
    logger.info("t2sql.generate: provider=%s", provider)
    if provider == "openai":
        if not openai_generate:
            raise RuntimeError("OpenAI provider not available in this build.")
        return openai_generate(prompt), "openai"
    if provider == "ollama":
        if not ollama_generate:
            raise RuntimeError("Ollama provider not available in this build.")
        return ollama_generate(prompt), "ollama"
    if ollama_generate:
        return ollama_generate(prompt), "ollama"
    if openai_generate:
        return openai_generate(prompt), "openai"
    raise RuntimeError("No Text-to-SQL provider configured.")

# ---------- idempotency helpers ----------
def _existing_exchange_indices(
    msgs: List[Dict], text_norm: str, client_id: Optional[str]
) -> Tuple[Optional[int], Optional[int]]:
    user_idx = None
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
def _run_chat_turn_locked(chat_id: str, text: str, client_msg_id: Optional[str]) -> Tuple[str, Dict[str, Any], Optional[str]]:
    t_norm = _norm_text(text)
    now = _now_ms()

    # Hard idempotency
    with _STORE_LOCK:
        msgs = MESSAGES.setdefault(chat_id, [])
        u_idx, ai_idx = _existing_exchange_indices(msgs, t_norm, client_msg_id)
        if u_idx is not None:
            if ai_idx is not None:
                return msgs[ai_idx].get("text") or "", {"mode": "reuse"}, None
            else:
                return "Still thinking…", {"mode": "pending"}, None

    # Soft window
    state = _TURN_STATE.get(chat_id) or {}
    if state and state.get("text_norm") == t_norm and (now - int(state.get("ts", 0))) < _DEDUPE_WINDOW_MS:
        with _STORE_LOCK:
            msgs = MESSAGES.get(chat_id, [])
            for i in range(len(msgs) - 1, -1, -1):
                if msgs[i].get("sender") == "ai" and not msgs[i].get("thinking"):
                    return msgs[i].get("text") or "", {"mode": "reuse-window"}, None
        return "Still thinking…", {"mode": "pending-window"}, None

    # Append user + thinking
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
        _TURN_STATE[chat_id] = {"text_norm": t_norm, "ts": now, "thinking_id": thinking_id}

    trace: Dict[str, Any] = {
        "mode": None,
        "provider": None,
        "picked_tables": [],
        "sql": None,
        "gen_ms": 0,
        "exec_ms": 0,
        "row_count": 0,
        "error": None,
    }
    model_used: Optional[str] = None

    reply_markdown = ""
    gen_ms = exec_ms = 0
    try:
        _ensure_db_and_catalog()

        # ===== Direct user SQL =====
        if _looks_like_user_sql(text):
            raw_sql = text
            # If prefixed with sql:/db:, strip it
            raw_sql = re.sub(r"^\s*(?:sql|db|data)\s*:\s*", "", raw_sql, flags=re.I)
            # If fenced, extract inside
            m = _SQL_FENCE.search(raw_sql)
            if m:
                raw_sql = m.group(1)
            # Keep only first statement and ensure SELECT/WITH
            safe_sql = validate_sql(raw_sql, allowed_tables=None, enforce_row_limit=True)
            timer = Timer()
            rows = run_select(safe_sql)
            exec_ms = timer.ms
            table_md, row_count = _format_markdown_table(rows, max_rows=60)
            trace.update({"mode": "direct-sql", "sql": safe_sql, "row_count": row_count})
            reply_markdown = f"SQL executed:\n```sql\n{safe_sql}\n```\n\nResult:\n{table_md}"

        # ===== “select all <word>” (deterministic, no model) =====
        # ===== Deterministic fallbacks (no model) =====
        else:
            # 1) Grouped totals like "total revenue by region/state/product ..." (+ date windows)
            gm_sql = _fallback_grouped_metric_union(text)
            if gm_sql:
                safe_sql = validate_sql(gm_sql, allowed_tables=None, enforce_row_limit=False)
                timer = Timer(); rows = run_select(safe_sql); exec_ms = timer.ms
                table_md, row_count = _format_markdown_table(rows, max_rows=60)
                trace.update({"mode": "grouped-metric", "sql": safe_sql, "row_count": row_count})
                reply_markdown = f"SQL executed:\n```sql\n{safe_sql}\n```\n\nResult:\n{table_md}"

            else:
                # 2) Single-number metric like "total revenue this year", "average price last month"
                mu_sql = _fallback_metric_union(text)
                if mu_sql:
                    safe_sql = validate_sql(mu_sql, allowed_tables=None, enforce_row_limit=False)
                    timer = Timer(); rows = run_select(safe_sql); exec_ms = timer.ms
                    table_md, row_count = _format_markdown_table(rows, max_rows=60)
                    trace.update({"mode": "metric-union", "sql": safe_sql, "row_count": row_count})
                    reply_markdown = f"SQL executed:\n```sql\n{safe_sql}\n```\n\nResult:\n{table_md}"

                else:
                    # 3) Multi-table COUNT like "count emails/phone numbers"
                    ca_sql = _fallback_count_union(text)
                    if ca_sql:
                        safe_sql = validate_sql(ca_sql, allowed_tables=None, enforce_row_limit=False)
                        timer = Timer(); rows = run_select(safe_sql); exec_ms = timer.ms
                        table_md, row_count = _format_markdown_table(rows, max_rows=60)
                        trace.update({"mode": "count-union", "sql": safe_sql, "row_count": row_count})
                        reply_markdown = f"SQL executed:\n```sql\n{safe_sql}\n```\n\nResult:\n{table_md}"

                    else:
                        # 4) “select all <word>”
                        select_all_info = _parse_select_all_target(text)
                        if select_all_info:
                            sa_sql = _fallback_select_all_union(text)
                            alias = select_all_info.get("alias") or "Value"
                            if sa_sql:
                                safe_sql = validate_sql(sa_sql, allowed_tables=None, enforce_row_limit=True)
                                timer = Timer(); rows = run_select(safe_sql); exec_ms = timer.ms
                                grouped_md, row_count = _format_grouped_by_source(
                                    rows, value_col=alias, source_table_col="SourceTable",
                                    source_col_col="SourceColumn", max_rows_per_group=60
                                )
                                trace.update({"mode": "select-all", "sql": safe_sql, "row_count": row_count})
                                reply_markdown = f"SQL executed:\n```sql\n{safe_sql}\n```\n\nResult:\n{grouped_md}"
                            else:
                                trace.update({"mode": "select-all", "sql": None, "row_count": 0})
                                reply_markdown = f"Result:\n_No '{alias}'-related columns found across scanned tables._"

                        else:
                            # ===== Model path =====
                            tables, _debug = pick_candidates(text, top_k=6, expand_fks=True, neighbors_per_table=2)
                            snippet = make_schema_snippet(tables)
                            trace["picked_tables"] = tables
                            timer = Timer()
                            model_out, provider = _generate_sql_via_model(text, snippet)
                            gen_ms = timer.ms
                            model_out_first = (model_out or "").strip().splitlines()[0] if model_out else "EMPTY"
                            logger.info("t2sql.model_out.first_line: %s", model_out_first)

                            model_used = provider
                            trace["provider"] = provider
                            trace["mode"] = "t2sql"

                            try:
                                sql_for_validation = _extract_sql_from_text(model_out)
                            except Exception:
                                sql_for_validation = _fallback_sql(text) or ""
                                if not sql_for_validation:
                                    raise

                            safe_sql = validate_sql(sql_for_validation, allowed_tables=tables, enforce_row_limit=True)

                            # If model referenced non-existent tables, switch to our fallback
                            try:
                                refs = list(extract_tables(safe_sql))
                                missing = [t for t in refs if not _table_exists(t)]
                                if missing:
                                    fb = _fallback_sql(text)
                                    if fb:
                                        safe_sql = fb
                            except Exception:
                                pass

                            # If user asked for contact-ish fields but model missed them, force union
                            if _needs_contact_columns(text) and not _sql_mentions_contact_cols(safe_sql):
                                fb = _fallback_sql_contacts_union(text)
                                if fb:
                                    safe_sql = fb

                            logger.info("t2sql.sql.validated: %s", safe_sql.replace("\n", " "))
                            trace["sql"] = safe_sql

                            timer = Timer(); rows = run_select(safe_sql); exec_ms = timer.ms
                            table_md, row_count = _format_markdown_table(rows, max_rows=60)
                            trace["row_count"] = row_count
                            reply_markdown = f"SQL executed:\n```sql\n{safe_sql}\n```\n\nResult:\n{table_md}"
                            try:
                                log_text2sql_event(text, picked_tables=tables, sql=safe_sql, ok=True, duration_ms=gen_ms)
                                log_sql_execution(safe_sql, row_count=row_count, duration_ms=exec_ms, tables=tables, ok=True)
                            except Exception:
                                pass


    except Exception as e:
        trace["error"] = f"{type(e).__name__}: {e}"
        try:
            fb_sql = _fallback_sql(text)
            if fb_sql:
                safe_sql = validate_sql(fb_sql, allowed_tables=None, enforce_row_limit=True)
                rows = run_select(safe_sql)
                table_md, row_count = _format_markdown_table(rows, max_rows=60)
                trace.update({"mode": "fallback", "sql": safe_sql, "row_count": row_count})
                reply_markdown = f"SQL executed:\n```sql\n{safe_sql}\n```\n\nResult:\n{table_md}"
            else:
                raise
        except Exception:
            reply_markdown = f"Failed to run query. {type(e).__name__}: {e}"
        try:
            log_text2sql_event(text, picked_tables=[], sql="", ok=False, error=str(e))
        except Exception:
            pass
    finally:
        trace["gen_ms"] = gen_ms
        trace["exec_ms"] = exec_ms

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

    return reply_markdown, trace, model_used

def _run_chat_turn(chat_id: str, text: Optional[str], client_msg_id: Optional[str]) -> Tuple[str, Dict[str, Any], Optional[str]]:
    _ensure_chat_loaded(chat_id, create_if_missing=True)
    t = (text or "").strip()
    if not t:
        return "", {"mode": "noop"}, None
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
def _bad_path_guard(chat_id: str):
    return get_messages(chat_id)

@router.get("/{chat_id}/messages")
def get_messages(chat_id: str):
    with _STORE_LOCK:
        _ensure_chat_loaded(chat_id, create_if_missing=True)
        msgs = MESSAGES.get(chat_id)
        if msgs is None:
            msgs = _load_messages_from_disk(chat_id)
            MESSAGES[chat_id] = msgs
        items = sorted(msgs, key=lambda m: (int(m.get("atMs") or 0), str(m.get("id") or "")))
        return {
            "chatId": chat_id,
            "title": CHATS[chat_id]["title"],
            "createdAt": CHATS[chat_id]["createdAt"],
            "items": items,
        }

@router.post("/{chat_id}")
def run_chat_turn_endpoint(chat_id: str):
    _ensure_chat_loaded(chat_id, create_if_missing=True)
    return {"ok": True, "chatId": chat_id, "reply": ""}

@router.post("/{chat_id}/messages")
def send_message(chat_id: str, payload: Optional[SendMessage] = Body(default=None)):
    text = (payload.text if payload else None) or ""
    client_id = payload.clientMsgId if payload else None
    reply, trace, model_used = _run_chat_turn(chat_id, text=text, client_msg_id=client_id)
    return {"ok": True, "chatId": chat_id, "reply": reply, "trace": trace, "model_used": model_used}

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
