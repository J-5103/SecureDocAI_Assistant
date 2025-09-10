# src/main.py
from src.utils.pydantic_bridge import apply_pydantic_v1_bridge
apply_pydantic_v1_bridge()

from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Body, Query, BackgroundTasks
# ...

import os, shutil, re, uuid, json, base64, mimetypes, importlib, io, difflib
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime
from time import perf_counter
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Body, Query, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import logging

# =========================================================
# Robust .env loader so GEMINI_API_KEY in src/.env is seen
# =========================================================
def _load_envs():
    try:
        from dotenv import load_dotenv  # type: ignore
    except Exception:
        return
    here = Path(__file__).resolve()
    candidates = [
        Path.cwd() / ".env",                          # where uvicorn was started
        here.parent / ".env",                         # src/.env
        here.parent / "pipeline" / ".env",            # src/pipeline/.env (just in case)
        here.parent.parent / ".env",                  # project root .env
    ]
    for p in candidates:
        try:
            if p.exists():
                load_dotenv(p, override=False)        # do not clobber already-set env
        except Exception:
            pass

_load_envs()

# Quick helper to check Gemini key
def _get_gemini_key() -> str:
    return (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
if _get_gemini_key():
    logger.info("✅ Gemini key detected in environment.")
else:
    logger.warning("⚠️  No Gemini key detected. Set GEMINI_API_KEY or GOOGLE_API_KEY in your backend .env and restart.")

from src.pipeline.question_answer_pipeline import QuestionAnswerPipeline
try:
    from src.components.rag_pipeline import RAGPipeline
except ImportError as e:
    logger.error(f"Failed to import RAGPipeline: {str(e)}")
    raise

from src.components.plot_generator import PlotGenerator  # noqa: F401
from src.components.file_loader import FileLoader        # noqa: F401
from src.pipeline.plot_pipeline import PlotGenerationPipeline

# =========================
# Include sub-routers
# =========================
visualizations_router = None
try:
    from src.routes.plot_routes import router as visualizations_router  # preferred
except Exception:
    try:
        from routes.plot_routes import router as visualizations_router  # fallback import
    except Exception:
        visualizations_router = None

chat_router = None
try:
    from src.routes.chat_routes import router as chat_router
except Exception:
    try:
        from routes.chat_routes import router as chat_router
    except Exception:
        chat_router = None

image_router = None
try:
    from src.routes.image_routes import router as image_router  # preferred
except Exception:
    try:
        from routes.image_routes import router as image_router
    except Exception:
        image_router = None

# =========================
# DB handler (for contacts)
# =========================
DBHandler = None
try:
    from src.components.db_handler import DBHandler as _DBH
    DBHandler = _DBH
except Exception:
    try:
        from src.components.db_handler import DBHandler as _DBH
        DBHandler = _DBH
    except Exception as e:
        logger.warning(f"DBHandler not available, contact auto-save will be skipped: {e}")

# ----- FS layout (make static dir before mounting!) -----
PROJECT_ROOT = Path(__file__).resolve().parents[1]
STATIC_ROOT = PROJECT_ROOT / "static"
STATIC_ROOT.mkdir(parents=True, exist_ok=True)

STATIC_VIS_DIR = STATIC_ROOT / "visualizations"
STATIC_VIS_DIR.mkdir(parents=True, exist_ok=True)

# persistent uploads for image previews / card snapshots
UPLOAD_IMAGES_DIR = STATIC_ROOT / "uploads"
UPLOAD_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

# vCards folder
VCARD_DIR = STATIC_ROOT / "vcards"
VCARD_DIR.mkdir(parents=True, exist_ok=True)

UPLOAD_BASE = "uploaded_docs"     # PDFs/Docs/Images (per chat_id)
UPLOAD_EXCEL = "uploaded_excels"  # Excel/CSV     (per chat_id OR loose files)
os.makedirs(UPLOAD_BASE, exist_ok=True)
os.makedirs(UPLOAD_EXCEL, exist_ok=True)

# ---------- init ----------
qa_pipeline = QuestionAnswerPipeline()
rag_pipeline = RAGPipeline()
plot_pipeline = PlotGenerationPipeline()

app = FastAPI(title="SecureDocAI Backend", version="1.5.0")  # version bump

# -------- CORS (friendlier for dev) --------
_frontend_origins = os.getenv("FRONTEND_ORIGINS")
if _frontend_origins:
    _allowed_origins = [o.strip() for o in _frontend_origins.split(",") if o.strip()]
else:
    _allowed_origins = [
        "http://192.168.0.109:3000",
        "http://192.168.0.109:3000",
        # "http://192.168.0.109:5173",
        "http://192.168.0.88:3000",
        "http://192.168.0.88:5173",
        "http://localhost:3000",
        "http://localhost:5173",
    ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# -------- CORS end --------

# Serve images and other assets from ./static
app.mount("/static", StaticFiles(directory=str(STATIC_ROOT)), name="static")

# Mount routers
if visualizations_router:
    app.include_router(visualizations_router)
if chat_router:
    app.include_router(chat_router)
if image_router:
    # exposes: POST /api/ask-image (Gemini-based card extractor in your image_routes)
    app.include_router(image_router, prefix="/api")

# ---- Allowed extensions ----
CHAT_DOC_EXTS = {".pdf", ".doc", ".docx", ".xls", ".xlsx", ".csv", ".png", ".jpg", ".jpeg"}
VIZ_DATA_EXTS = {".xlsx", ".xls", ".csv"}
PDF_EXTS = {".pdf"}

# ---------- helpers ----------
def _safe_name(name: str) -> str:
    return os.path.basename((name or "").replace("\\", "/"))

def _ext(fname: str) -> str:
    return os.path.splitext(fname)[-1].lower()

def _is_chat_doc(fname: str) -> bool:
    return _ext(fname) in CHAT_DOC_EXTS

def _is_excel(fname: str) -> bool:
    return _ext(fname) in VIZ_DATA_EXTS

# ==============================
# Helpers to persist image bytes
# ==============================
_ALLOWED_IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp", ".gif"}

def _save_image_to_static(chat_id: str, up: UploadFile) -> str:
    """Save a single image into /static/uploads/<chat_id>/ and return public URL."""
    ext = _ext(up.filename or "")
    if ext not in _ALLOWED_IMAGE_EXT or not (up.content_type or "").startswith("image/"):
        raise HTTPException(status_code=400, detail=f"Only image files are allowed: {', '.join(sorted(_ALLOWED_IMAGE_EXT))}")
    chat_dir = UPLOAD_IMAGES_DIR / chat_id
    chat_dir.mkdir(parents=True, exist_ok=True)
    fname = f"{uuid.uuid4().hex}{ext}"
    with open(chat_dir / fname, "wb") as f:
        shutil.copyfileobj(up.file, f)
    return f"/static/uploads/{chat_id}/{fname}"

def _save_bytes_to_static(chat_id: str, original_name: Optional[str], data: bytes) -> str:
    """Save raw bytes as an image under /static/uploads/<chat_id>/ and return public URL."""
    ext = _ext(original_name or "") or ".jpg"
    if ext not in _ALLOWED_IMAGE_EXT:
        ext = ".jpg"
    chat_dir = UPLOAD_IMAGES_DIR / (chat_id or "cards")
    chat_dir.mkdir(parents=True, exist_ok=True)
    fname = f"{uuid.uuid4().hex}{ext}"
    with open(chat_dir / fname, "wb") as f:
        f.write(data)
    return f"/static/uploads/{chat_id or 'cards'}/{fname}"

# =========================
# NEW: Stats utilities
# =========================
import pandas as pd
import numpy as np

# detect stats
_STAT_PATTERNS = [
    ("mean",   re.compile(r"\b(average|avg|mean)\b", re.I)),
    ("median", re.compile(r"\bmedian\b", re.I)),
    ("max",    re.compile(r"\b(max|maximum|highest|largest|top)\b", re.I)),
    ("min",    re.compile(r"\b(min|minimum|lowest|smallest|bottom)\b", re.I)),
    ("sum",    re.compile(r"\b(sum|total)\b", re.I)),
    ("count",  re.compile(r"\b(count|how many|number of|no\.?\s*of)\b", re.I)),
]

_VALUE_ALIASES = {
    "sales":   ["sales", "sale", "amount", "revenue"],
    "profit":  ["profit", "margin", "gain"],
    "quantity":["quantity", "qty", "units"],
    "price":   ["price", "unit price", "avg price", "mrp", "rate"],
}

_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
_BY_RE   = re.compile(r"\bby\s+([a-zA-Z0-9 \-_]+)", re.I)
_WISE_RE = re.compile(r"\b([a-zA-Z0-9 \-_]+?)\s*(?:-|\s)?wise\b", re.I)

# --- sheet selection from question ---
_SHEET_PATTERNS = [
    re.compile(r"\bfrom\s+sheet\s+([A-Za-z0-9 _\-]+)\b", re.I),
    re.compile(r"\bsheet\s*[:=]\s*([A-Za-z0-9 _\-]+)\b", re.I),
    re.compile(r"\btab\s*[:=]?\s*([A-Za-z0-9 _\-]+)\b", re.I),
    re.compile(r"\bsheet\s+([A-Za-z0-9 _\-]+)\b", re.I),
]

def _parse_sheet_from_question(q: str) -> Optional[str]:
    q = (q or "").strip()
    for pat in _SHEET_PATTERNS:
        m = pat.search(q)
        if m:
            name = (m.group(1) or "").strip()
            if name and name.lower() not in {"the", "first", "sheet"}:
                return name
    return None

def _pick_sheet(xl: pd.ExcelFile, requested: Optional[str]) -> str:
    names = xl.sheet_names or []
    if not names:
        raise ValueError("Excel file has no sheets.")
    if requested:
        low = [n.lower() for n in names]
        if requested.lower() in low:
            return names[low.index(requested.lower())]
        match = difflib.get_close_matches(requested.lower(), low, n=1, cutoff=0.7)
        if match:
            return names[low.index(match[0])]
    return names[0]

def _norm(s: str) -> str:
    return (s or "").strip().lower().replace("_"," ").replace("-"," ")

def _detect_stat_intent(q: str) -> Optional[str]:
    for name, pat in _STAT_PATTERNS:
        if pat.search(q):
            return name
    return None

def _match_col(cols: List[str], aliases: List[str]) -> Optional[str]:
    ncols = [_norm(c) for c in cols]
    for alias in aliases:
        al = _norm(alias)
        for i, nc in enumerate(ncols):
            if al == nc:
                return cols[i]
            if re.search(rf"(?<![a-z0-9]){re.escape(al)}(?![a-z0-9])", nc):
                return cols[i]
    return None

def _pick_metric_col(df: pd.DataFrame, question: str) -> Optional[str]:
    cols = list(df.columns)
    q = _norm(question)
    for key, aliases in _VALUE_ALIASES.items():
        if key in q:
            c = _match_col(cols, aliases)
            if c and pd.api.types.is_numeric_dtype(df[c]): return c
    for c in cols:
        if pd.api.types.is_numeric_dtype(df[c]): return c
    return None

def _pick_label_col(df: pd.DataFrame, question: str) -> Optional[str]:
    cols = list(df.columns); q = _norm(question)
    m = _BY_RE.search(q) or _WISE_RE.search(q)
    if m:
        want = m.group(1).strip()
        col = _match_col(cols, [want])
        if col is not None and not pd.api.types.is_numeric_dtype(df[col]):
            return col
    for pref in ["state","city","country","region","market","payment mode","payment method","year","category","segment","product name","product"]:
        col = _match_col(cols, [pref])
        if col is not None and not pd.api.types.is_numeric_dtype(df[col]):
            return col
    return None

def _find_date_col(df: pd.DataFrame) -> Optional[str]:
    for pref in ["order date","ship date","date","order_date","ship_date","orderdate","shipdate"]:
        c = _match_col(list(df.columns), [pref])
        if c is not None: return c
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            return c
    for c in df.columns:
        if df[c].dtype == object:
            try:
                pd.to_datetime(df[c], errors="raise")
                return c
            except Exception:
                pass
    return None

def _apply_year_filter(df: pd.DataFrame, question: str) -> Tuple[pd.DataFrame, Optional[int]]:
    m = _YEAR_RE.search(question or "")
    if not m: return df, None
    year = int(m.group(0))
    dcol = _find_date_col(df)
    if not dcol:
        return df, year
    dfx = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(dfx[dcol]):
        dfx[dcol] = pd.to_datetime(dfx[dcol], errors="coerce")
    mask = dfx[dcol].dt.year == year
    if mask.any():
        return dfx.loc[mask].copy(), year
    return dfx, year

def _read_df_any(path: str, question: Optional[str] = None) -> pd.DataFrame:
    ext = _ext(path)
    if ext == ".csv":
        try:
            return pd.read_csv(path, encoding="utf-8-sig")
        except UnicodeDecodeError:
            return pd.read_csv(path, encoding="latin1")
    # Excel: select sheet from question if present
    xl = pd.ExcelFile(path)
    sheet = _pick_sheet(xl, _parse_sheet_from_question(question or ""))
    return pd.read_excel(xl, sheet_name=sheet)

def _compute_stat_from_df(df: pd.DataFrame, question: str) -> Dict[str, Any]:
    q = (question or "").strip()
    stat = _detect_stat_intent(q)
    if not stat:
        raise ValueError("No statistical intent detected.")
    df, year = _apply_year_filter(df, q)

    metric = _pick_metric_col(df, q)
    if not metric:
        raise ValueError("Could not find a numeric column to compute on.")

    label = _pick_label_col(df, q)

    agg_fun = {
        "mean":   "mean",
        "median": "median",
        "max":    "max",
        "min":    "min",
        "sum":    "sum",
        "count":  "count",
    }[stat]

    answer: Dict[str, Any] = {
        "intent": "stat",
        "stat": stat,
        "metric": metric,
        "year": year,
    }

    if label:
        if stat == "count":
            g = df.groupby(label, dropna=False).size().rename("count").reset_index()
        else:
            g = df.groupby(label, dropna=False)[metric].agg(agg_fun).reset_index()
        ascending = (stat == "min")
        sort_col = "count" if stat == "count" else metric
        g = g.sort_values(by=sort_col, ascending=ascending).head(20).reset_index(drop=True)

        head_val = g.iloc[0][sort_col]
        head_lab = g.iloc[0][label]
        yr_txt = f" in {year}" if year is not None else ""
        answer["answer"] = (
            f"Top {stat} of {metric} by {label}{yr_txt}: {head_lab} → {head_val:,.2f}"
            if stat != "count" else
            f"Top count by {label}{yr_txt}: {head_lab} → {int(head_val):,}"
        )
        answer["table"] = g.to_dict(orient="records")
        return answer

    if stat == "count":
        val = int(df.shape[0])
    else:
        val = getattr(df[metric], agg_fun)()
        if pd.isna(val):
            val = None

    yr_txt = f" in {year}" if year is not None else ""
    if stat == "count":
        answer["answer"] = f"Row count{yr_txt}: {val:,}"
    else:
        answer["answer"] = f"{stat.title()} of {metric}{yr_txt}: {val:,.2f}"
    answer["value"]  = val
    return answer

def _compute_stat_from_paths(paths: List[str], question: str) -> Dict[str, Any]:
    frames = []
    for p in paths:
        frames.append(_read_df_any(p, question))
    df = pd.concat(frames, ignore_index=True, sort=False)
    return _compute_stat_from_df(df, question)

# ===============  Chat doc manifest helpers  ===============
DOCS_MANIFEST = "manifest.json"

def _chat_docs_dir(chat_id: str) -> Path:
    p = Path(UPLOAD_BASE) / chat_id
    p.mkdir(parents=True, exist_ok=True)
    return p

def _manifest_path(chat_id: str) -> Path:
    return _chat_docs_dir(chat_id) / DOCS_MANIFEST

def _load_manifest(chat_id: str) -> Dict[str, Any]:
    mpath = _manifest_path(chat_id)
    if mpath.exists():
        return json.loads(mpath.read_text(encoding="utf-8"))
    return {"chat_id": chat_id, "docs": []}

def _save_manifest(chat_id: str, data: Dict[str, Any]) -> None:
    _manifest_path(chat_id).write_text(json.dumps(data, indent=2), encoding="utf-8")

def _new_doc_id(filename: str) -> str:
    base = Path(filename).stem
    base = re.sub(r"[^a-zA-Z0-9._-]+", "_", base).strip("._-").lower() or "doc"
    return f"{base}-{uuid.uuid4().hex[:8]}"

# viz chat helpers
CHAT_META_FILE = "meta.json"

def _slugify(text: str) -> str:
    import re as _re
    s = _re.sub(r"[^a-zA-Z0-9\- ]+", "", (text or "").strip())
    s = _re.sub(r"\s+", "-", s)
    return s.lower() or "chat"

def _chat_dir(chat_id: str) -> str:
    return os.path.join(UPLOAD_EXCEL, chat_id)

def _chat_meta_path(chat_id: str) -> str:
    return os.path.join(_chat_dir(chat_id), CHAT_META_FILE)

def _read_chat_meta(chat_id: str) -> dict:
    path = _chat_meta_path(chat_id)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"chat_id": chat_id, "chat_name": chat_id, "created_at": datetime.utcnow().isoformat()}

def _write_chat_meta(chat_id: str, chat_name: str):
    os.makedirs(_chat_dir(chat_id), exist_ok=True)
    with open(_chat_meta_path(chat_id), "w", encoding="utf-8") as f:
        json.dump({
            "chat_id": chat_id,
            "chat_name": chat_name,
            "created_at": datetime.utcnow().isoformat()
        }, f, ensure_ascii=False, indent=2)

# ---------- Task Status Tracking ----------
task_status: Dict[str, Dict[str, Any]] = {}  # In-memory store (use Redis/Db in production)

def update_task_status(task_id: str, status: str, progress: int = 0, error: str = None, document_id: str = None):
    task_status[task_id] = {
        "status": status,
        "progress": progress,
        "error": error,
        "document_id": document_id,
        "updated_at": datetime.utcnow().isoformat()
    }

# ------------------ Gemini Card Extractor (imports + fallbacks) ------------------
BusinessCard = None
extract_business_card = None
make_vcard = None

try:
    from src.pipeline.card_extractor import extract_business_card as _ext_extract_business_card, BusinessCard as _ExtBusinessCard  # type: ignore
    extract_business_card = _ext_extract_business_card
    BusinessCard = _ExtBusinessCard
    logger.info("Using external card_extractor module.")
except Exception as e:
    logger.warning(f"card_extractor module not found, will use inline Gemini extractor. ({e})")

try:
    from src.utils.vcard import make_vcard as _ext_make_vcard  # type: ignore
    make_vcard = _ext_make_vcard
    logger.info("Using external vcard util module.")
except Exception as e:
    logger.warning(f"vcard util module not found, will use inline vCard generator. ({e})")

# If needed, define inline schema + extractor + vcard
if BusinessCard is None:
    class PostalAddress(BaseModel):
        street: Optional[str] = None
        city: Optional[str] = None
        state: Optional[str] = None
        postal_code: Optional[str] = None
        country: Optional[str] = None

    class BusinessCard(BaseModel):
        full_name: Optional[str] = Field(None, description="Person full name")
        first_name: Optional[str] = None
        last_name: Optional[str] = None
        organization: Optional[str] = None
        job_title: Optional[str] = None
        phones: List[str] = Field(default_factory=list)
        emails: List[str] = Field(default_factory=list)
        websites: List[str] = Field(default_factory=list)
        address: Optional[PostalAddress] = None
        social: Dict[str, Optional[str]] = Field(default_factory=dict)
        notes: Optional[str] = None
        raw_text: Optional[str] = None

if extract_business_card is None:
    # ---- JSON extraction helpers (robust to extra prose) ----
    def _extract_first_json(text: str) -> Optional[str]:
        s = (text or "").strip()
        if s.startswith("```"):
            s = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", s)
            s = re.sub(r"\s*```$", "", s)
        if s.startswith("{") and s.endswith("}"):
            return s
        start = s.find("{")
        if start == -1:
            return None
        depth = 0
        for i in range(start, len(s)):
            c = s[i]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return s[start : i + 1]
        return None

    def _safe_json_from_text(text: str) -> Dict[str, Any]:
        try:
            return json.loads(text or "")
        except Exception:
            pass
        obj = _extract_first_json(text or "")
        if obj:
            return json.loads(obj)
        raise RuntimeError("Model did not return valid JSON.")

    # ---- Dynamic Gemini loader (no top-level google.* imports)
    def _load_gemini():
        try:
            genai_mod = importlib.import_module("google.genai")
            types_mod = importlib.import_module("google.genai.types")
            return "new", genai_mod, types_mod
        except Exception:
            try:
                genai_mod = importlib.import_module("google.generativeai")
                return "old", genai_mod, None
            except Exception:
                return None, None, None

    _sdk, _genai, _types = _load_gemini()

    def extract_business_card(image_bytes: bytes, mime_type: str = "image/jpeg") -> BusinessCard:
        if _sdk is None:
            raise HTTPException(status_code=500, detail="Gemini SDK not installed. Install google-genai or google-generativeai.")
        api_key = _get_gemini_key()
        if not api_key:
            raise HTTPException(status_code=500, detail="Gemini API key missing. Set GEMINI_API_KEY (or GOOGLE_API_KEY) in backend .env")

        instruction = (
            "You are a precise information extraction engine for business cards. "
            "Return ONLY a strict JSON object with fields: "
            "full_name, first_name, last_name, organization, job_title, phones[], emails[], websites[], "
            "address{street,city,state,postal_code,country}, social{linkedin,twitter,instagram,facebook,github}, "
            "notes, raw_text. Use null/[] if missing. Preserve visible punctuation/country codes."
        )

        if _sdk == "new":
            client = _genai.Client(api_key=api_key)  # type: ignore[attr-defined]
            try:
                img_part = _types.Part.from_bytes(data=image_bytes, mime_type=mime_type)  # type: ignore[attr-defined]
            except Exception:
                img_part = {"mime_type": mime_type, "data": image_bytes}
            try:
                txt_part = _types.Part.from_text(instruction)  # type: ignore[attr-defined]
            except Exception:
                txt_part = instruction

            try:
                contents = [_types.Content(role="user", parts=[img_part, txt_part])]  # type: ignore[attr-defined]
            except Exception:
                contents = [img_part, txt_part]

            cfg = None
            try:
                cfg = _types.GenerateContentConfig(response_mime_type="application/json")  # type: ignore[attr-defined]
            except Exception:
                cfg = None

            if cfg is not None:
                resp = client.models.generate_content(model="gemini-2.5-flash", contents=contents, config=cfg)  # type: ignore[attr-defined]
            else:
                resp = client.models.generate_content(model="gemini-2.5-flash", contents=contents)  # type: ignore[attr-defined]

            text = getattr(resp, "text", "") or ""
            data = _safe_json_from_text(text)
            return BusinessCard(**data)

        # ---- old SDK fallback (google-generativeai)
        _genai.configure(api_key=api_key)  # type: ignore[attr-defined]
        model = _genai.GenerativeModel("gemini-1.5-flash")  # type: ignore[attr-defined]
        image_part = {"mime_type": mime_type, "data": image_bytes}
        resp = model.generate_content(  # type: ignore[attr-defined]
            [image_part, instruction],
            generation_config={"response_mime_type": "application/json"},
        )
        text = getattr(resp, "text", None) or ""
        data = _safe_json_from_text(text)
        return BusinessCard(**data)

if make_vcard is None:
    def _esc_v(s: Optional[str]) -> str:
        if not s:
            return ""
        return s.replace("\\", "\\\\").replace(";", r"\;").replace(",", r"\,").replace("\n", r"\n")

    def make_vcard(
        full_name: str,
        last_name: str = "",
        first_name: str = "",
        org: Optional[str] = None,
        title: Optional[str] = None,
        phones: Optional[List[str]] = None,
        emails: Optional[List[str]] = None,
        websites: Optional[List[str]] = None,
        address: Optional[Dict[str, Optional[str]]] = None,
        notes: Optional[str] = None,
    ) -> str:
        phones = phones or []
        emails = emails or []
        websites = websites or []
        lines = ["BEGIN:VCARD", "VERSION:3.0"]
        lines.append(f"N:{_esc_v(last_name)};{_esc_v(first_name)};;;")
        lines.append(f"FN:{_esc_v(full_name)}")
        if org:
            lines.append(f"ORG:{_esc_v(org)}")
        if title:
            lines.append(f"TITLE:{_esc_v(title)}")
        for p in phones:
            if p:
                lines.append(f"TEL;TYPE=CELL:{_esc_v(p)}")
        for e in emails:
            if e:
                lines.append(f"EMAIL;TYPE=INTERNET:{_esc_v(e)}")
        for u in websites:
            if u:
                lines.append(f"URL:{_esc_v(u)}")
        if address:
            street = _esc_v(address.get("street"))
            city = _esc_v(address.get("city"))
            state = _esc_v(address.get("state"))
            postal = _esc_v(address.get("postal_code"))
            country = _esc_v(address.get("country"))
            lines.append(f"ADR;TYPE=WORK:;;{street};{city};{state};{postal};{country}")
        if notes:
            lines.append(f"NOTE:{_esc_v(notes)}")
        lines.append(f"REV:{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}")
        lines.append("END:VCARD")
        return "\n".join(lines)

def _preprocess_card_image(image_bytes: bytes) -> Tuple[bytes, str]:
    """
    Light enhancement to help OCR: auto-orient, upscale small sides to ≥800px,
    boost contrast & sharpness, convert to high-quality JPEG.
    Returns (processed_bytes, mime_type).
    """
    try:
        from PIL import Image, ImageOps, ImageEnhance  # type: ignore
        im = Image.open(io.BytesIO(image_bytes))
        im = ImageOps.exif_transpose(im)
        w, h = im.size
        if min(w, h) < 800:
            scale = max(1.0, 800 / float(min(w, h)))
            im = im.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        im = im.convert("RGB")
        im = ImageEnhance.Contrast(im).enhance(1.25)
        im = ImageEnhance.Sharpness(im).enhance(1.1)
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=92, optimize=True)
        return buf.getvalue(), "image/jpeg"
    except Exception:
        # If Pillow isn't installed or any error occurs, return original
        return image_bytes, "image/jpeg"

# ---------- Excel helpers ----------
def _find_excel_in_dir(d: Path) -> Optional[Path]:
    if not d.exists() or not d.is_dir():
        return None
    for p in d.iterdir():
        if p.is_file() and _is_excel(p.name):
            return p
    for ext in (".xlsx", ".xls", ".csv"):
        ms = list(d.rglob(f"*{ext}"))
        if ms:
            return ms[0]
    return None

def _resolve_excel_file(file_path: Optional[str], chat_id: Optional[str]) -> str:
    def _is_excel_file(p: Path) -> bool:
        return p.is_file() and _is_excel(p.name)

    if file_path:
        p = Path(file_path)

        if p.is_absolute():
            if not p.exists():
                raise HTTPException(status_code=404, detail=f"File or directory not found: {p}")
            if p.is_dir():
                found = _find_excel_in_dir(p)
                if not found:
                    raise HTTPException(status_code=404, detail=f"No Excel/CSV found inside: {p}")
                return str(found)
            if not _is_excel(p.name):
                raise HTTPException(status_code=400, detail="file_path must be an Excel/CSV (.xlsx, .xls, .csv)")
            return str(p)

        bases: List[Path] = []
        if chat_id:
            bases.append(Path(UPLOAD_EXCEL) / chat_id)
        bases.append(Path(UPLOAD_EXCEL))

        for b in bases:
            cand = (b / file_path).resolve()
            if cand.exists():
                if cand.is_dir():
                    found = _find_excel_in_dir(cand)
                    if found:
                        return str(found)
                elif _is_excel(cand.name):
                    return str(cand)

        name = Path(file_path).name
        for b in bases:
            for ext in (".xlsx", ".xls", ".csv"):
                cand = (b / f"{name}{ext}").resolve()
                if cand.exists() and _is_excel(cand.name):
                    return str(cand)

        raise HTTPException(
            status_code=404,
            detail=f"Could not resolve '{file_path}' in: {', '.join(str(b) for b in bases)}"
        )

    bases: List[Path] = []
    if chat_id:
        bases.append(Path(UPLOAD_EXCEL) / chat_id)
    bases.append(Path(UPLOAD_EXCEL))

    files: List[Path] = []
    for b in bases:
        if b.exists():
            files.extend([p for p in b.rglob("*") if _is_excel_file(p)])

    if not files:
        where = f"chat_id='{chat_id}'" if chat_id else "uploaded_excels root"
        raise HTTPException(status_code=404, detail=f"No Excel/CSV found for {where}.")

    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return str(files[0])

# ---------- uploads (Chat Sessions: all supported types) ----------
@app.post("/api/upload/upload_file")
async def upload_file(
    chat_id: str = Form(...),
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None
):
    """
    (Legacy single-file) Chat Sessions uploader — SUPPORTS: .pdf, .doc, .docx, .xls, .xlsx, .csv, .png, .jpg, .jpeg
    """
    try:
        safe_name = _safe_name(file.filename)
        if not _is_chat_doc(safe_name):
            raise HTTPException(
                status_code=400,
                detail="Only PDF, Word, Excel/CSV, or image files are allowed (.pdf, .doc, .docx, .xls, .xlsx, .csv, .png, .jpg, .jpeg).",
            )

        chat_folder = os.path.join(UPLOAD_BASE, chat_id)
        os.makedirs(chat_folder, exist_ok=True)

        # stable doc_id and save
        doc_id = _new_doc_id(safe_name)
        ext = _ext(safe_name)
        saved_path = os.path.join(chat_folder, f"{doc_id}{ext}")
        with open(saved_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        vector_store_path = os.path.join("vectorstores", chat_id, doc_id)
        os.makedirs(os.path.dirname(vector_store_path), exist_ok=True)

        # record in manifest
        manifest = _load_manifest(chat_id)
        meta = {
            "doc_id": doc_id,
            "file_name": safe_name,
            "ext": ext,
            "size": os.path.getsize(saved_path),
            "stored_path": saved_path.replace("\\", "/"),
            "vector_dir": vector_store_path.replace("\\", "/"),
            "created_at": datetime.utcnow().isoformat() + "Z",
        }
        manifest["docs"].append(meta)
        _save_manifest(chat_id, manifest)

        task_id = str(uuid.uuid4())
        update_task_status(task_id, "pending", progress=0)
        logger.info(f"[LEGACY UPLOAD] {safe_name} → {doc_id}. Task ID: {task_id}")

        async def create_vectorstore_task():
            try:
                update_task_status(task_id, "processing", progress=10)
                logger.info(f"Indexing (legacy) {safe_name} at {vector_store_path}")
                rag_pipeline.create_vectorstore(file_path=saved_path, vector_store_path=vector_store_path)
                update_task_status(task_id, "ready", progress=100, document_id=doc_id)
            except Exception as e:
                update_task_status(task_id, "failed", progress=0, error=str(e))
                logger.error(f"Vectorstore creation failed for {safe_name}: {str(e)}")

        if background_tasks:
            background_tasks.add_task(create_vectorstore_task)
            return {
                "document_path": saved_path.replace("\\", "/"),
                "filename": safe_name,
                "chat_id": chat_id,
                "status": "processing",
                "task_id": task_id,
                "document_id": doc_id,
            }
        else:
            await create_vectorstore_task()  # type: ignore
            status = task_status.get(task_id, {}).get("status", "unknown")
            return {
                "document_path": saved_path.replace("\\", "/"),
                "filename": safe_name,
                "chat_id": chat_id,
                "status": status,
                "task_id": task_id,
                "document_id": doc_id,
            }
    except OSError as e:
        logger.error(f"Directory or file access error: {str(e)}")
        return JSONResponse(status_code=500, content={"error": f"Directory or file access error: {str(e)}"})
    except Exception as e:
        logger.error(f"Upload failed: {str(e)}")
        return JSONResponse(status_code=500, content={"error": f"Upload failed: {str(e)}"})

# ===============  Multi-file upload tied to a chat  ===============
class MultiUploadResult(BaseModel):
    chat_id: str
    docs: List[Dict[str, Any]]

@app.post("/api/chats/{chat_id}/upload", response_model=MultiUploadResult)
async def upload_many(chat_id: str, files: List[UploadFile] = File(...), background_tasks: BackgroundTasks = None):
    """
    Multi-file upload. Saves each under uploaded_docs/<chat_id>/<doc_id>.<ext>,
    creates vectorstore under vectorstores/<chat_id>/<doc_id>, and writes chat manifest.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files received.")
    manifest = _load_manifest(chat_id)
    added = []

    for up in files:
        safe = _safe_name(up.filename or f"file-{uuid.uuid4().hex}.pdf")
        if not _is_chat_doc(safe):
            raise HTTPException(status_code=400, detail=f"Unsupported file: {safe}")
        doc_id = _new_doc_id(safe)
        ext = _ext(safe)

        chat_dir = _chat_docs_dir(chat_id)
        save_path = chat_dir / f"{doc_id}{ext}"
        with open(save_path, "wb") as f:
            shutil.copyfileobj(up.file, f)

        vdir = Path("vectorstores") / chat_id / doc_id
        vdir.parent.mkdir(parents=True, exist_ok=True)

        meta = {
            "doc_id": doc_id,
            "file_name": safe,
            "ext": ext,
            "size": save_path.stat().st_size,
            "stored_path": str(save_path).replace("\\", "/"),
            "vector_dir": str(vdir).replace("\\", "/"),
            "created_at": datetime.utcnow().isoformat() + "Z",
        }
        manifest["docs"].append(meta)
        added.append(meta)

        # vectorize async
        tid = uuid.uuid4().hex
        update_task_status(tid, "pending", progress=0, document_id=doc_id)

        async def _task(p=str(save_path), vd=str(vdir), t=tid, name=safe):
            try:
                update_task_status(t, "processing", progress=10, document_id=doc_id)
                rag_pipeline.create_vectorstore(file_path=p, vector_store_path=vd)
                update_task_status(t, "ready", progress=100, document_id=doc_id)
                logger.info(f"Indexed {name} → {vd}")
            except Exception as e:
                update_task_status(t, "failed", progress=0, error=str(e), document_id=doc_id)
                logger.exception(f"Index failed for {name}")

        if background_tasks:
            background_tasks.add_task(_task)
        else:
            await _task()  # type: ignore

    _save_manifest(chat_id, manifest)
    return {"chat_id": chat_id, "docs": added}

# ---------- create vector store ----------
@app.post("/api/create-vector-store")
async def create_vector_store(
    chat_id: str = Form(...),
    file_id: str = Form(...),
    background_tasks: BackgroundTasks = None
):
    try:
        # resolve doc_id → path (manifest first, legacy fallback)
        man = _load_manifest(chat_id)
        entry = next((d for d in man["docs"] if d["doc_id"] == file_id or d["file_name"] == file_id), None)
        if entry:
            file_path = entry["stored_path"]
            vector_store_path = entry["vector_dir"]
        else:
            file_path = os.path.join(UPLOAD_BASE, chat_id, file_id)
            if not os.path.exists(file_path):
                raise HTTPException(status_code=404, detail=f"File not found: {file_path}")
            vector_store_path = os.path.join("vectorstores", chat_id, os.path.splitext(file_id)[0])

        os.makedirs(os.path.dirname(vector_store_path), exist_ok=True)
        logger.info(f"Starting vector store creation for {file_id} in chat {chat_id}")

        task_id = str(uuid.uuid4())
        update_task_status(task_id, "pending", progress=0)

        async def create_vectorstore_task():
            try:
                update_task_status(task_id, "processing", progress=10)
                rag_pipeline.create_vectorstore(file_path=file_path, vector_store_path=vector_store_path)
                update_task_status(task_id, "ready", progress=100, document_id=file_id)
            except Exception as e:
                update_task_status(task_id, "failed", progress=0, error=str(e))
                logger.error(f"Vector store creation failed for {file_id}: {str(e)}")

        if background_tasks:
            background_tasks.add_task(create_vectorstore_task)
            return {"status": "processing", "task_id": task_id, "file_id": file_id, "chat_id": chat_id}
        else:
            await create_vectorstore_task()  # type: ignore
            status = task_status.get(task_id, {}).get("status", "unknown")
            return {"status": status, "task_id": task_id, "file_id": file_id, "chat_id": chat_id}
    except OSError as e:
        logger.error(f"Directory or file access error: {str(e)}")
        return JSONResponse(status_code=500, content={"error": f"Directory or file access error: {str(e)}"})
    except Exception as e:
        logger.error(f"Vector store creation failed: {str(e)}")
        return JSONResponse(status_code=500, content={"error": f"Vector store creation failed: {str(e)}"})

# ---------- status check ----------
@app.get("/api/status/{task_id}")
async def get_status(task_id: str):
    return task_status.get(task_id, {"status": "unknown", "progress": 0})

# ---------- list documents for a chat ----------
class AskRequest(BaseModel):
    chat_id: str
    question: str
    document_id: Optional[str] = None
    combine_docs: Optional[List[str]] = None
    doc_ids: Optional[List[str]] = None
    intent: Optional[str] = None  # "qa" | "plot"

@app.get("/api/list_documents")
async def list_documents(chat_id: str = Query(...)):
    try:
        # Prefer manifest
        man = _load_manifest(chat_id)
        if man["docs"]:
            docs = [
                {"name": d["file_name"], "documentId": d["doc_id"], "size": d.get("size", 0)}
                for d in man["docs"]
                if _is_chat_doc(d["file_name"])
            ]
            return {"documents": docs}

        # Fallback (legacy)
        folder = os.path.join(UPLOAD_BASE, chat_id)
        if not os.path.isdir(folder):
            return {"documents": []}
        docs = []
        for f in os.listdir(folder):
            if _is_chat_doc(f):
                p = os.path.join(folder, f)
                docs.append({"name": f, "documentId": os.path.splitext(f)[0], "size": os.path.getsize(p)})
        return {"documents": docs}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# =============== list docs straight from manifest ===============
@app.get("/api/chats/{chat_id}/docs")
def chat_docs(chat_id: str):
    man = _load_manifest(chat_id)
    return {"chat_id": chat_id, "docs": man["docs"]}

# ---------- Visualization Chats (Excel/CSV only) ----------
@app.post("/api/excel/upload/")
def upload_excel(file: UploadFile = File(...), chat_id: Optional[str] = Form(None)):
    try:
        safe = _safe_name(file.filename)
        if not _is_excel(safe):
            raise HTTPException(status_code=400, detail="Only Excel/CSV files are allowed (.xlsx, .xls, .csv).")
        dest_dir = os.path.join(UPLOAD_EXCEL, chat_id) if chat_id else UPLOAD_EXCEL
        os.makedirs(dest_dir, exist_ok=True)

        file_path = os.path.join(dest_dir, safe)
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        return {"file_path": file_path.replace("\\", "/"), "chat_id": chat_id, "message": "Upload successful."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/excel/list")
def list_excel_files(chat_id: str = Query(...)):
    try:
        base = os.path.join(UPLOAD_EXCEL, chat_id)
        if not os.path.isdir(base):
            return {"files": []}

        out = []
        for f in os.listdir(base):
            if _is_excel(f):
                p = os.path.join(base, f)
                stat = os.stat(p)
                out.append({
                    "name": f,
                    "file_path": p.replace("\\", "/"),
                    "size": stat.st_size,
                    "uploaded_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                })
        out.sort(key=lambda x: x["uploaded_at"], reverse=True)
        return {"files": out}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/excel/chats")
def create_viz_chat(chat_name: str = Form(...), file: UploadFile = File(...)):
    try:
        safe = _safe_name(file.filename)
        if not _is_excel(safe):
            raise HTTPException(status_code=400, detail="Only Excel/CSV files are allowed (.xlsx, .xls, .csv).")

        slug = _slugify(chat_name)
        short = uuid.uuid4().hex[:8]
        chat_id = f"{slug}-{short}"

        dest_dir = _chat_dir(chat_id)
        os.makedirs(dest_dir, exist_ok=True)

        file_path = os.path.join(dest_dir, safe)
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        _write_chat_meta(chat_id, chat_name)

        return {
            "chat_id": chat_id,
            "chat_name": chat_name,
            "files": [{"name": safe, "file_path": file_path.replace('\\', '/')}],
            "created_at": datetime.utcnow().isoformat(),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Create viz chat failed: {e}")

@app.get("/api/excel/chats")
def list_viz_chats():
    try:
        if not os.path.isdir(UPLOAD_EXCEL):
            return {"chats": []}

        chats = []
        for d in os.listdir(UPLOAD_EXCEL):
            dir_path = os.path.join(UPLOAD_EXCEL, d)
            if not os.path.isdir(dir_path):
                continue
            meta = _read_chat_meta(d)
            files = [f for f in os.listdir(dir_path) if _is_excel(f)]
            latest = None
            if files:
                files.sort(key=lambda fn: os.stat(os.path.join(dir_path, fn)).st_mtime, reverse=True)
                latest = files[0]

            chats.append({
                "chat_id": meta.get("chat_id", d),
                "chat_name": meta.get("chat_name", d),
                "latest_file": latest,
            })

        chats.sort(key=lambda c: os.stat(_chat_dir(c["chat_id"])).st_mtime, reverse=True)
        return {"chats": chats}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"List viz chats failed: {e}")

# ---------- Excel/CSV -> plot (single file) ----------
@app.post("/api/excel/plot/")
def plot_excel_data(
    file_path: Optional[str] = Form(None),
    question: str = Form(...),
    title: Optional[str] = Form(None),
    chat_id: Optional[str] = Form(None),
):
    try:
        resolved_path = _resolve_excel_file(file_path, chat_id)

        # pre-validate
        try:
            df_check = _read_df_any(resolved_path, question)
            if df_check is None or df_check.empty or df_check.shape[1] == 0:
                raise HTTPException(status_code=422, detail="No usable table found in the file.")
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=422, detail=f"Cannot read the selected file: {e}")

        if _detect_stat_intent(question):
            ans = _compute_stat_from_df(df_check, question)
            return {
                "answer": ans["answer"],
                "meta": ans,
                "chat_id": chat_id,
                "dataset_path": resolved_path,
            }

        t0 = perf_counter()
        meta = plot_pipeline.generate_and_store(
            resolved_path,
            question,
            title=title,
            chat_id=chat_id,
        )

        img_path = STATIC_VIS_DIR / f"{meta['id']}.png"
        with open(img_path, "rb") as f:
            image_base64 = base64.b64encode(f.read()).decode("utf-8")

        logger.info(f"✅ Plot generated & saved in {perf_counter()-t0:.2f}s  ({meta.get('id')})")
        return {
            "image_base64": image_base64,
            "image_url": meta.get("image_url"),
            "thumb_url": meta.get("thumb_url"),
            "meta": meta,
            "message": "Plot generated & saved.",
            "chat_id": chat_id,
        }
    except HTTPException:
        raise
    except Exception as e:
        msg = str(e)
        if "No objects to concatenate" in msg or "empty" in msg.lower():
            raise HTTPException(status_code=422, detail="No usable data found to create the plot.")
        raise HTTPException(status_code=500, detail=f"Plot failed: {e}")

# ---------- Excel/CSV -> plot (combine files) ----------
class CombinePayload(BaseModel):
    file_paths: List[str]
    question: str
    title: Optional[str] = None
    chat_id: Optional[str] = None

@app.post("/api/excel/plot/combine")
def plot_excel_data_combine(payload: CombinePayload):
    try:
        if not payload.file_paths or len(payload.file_paths) < 2:
            raise HTTPException(status_code=400, detail="At least two files are required to combine.")

        resolved: List[str] = []
        for fp in payload.file_paths:
            p = Path(fp)
            if not p.is_absolute():
                if not payload.chat_id:
                    raise HTTPException(status_code=400, detail="chat_id is required when file_paths are not absolute.")
                p = (Path(UPLOAD_EXCEL) / payload.chat_id / fp).resolve()
            if not p.exists():
                raise HTTPException(status_code=404, detail=f"File not found: {p.name}")
            if not _is_excel(p.name):
                raise HTTPException(status_code=400, detail=f"Unsupported file type: {p.name}")
            # pre-validate readability
            try:
                dft = _read_df_any(str(p), payload.question)
                if dft is None or dft.empty or dft.shape[1] == 0:
                    continue
            except Exception:
                continue
            resolved.append(str(p))

        if len(resolved) == 0:
            raise HTTPException(status_code=422, detail="No usable tables found across the selected files.")

        if _detect_stat_intent(payload.question):
            ans = _compute_stat_from_paths(resolved, payload.question)
            return {
                "answer": ans["answer"],
                "meta": ans,
                "chat_id": payload.chat_id,
            }

        meta = plot_pipeline.generate_and_store_combine(
            file_paths=resolved,
            question=payload.question,
            title=payload.title,
            chat_id=payload.chat_id,
        )

        img_path = STATIC_VIS_DIR / f"{meta['id']}.png"
        with open(img_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")

        return {
            "image_base64": b64,
            "image_url": meta.get("image_url"),
            "thumb_url": meta.get("thumb_url"),
            "meta": meta,
            "message": "Combined plot generated & saved.",
            "chat_id": payload.chat_id,
        }
    except HTTPException:
        raise
    except Exception as e:
        msg = str(e)
        if "No objects to concatenate" in msg or "empty" in msg.lower():
            raise HTTPException(status_code=422, detail="No usable data found across the selected files.")
        raise HTTPException(status_code=500, detail=f"Combined plot failed: {e}")

# ---------- Visualizations: list + serve + generate ----------
if visualizations_router is None:
    @app.get("/api/visualizations/{plot_id}/image")
    def vis_image(plot_id: str):
        img = STATIC_VIS_DIR / f"{plot_id}.png"
        if not img.exists():
            raise HTTPException(status_code=404, detail="Image not found")
        return FileResponse(str(img), media_type="image/png")

    @app.get("/api/visualizations/{plot_id}/thumb")
    def vis_thumb(plot_id: str):
        img = STATIC_VIS_DIR / f"{plot_id}_thumb.png"
        if not img.exists():
            raise HTTPException(status_code=404, detail="Thumbnail not found")
        return FileResponse(str(img), media_type="image/png")

    @app.get("/api/visualizations/list")
    def vis_list(
        chat_id: Optional[str] = None,
        q: Optional[str] = None,
        limit: Optional[int] = Query(None, ge=1),
        offset: Optional[int] = Query(0, ge=0),
        order: Optional[str] = Query("desc"),
    ):
        items = plot_pipeline.list_meta(chat_id=chat_id)

        if q:
            ql = q.lower()
            items = [
                it for it in items
                if ql in (it.get("title", "") or "").lower()
                or ql in (it.get("kind", "") or "").lower()
                or ql in (it.get("x", "") or "").lower()
                or ql in (it.get("y", "") or "").lower()
            ]

        items.sort(key=lambda m: m.get("created_at", ""), reverse=(order != "asc"))

        total = len(items)
        if limit is not None:
            items = items[offset: offset + limit]
        else:
            items = items[offset:]

        return {"items": items, "total": total, "chat_ids": list({m.get("chat_id") for m in items if m.get("chat_id")})}

    @app.post("/api/visualizations/generate")
    async def vis_generate(
        file: Optional[UploadFile] = File(None),
        file_path: Optional[str] = Form(None),
        question: str = Form(...),
        title: Optional[str] = Form(None),
        chat_id: Optional[str] = Form(None),
    ):
        try:
            if file is not None:
                if not _is_excel(file.filename or ""):
                    raise HTTPException(status_code=400, detail="Only Excel/CSV files are allowed (.xlsx, .xls, .csv).")
                cid = chat_id or "default"
                chat_dir = Path(UPLOAD_EXCEL) / cid
                chat_dir.mkdir(parents=True, exist_ok=True)
                dest = chat_dir / _safe_name(file.filename)
                with open(dest, "wb") as f:
                    shutil.copyfileobj(file.file, f)

                # pre-validate
                dfv = _read_df_any(str(dest), question)
                if dfv is None or dfv.empty or dfv.shape[1] == 0:
                    raise HTTPException(status_code=422, detail="No usable table found in the file.")

                if _detect_stat_intent(question):
                    ans = _compute_stat_from_df(dfv, question)
                    return {"answer": ans["answer"], "meta": ans, "chat_id": cid, "dataset_path": str(dest)}

                meta = plot_pipeline.generate_and_store(str(dest), question, title=title, chat_id=cid)
                return meta

            if not file_path:
                raise HTTPException(status_code=400, detail="Either 'file' or 'file_path' is required.")
            resolved = _resolve_excel_file(file_path, chat_id)

            dfv = _read_df_any(resolved, question)
            if dfv is None or dfv.empty or dfv.shape[1] == 0:
                raise HTTPException(status_code=422, detail="No usable table found in the file.")

            if _detect_stat_intent(question):
                ans = _compute_stat_from_df(dfv, question)
                return {"answer": ans["answer"], "meta": ans, "chat_id": chat_id, "dataset_path": resolved}

            meta = plot_pipeline.generate_and_store(resolved, question, title=title, chat_id=chat_id)
            return meta
        except HTTPException:
            raise
        except Exception as e:
            msg = str(e)
            if "No objects to concatenate" in msg or "empty" in msg.lower():
                raise HTTPException(status_code=422, detail="No usable data found to create the plot.")
            raise HTTPException(status_code=500, detail=f"Visualization generate failed: {e}")

# ---------- Dedicated Excel Q&A endpoint ----------
class VizAskBody(BaseModel):
    chat_id: Optional[str] = None
    question: str
    file_id: Optional[str] = None
    file_name: Optional[str] = None
    file_path: Optional[str] = None

@app.post("/api/viz/ask")
def ask_viz_excel(payload: VizAskBody):
    candidate = payload.file_path or payload.file_id or payload.file_name or None
    resolved_path = _resolve_excel_file(candidate, payload.chat_id)

    try:
        # pre-validate first
        dfv = _read_df_any(resolved_path, payload.question)
        if dfv is None or dfv.empty or dfv.shape[1] == 0:
            raise HTTPException(status_code=422, detail="No usable table found in the file.")

        if _detect_stat_intent(payload.question):
            ans = _compute_stat_from_df(dfv, payload.question)
            return {
                "answer": ans["answer"],
                "meta": ans,
                "chat_id": payload.chat_id,
                "dataset_path": resolved_path,
            }

        meta = plot_pipeline.generate_and_store(
            resolved_path,
            payload.question,
            title=None,
            chat_id=payload.chat_id,
        )

        img_path = STATIC_VIS_DIR / f"{meta['id']}.png"
        with open(img_path, "rb") as f:
            image_base64 = base64.b64encode(f.read()).decode("utf-8")

        return {
            "answer": "Here is the plot.",
            "image_base64": image_base64,
            "image_url": meta.get("image_url"),
            "thumb_url": meta.get("thumb_url"),
            "meta": meta,
            "chat_id": payload.chat_id,
            "dataset_path": resolved_path,
        }
    except HTTPException:
        raise
    except Exception as e:
        msg = str(e)
        if "No objects to concatenate" in msg or "empty" in msg.lower():
            raise HTTPException(status_code=422, detail="No usable table found to create the plot.")
        raise HTTPException(status_code=500, detail=f"Visualization failed: {e}")

# ---------- Q&A (all supported types) ----------
PLOT_KEYWORDS = re.compile(
    r"\b(plot|graph|chart|visuali[sz]e|bar|line|scatter|hist(ogram)?|pie|box|trend|first\s+\d+\s+rows?|last\s+\d+\s+rows?)\b",
    flags=re.I
)

@app.post("/api/ask")
async def ask_question(request: AskRequest = Body(...)):
    try:
        question = (request.question or "").strip()
        if not question or not request.chat_id:
            raise HTTPException(status_code=400, detail="Missing chat_id or question.")

        stat_intent = _detect_stat_intent(question)
        wants_plot = ((request.intent or "").lower() == "plot") or bool(PLOT_KEYWORDS.search(question))

        # ---- Multi-file QA
        if request.doc_ids:
            logger.info(f"QA across {len(request.doc_ids)} docs for chat {request.chat_id}: {request.doc_ids}")
            answer = qa_pipeline.run(
                question=question,
                chat_id=request.chat_id,
                document_id=None,
                combine_docs=request.doc_ids,
            )
            if isinstance(answer, tuple):
                answer = answer[0]
            return {"answer": str(answer or ""), "used_doc_ids": request.doc_ids}

        # ---- Excel: combine
        if request.combine_docs:
            resolved: List[str] = []
            for fp in request.combine_docs:
                p = Path(fp)
                if not p.is_absolute():
                    p = (Path(UPLOAD_EXCEL) / request.chat_id / fp).resolve()
                if not p.exists() or not _is_excel(p.name):
                    raise HTTPException(status_code=404, detail=f"File not found or not Excel/CSV: {p.name}")
                resolved.append(str(p))

            if stat_intent:
                ans = _compute_stat_from_paths(resolved, question)
                return {"answer": ans["answer"], "meta": ans, "chat_id": request.chat_id}

            if wants_plot:
                payload = CombinePayload(file_paths=request.combine_docs, question=question, chat_id=request.chat_id)
                res = plot_excel_data_combine(payload)
                ans = "Here is the combined plot based on your selected files."
                return {"answer": ans, **res}

        # ---- Single specific document (legacy)
        if request.document_id:
            ext = _ext(request.document_id)

            if ext in VIZ_DATA_EXTS and stat_intent:
                resolved_path = _resolve_excel_file(request.document_id, request.chat_id)
                df = _read_df_any(resolved_path, question)
                ans = _compute_stat_from_df(df, question)
                return {"answer": ans["answer"], "meta": ans, "chat_id": request.chat_id, "dataset_path": resolved_path}

            if wants_plot and ext in VIZ_DATA_EXTS:
                res = plot_excel_data(
                    file_path=request.document_id,
                    question=question,
                    title=None,
                    chat_id=request.chat_id,
                )
                ans = f"Here is the plot for “{request.document_id}”."
                return {"answer": ans, **res}
            else:
                answer = qa_pipeline.run(
                    question=question,
                    chat_id=request.chat_id,
                    document_id=request.document_id,
                    combine_docs=[],
                )
                if isinstance(answer, tuple):
                    answer = answer[0]
                return {"answer": str(answer or "")}

        # ---- Plot intent without explicit file
        if wants_plot:
            try:
                res = plot_excel_data(
                    file_path=None,
                    question=question,
                    title=None,
                    chat_id=request.chat_id,
                )
                return {"answer": "Here is the plot.", **res}
            except Exception:
                pass

        # ---- Stats intent without explicit file
        if stat_intent:
            try:
                resolved_path = _resolve_excel_file(None, request.chat_id)
                df = _read_df_any(resolved_path, question)
                ans = _compute_stat_from_df(df, question)
                return {"answer": ans["answer"], "meta": ans, "chat_id": request.chat_id, "dataset_path": resolved_path}
            except Exception as e:
                logger.warning(f"Stat attempt failed: {e}")

        # ---- Generic QA (pipeline decides)
        answer = qa_pipeline.run(
            question=question,
            chat_id=request.chat_id,
            document_id=None,
            combine_docs=[],
        )
        if isinstance(answer, tuple):
            answer = answer[0]
        return {"answer": str(answer or "")}
    except HTTPException:
        raise
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# ---------- health ----------
@app.get("/api/health")
async def health():
    return {"status": "ok"}

# Gemini key health (useful to debug API-only flow)
@app.get("/api/health/gemini")
async def health_gemini():
    k = _get_gemini_key()
    return {"ok": bool(k), "has_key": bool(k), "uses": "GEMINI_API_KEY or GOOGLE_API_KEY"}

# Ollama health placeholder (feature flag only)
ENABLE_OLLAMA = (os.getenv("ENABLE_OLLAMA", "").lower() in ("1", "true", "yes"))

@app.get("/api/health/ollama")
async def ollama_health():
    ok = bool(ENABLE_OLLAMA)
    return {"ok": ok, "enabled": ENABLE_OLLAMA, "note": "Set ENABLE_OLLAMA=true to enable local vision routes."}

# ==========================================
# === Generic chat message + images (save) ==
# ==========================================
CHAT_MESSAGES: Dict[str, list] = {}

class ChatMessageOut(BaseModel):
    message_id: str
    chat_id: str
    text: Optional[str] = ""
    attachments: List[Dict[str, str]] = Field(default_factory=list)
    created_at: str

@app.post("/api/chat", response_model=ChatMessageOut)
async def post_chat_message(
    chat_id: str = Form(...),
    text: Optional[str] = Form(""),
    files: Optional[List[UploadFile]] = File(None),
):
    """
    Accepts free-form chat text + multiple images.
    Saves images under /static/uploads/<chat_id>/ and returns public URLs for immediate UI preview.
    Also (best-effort) logs a bare contact row per image to DB so images are traceable in SQL.
    """
    try:
        attachments: List[Dict[str, str]] = []
        urls_for_db: List[str] = []
        if files:
            for up in files:
                url = _save_image_to_static(chat_id, up)
                attachments.append({"filename": _safe_name(up.filename or "image"), "url": url})
                urls_for_db.append(url)

        msg = {
            "message_id": uuid.uuid4().hex,
            "chat_id": chat_id,
            "text": text or "",
            "attachments": attachments,
            "created_at": datetime.utcnow().isoformat(),
        }
        CHAT_MESSAGES.setdefault(chat_id, []).append(msg)

        # Optional: persist an empty contact row so image refs exist in DB (even without extraction)
        if DBHandler is not None and urls_for_db:
            try:
                db = DBHandler()
                db.save_business_contacts(
                    contacts=[{"raw_text": (text or "")}],
                    chat_id=chat_id,
                    source="chat",
                    attachment_urls=urls_for_db,
                )
            except Exception as e:
                logger.warning(f"DB image-log skipped: {e}")

        return ChatMessageOut(**msg)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Chat upload failed")
        raise HTTPException(status_code=500, detail=f"Chat upload failed: {e}")

@app.get("/api/chat/history")
async def chat_history(chat_id: str = Query(...)):
    """Return all messages (text + image URLs) for a chat."""
    return {"chat_id": chat_id, "messages": CHAT_MESSAGES.get(chat_id, [])}

# ============================================================
# === Fallback Vision route (only if image_routes missing) ===
# ============================================================
if image_router is None:
    @app.post("/api/ask-image")
    async def ask_image_fallback(
        prompt: Optional[str] = Form(""),
        chat_id: Optional[str] = Form(None),
        images: Optional[List[UploadFile]] = File(None),
        images_array: Optional[List[UploadFile]] = File(None, alias="images[]"),
        front_image: Optional[UploadFile] = File(None),
        back_image: Optional[UploadFile] = File(None),
        frontFile: Optional[UploadFile] = File(None),
        backFile: Optional[UploadFile] = File(None),
        file: Optional[UploadFile] = File(None),
    ):
        """
        Fallback implementation that accepts many field-name variants:
        - images / images[] (multiple)
        - front_image / back_image
        - frontFile / backFile
        - file
        Saves images to /static and returns their URLs. Also logs placeholder rows in DB.
        """
        try:
            uploads: List[UploadFile] = []
            for group in (images, images_array):
                if group:
                    uploads.extend(group)
            for single in (front_image, back_image, frontFile, backFile, file):
                if single:
                    uploads.append(single)

            if not uploads:
                raise HTTPException(
                    status_code=422,
                    detail="No image(s) received. Send 'images', 'images[]', 'front_image', 'back_image', 'frontFile', 'backFile' or 'file'.",
                )

            cid = chat_id or "vision"
            image_urls: List[str] = []
            saved = 0
            pdfs = 0

            for up in uploads:
                ext = _ext(up.filename or "")
                if ext in _ALLOWED_IMAGE_EXT and (up.content_type or "").startswith("image/"):
                    url = _save_image_to_static(cid, up)
                    image_urls.append(url)
                    saved += 1
                elif ext == ".pdf":
                    cid_dir = UPLOAD_IMAGES_DIR / cid
                    cid_dir.mkdir(parents=True, exist_ok=True)
                    fname = f"{uuid.uuid4().hex}{ext}"
                    with open(cid_dir / fname, "wb") as f:
                        shutil.copyfileobj(up.file, f)
                    image_urls.append(f"/static/uploads/{cid}/{fname}")
                    pdfs += 1
                else:
                    raise HTTPException(status_code=400, detail=f"Unsupported file type: {up.filename}")

            # best-effort DB log (no fields yet, just attachments)
            if DBHandler is not None and image_urls:
                try:
                    db = DBHandler()
                    db.save_business_contacts(
                        contacts=[{"raw_text": (prompt or "")}],
                        chat_id=cid,
                        source="chat",
                        attachment_urls=image_urls,
                    )
                except Exception as e:
                    logger.warning(f"DB image-log skipped: {e}")

            text = (prompt or "").strip()
            msg = f"Processed {saved} image(s){(' and ' + str(pdfs) + ' PDF(s)') if pdfs else ''}. Prompt: {text}"

            return {
                "status": "ok",
                "data": {
                    "whatsapp": msg,
                    "json": None,
                    "image_urls": image_urls,
                },
                "meta": {
                    "received": len(uploads),
                    "saved_images": saved,
                    "saved_pdfs": pdfs,
                    "chat_id": cid,
                },
            }
        except HTTPException:
            raise
        except Exception as e:
            logger.exception("ask-image fallback failed")
            raise HTTPException(status_code=500, detail=f"ask-image failed: {e}")

# ================================
# === Gemini Card Endpoints ======
# ================================
class VCardJson(BaseModel):
    card: BusinessCard  # uses external or inline schema

@app.post("/api/cards/extract")
async def extract_card_api(
    file: UploadFile = File(...),
    return_vcard: bool = Form(True),
    chat_id: Optional[str] = Form(None),
):
    """
    Extract contact fields from a business card image using Google Gemini.
    - Saves the uploaded image into /static/uploads/<chat_id|cards> for traceability
    - Extracts structured JSON
    - Optionally writes a .vcf under /static/vcards
    - ✅ Persists the contact to Postgres (card_contacts + child tables) when DBHandler is available
    """
    try:
        if not _get_gemini_key():
            raise HTTPException(status_code=500, detail="Gemini API key missing. Set GEMINI_API_KEY or GOOGLE_API_KEY in backend .env and restart.")

        # Read raw bytes and persist original image for reference
        raw_bytes = await file.read()
        saved_image_url = _save_bytes_to_static(chat_id or "cards", file.filename, raw_bytes)

        # Preprocess for better OCR
        mime = file.content_type or mimetypes.guess_type(file.filename or "")[0] or "image/jpeg"
        proc_bytes, proc_mime = _preprocess_card_image(raw_bytes)

        # Extract
        card: BusinessCard = extract_business_card(proc_bytes, mime_type=proc_mime)

        # Build vCard (optional)
        vcard_txt = None
        vcard_url = None
        if return_vcard:
            fn = (card.full_name or "contact").replace(" ", "_") or "contact"
            vcf_name = f"{fn}.vcf"
            vcard_txt = make_vcard(
                full_name=card.full_name or "",
                last_name=card.last_name or "",
                first_name=card.first_name or "",
                org=card.organization,
                title=card.job_title,
                phones=card.phones,
                emails=card.emails,
                websites=card.websites,
                address=card.address.model_dump() if card.address else None,
                notes=card.notes,
            )
            (VCARD_DIR / vcf_name).write_text(vcard_txt, encoding="utf-8")
            vcard_url = f"/static/vcards/{vcf_name}"

        # ✅ Save to DB
        db_saved_ids: List[str] = []
        if DBHandler is not None:
            try:
                db = DBHandler()
                db_saved_ids = db.save_business_contacts(
                    contacts=[card.model_dump()],
                    chat_id=chat_id,
                    source="chat",  # or "viz"
                    attachment_urls=[saved_image_url],
                )
            except Exception as e:
                logger.exception(f"DB save failed: {e}")

        return {
            "ok": True,
            "card": card.model_dump(),
            "vcard": vcard_txt,
            "vcard_url": vcard_url,
            "image_url": saved_image_url,
            "db_ids": db_saved_ids,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Card extraction failed")
        raise HTTPException(status_code=500, detail=f"Card extraction failed: {e}")

@app.post("/api/cards/from-json")
async def vcard_from_json(payload: VCardJson):
    """
    Build a .vcf from a JSON card payload (e.g., after manual edits on UI).
    Saves to /static/vcards and returns the link.
    """
    try:
        c = payload.card
        vcard_txt = make_vcard(
            full_name=c.full_name or "",
            last_name=c.last_name or "",
            first_name=c.first_name or "",
            org=c.organization,
            title=c.job_title,
            phones=c.phones,
            emails=c.emails,
            websites=c.websites,
            address=c.address.model_dump() if c.address else None,
            notes=c.notes,
        )
        fn = (c.full_name or "contact").replace(" ", "_") or "contact"
        vcf_path = VCARD_DIR / f"{fn}.vcf"
        vcf_path.write_text(vcard_txt, encoding="utf-8")
        return {"ok": True, "vcard": vcard_txt, "vcard_url": f"/static/vcards/{fn}.vcf"}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("vCard creation failed")
        raise HTTPException(status_code=500, detail=f"vCard creation failed: {e}")

# ================================
# === Optional: URL-based Vision ==
# ================================
@app.post("/api/vision/ask")
async def vision_ask(payload: Dict[str, Any] = Body(...)):
    """
    Lightweight JSON-based vision ask:
    { "prompt": "...", "image_urls": ["https://...", "/static/uploads/..."] }
    This does not run OCR here; it simply echoes payload structure so the FE flow won't break
    even if this endpoint isn't used.
    """
    prompt = (payload or {}).get("prompt") or ""
    image_urls = (payload or {}).get("image_urls") or []
    return {"ok": True, "prompt": prompt, "image_urls": image_urls, "note": "Stub endpoint; hook your vision logic here if needed."}
