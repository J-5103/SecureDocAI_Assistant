# src/main.py
import os, shutil, re, uuid, json, base64
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime
from time import perf_counter
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Body, Query, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import logging

# NEW: data utils
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from src.pipeline.question_answer_pipeline import QuestionAnswerPipeline
try:
    from src.components.rag_pipeline import RAGPipeline
except ImportError as e:
    logger.error(f"Failed to import RAGPipeline: {str(e)}")
    raise

from src.components.plot_generator import PlotGenerator  # noqa: F401
from src.components.file_loader import FileLoader        # noqa: F401
from src.pipeline.plot_pipeline import PlotGenerationPipeline

# If you already expose visualization routes elsewhere, you can still keep this:
try:
    from src.routes.plot_routes import router as visualizations_router
except Exception:
    visualizations_router = None

# === NEW (Vision): import Ollama+LLaVA pipeline ===
try:
    from src.pipeline.image_question_pipeline import ImageQuestionPipeline
except Exception as _e:
    logger.warning(f"Vision pipeline not available yet: { _e }")
    ImageQuestionPipeline = None  # type: ignore

# ----- FS layout (make static dir before mounting!) -----
PROJECT_ROOT = Path(__file__).resolve().parents[1]
STATIC_ROOT = PROJECT_ROOT / "static"
STATIC_ROOT.mkdir(parents=True, exist_ok=True)

STATIC_VIS_DIR = STATIC_ROOT / "visualizations"
STATIC_VIS_DIR.mkdir(parents=True, exist_ok=True)

# NEW: persistent uploads for image previews
UPLOAD_IMAGES_DIR = STATIC_ROOT / "uploads"
UPLOAD_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

UPLOAD_BASE = "uploaded_docs"     # PDFs/Docs/Images (per chat_id)
UPLOAD_EXCEL = "uploaded_excels"  # Excel/CSV     (per chat_id OR loose files)
os.makedirs(UPLOAD_BASE, exist_ok=True)
os.makedirs(UPLOAD_EXCEL, exist_ok=True)

# ---------- init ----------
qa_pipeline = QuestionAnswerPipeline()
rag_pipeline = RAGPipeline()
plot_pipeline = PlotGenerationPipeline()

app = FastAPI(title="SecureDocAI Backend", version="1.2.1")

# -------- CORS (UPDATED) --------
_frontend_origins = os.getenv("FRONTEND_ORIGINS")
if _frontend_origins:
    _allowed_origins = [o.strip() for o in _frontend_origins.split(",") if o.strip()]
else:
    _allowed_origins = [
        "http://192.168.0.190:3000",
        "http://192.168.0.109:3000",
        "http://192.168.0.109:5173",
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

# Try mounting your router if present
if visualizations_router:
    app.include_router(visualizations_router)

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

# =========================
# NEW: Stats utilities
# =========================

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

def _read_df_any(path: str) -> pd.DataFrame:
    ext = _ext(path)
    if ext == ".csv":
        return pd.read_csv(path)
    try:
        return pd.read_excel(path)
    except Exception:
        return pd.read_excel(path, engine="openpyxl")

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
        frames.append(_read_df_any(p))
    df = pd.concat(frames, ignore_index=True, sort=False)
    return _compute_stat_from_df(df, question)

# ---------- uploads (Chat Sessions: all supported types) ----------
@app.post("/api/upload/upload_file")
async def upload_file(
    chat_id: str = Form(...),
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None
):
    """
    Chat Sessions uploader — SUPPORTS: .pdf, .doc, .docx, .xls, .xlsx, .csv, .png, .jpg, .jpeg
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

        saved_path = os.path.join(chat_folder, safe_name)
        with open(saved_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        doc_base_name = os.path.splitext(safe_name)[0]
        vector_store_path = os.path.join("vectorstores", chat_id, doc_base_name)
        os.makedirs(os.path.dirname(vector_store_path), exist_ok=True)

        task_id = str(uuid.uuid4())
        update_task_status(task_id, "pending", progress=0)
        logger.info(f"Upload received for {safe_name}. Task ID: {task_id}")

        async def create_vectorstore_task():
            try:
                update_task_status(task_id, "processing", progress=10)
                logger.info(f"Starting vectorstore creation for: {safe_name} at {vector_store_path}")
                rag_pipeline.create_vectorstore(file_path=saved_path, vector_store_path=vector_store_path)
                update_task_status(task_id, "ready", progress=100, document_id=doc_base_name)
                logger.info(f"Vectorstore created for {safe_name}. Task ID: {task_id}")
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
                "document_id": doc_base_name,
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
                "document_id": doc_base_name,
            }
    except OSError as e:
        logger.error(f"Directory or file access error: {str(e)}")
        return JSONResponse(status_code=500, content={"error": f"Directory or file access error: {str(e)}"})
    except Exception as e:
        logger.error(f"Upload failed: {str(e)}")
        return JSONResponse(status_code=500, content={"error": f"Upload failed: {str(e)}"})

# ---------- create vector store ----------
@app.post("/api/create-vector-store")
async def create_vector_store(
    chat_id: str = Form(...),
    file_id: str = Form(...),
    background_tasks: BackgroundTasks = None
):
    try:
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
                logger.info(f"Starting vectorstore creation for: {file_id} at {vector_store_path}")
                rag_pipeline.create_vectorstore(file_path=file_path, vector_store_path=vector_store_path)
                update_task_status(task_id, "ready", progress=100, document_id=os.path.splitext(file_id)[0])
                logger.info(f"Vectorstore created for {file_id}. Task ID: {task_id}")
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
    intent: Optional[str] = None  # "qa" | "plot"

@app.get("/api/list_documents")
async def list_documents(chat_id: str = Query(...)):
    try:
        folder = os.path.join(UPLOAD_BASE, chat_id)
        if not os.path.isdir(folder):
            return {"documents": []}
        docs = []
        for f in os.listdir(folder):
            if _is_chat_doc(f):
                p = os.path.join(folder, f)
                docs.append({"name": f, "documentId": f, "size": os.path.getsize(p)})
        return {"documents": docs}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

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

        if _detect_stat_intent(question):
            df = _read_df_any(resolved_path)
            ans = _compute_stat_from_df(df, question)
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
            resolved.append(str(p))

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
        raise HTTPException(status_code=500, detail=f"Combined plot failed: {e}")

# ---------- Visualizations: list + serve + generate ----------
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

            if _detect_stat_intent(question):
                df = _read_df_any(str(dest))
                ans = _compute_stat_from_df(df, question)
                return {"answer": ans["answer"], "meta": ans, "chat_id": cid, "dataset_path": str(dest)}

            meta = plot_pipeline.generate_and_store(str(dest), question, title=title, chat_id=cid)
            return meta

        if not file_path:
            raise HTTPException(status_code=400, detail="Either 'file' or 'file_path' is required.")
        resolved = _resolve_excel_file(file_path, chat_id)

        if _detect_stat_intent(question):
            df = _read_df_any(resolved)
            ans = _compute_stat_from_df(df, question)
            return {"answer": ans["answer"], "meta": ans, "chat_id": chat_id, "dataset_path": resolved}

        meta = plot_pipeline.generate_and_store(resolved, question, title=title, chat_id=chat_id)
        return meta
    except HTTPException:
        raise
    except Exception as e:
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
        if _detect_stat_intent(payload.question):
            df = _read_df_any(resolved_path)
            ans = _compute_stat_from_df(df, payload.question)
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
    except Exception as e:
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

        if request.document_id:
            ext = _ext(request.document_id)

            if ext in VIZ_DATA_EXTS and stat_intent:
                resolved_path = _resolve_excel_file(request.document_id, request.chat_id)
                df = _read_df_any(resolved_path)
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

        if stat_intent:
            try:
                resolved_path = _resolve_excel_file(None, request.chat_id)
                df = _read_df_any(resolved_path)
                ans = _compute_stat_from_df(df, question)
                return {"answer": ans["answer"], "meta": ans, "chat_id": request.chat_id, "dataset_path": resolved_path}
            except Exception as e:
                logger.warning(f"Stat attempt failed: {e}")

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

# ============================
# === NEW (Vision) Endpoints ==
# ============================
_VISION_OLLAMA_BASE = os.getenv("OLLAMA_BASE", "http://192.168.0.88:11434")
_LLaVA_MODEL_TAG = os.getenv("LLAVA_MODEL", "llava:13b")

if ImageQuestionPipeline is not None:
    try:
        image_pipeline = ImageQuestionPipeline(
            ollama_base=_VISION_OLLAMA_BASE,
            model_name=_LLaVA_MODEL_TAG
        )
    except TypeError:
        image_pipeline = ImageQuestionPipeline(
            ollama_url=f"{_VISION_OLLAMA_BASE}/api/generate",
            model_name=_LLaVA_MODEL_TAG
        )
else:
    image_pipeline = None

# simple in-memory visualization chat store
VISION_CHAT: Dict[str, list] = {}

class AskImageOut(BaseModel):
    status: str
    data: Dict[str, Any]
    session_id: str

@app.post("/api/ask-image", response_model=AskImageOut)
async def ask_image(
    front_image: UploadFile = File(...),
    back_image: Optional[UploadFile] = File(None),
    session_id: Optional[str] = Form(None)
):
    if image_pipeline is None:
        raise HTTPException(status_code=500, detail="Vision pipeline not initialized. Ensure src/pipeline/image_question_pipeline.py exists.")
    try:
        fb = await front_image.read()
        bb = await back_image.read() if back_image else None

        sid = session_id or str(uuid.uuid4())

        session_dir = UPLOAD_IMAGES_DIR / sid
        session_dir.mkdir(parents=True, exist_ok=True)

        def _safelike(n: str) -> str:
            return os.path.basename((n or "").replace("\\", "/"))

        ts = str(int(datetime.utcnow().timestamp()))
        saved_urls: List[str] = []

        fname_front = f"{ts}_front_{_safelike(front_image.filename)}"
        with open(session_dir / fname_front, "wb") as f:
            f.write(fb)
        saved_urls.append(f"/static/uploads/{sid}/{fname_front}")

        if bb:
            fname_back = f"{ts}_back_{_safelike(back_image.filename)}"
            with open(session_dir / fname_back, "wb") as f:
                f.write(bb)
            saved_urls.append(f"/static/uploads/{sid}/{fname_back}")

        out = image_pipeline.run(fb, bb)

    except Exception as e:
        logger.exception("Vision extraction failed")
        raise HTTPException(status_code=500, detail=f"Vision extraction failed: {e}")

    VISION_CHAT.setdefault(sid, []).append({
        "role": "user",
        "type": "image",
        "image_urls": saved_urls,
        "front_name": front_image.filename,
        "back_name": back_image.filename if back_image else None
    })
    VISION_CHAT[sid].append({
        "role": "assistant",
        "whatsapp": out.get("whatsapp") if isinstance(out, dict) else out,
        "vcard": out.get("vcard") if isinstance(out, dict) else "",
        "json": out.get("json") if isinstance(out, dict) else {},
    })

    data_with_urls = out if isinstance(out, dict) else {"whatsapp": out, "vcard": "", "json": {}}
    data_with_urls = {**data_with_urls, "image_urls": saved_urls}
    return AskImageOut(status="success", data=data_with_urls, session_id=sid)

@app.get("/api/chat/{session_id}")
async def get_chat(session_id: str):
    return {"session_id": session_id, "history": VISION_CHAT.get(session_id, [])}

# ==========================================
# === NEW: Generic chat message + images ===
# ==========================================
# In-memory chat store for generic text+images (separate from VISION_CHAT)
CHAT_MESSAGES: Dict[str, list] = {}

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

class ChatMessageOut(BaseModel):
    message_id: str
    chat_id: str
    text: Optional[str] = ""
    attachments: List[Dict[str, str]] = []
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
    """
    try:
        attachments: List[Dict[str, str]] = []
        if files:
            for up in files:
                url = _save_image_to_static(chat_id, up)
                attachments.append({"filename": _safe_name(up.filename or "image"), "url": url})

        msg = {
            "message_id": uuid.uuid4().hex,
            "chat_id": chat_id,
            "text": text or "",
            "attachments": attachments,
            "created_at": datetime.utcnow().isoformat(),
        }
        CHAT_MESSAGES.setdefault(chat_id, []).append(msg)
        return msg  # pydantic will coerce to ChatMessageOut
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Chat upload failed")
        raise HTTPException(status_code=500, detail=f"Chat upload failed: {e}")

@app.get("/api/chat/history")
async def chat_history(chat_id: str = Query(...)):
    """Return all messages (text + image URLs) for a chat."""
    return {"chat_id": chat_id, "messages": CHAT_MESSAGES.get(chat_id, [])}
