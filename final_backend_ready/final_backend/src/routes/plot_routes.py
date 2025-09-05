# src/routes/plot_routes.py

from __future__ import annotations

from typing import Optional, List, Dict, Any
from fastapi import APIRouter, UploadFile, File, Form, Query, HTTPException, Request
from fastapi.responses import FileResponse
from pathlib import Path
import os, uuid, shutil, csv, json, re, difflib

import pandas as pd
from src.pipeline.plot_pipeline import PlotGenerationPipeline

# Optional DB integration (fails softly if not present)
try:
    from src.db_handler import DBHandler  # type: ignore
except Exception:
    DBHandler = None  # type: ignore

router = APIRouter(prefix="/api/visualizations", tags=["visualizations"])
pipeline = PlotGenerationPipeline()

ALLOWED_EXCEL_EXTS = (".xlsx", ".xls", ".csv")
UPLOAD_DIR = Path("uploaded_excels")  # Excel/CSV base (per-chat subfolders)
(UPLOAD_DIR / "_tmp").mkdir(parents=True, exist_ok=True)

# ========================
# Path resolving helpers
# ========================

def _is_excel_name(name: str) -> bool:
    return str(name).lower().endswith(ALLOWED_EXCEL_EXTS)

def _find_excel_in_dir(d: Path) -> Optional[Path]:
    if not d.exists() or not d.is_dir():
        return None
    # Prefer top-level files first
    for p in d.iterdir():
        if p.is_file() and _is_excel_name(p.name):
            return p
    # Then try deep search
    for ext in ALLOWED_EXCEL_EXTS:
        found = list(d.rglob(f"*{ext}"))
        if found:
            return found[0]
    return None

def _resolve_excel_path(file_path: str, chat_id: Optional[str]) -> Path:
    """
    Accepts absolute path OR a relative filename (e.g., 'Sales.xlsx').
    If relative, tries uploaded_excels/<chat_id>/ and then uploaded_excels/.
    Also accepts a directory and picks the first Excel/CSV inside it.
    """
    if not file_path:
        raise HTTPException(status_code=400, detail="file_path is required.")

    p = Path(file_path)

    # 1) Absolute or already-resolvable relative to CWD
    if p.is_absolute() or p.exists():
        if not p.exists():
            raise HTTPException(status_code=404, detail=f"File or directory not found: {p}")
        if p.is_dir():
            pick = _find_excel_in_dir(p)
            if not pick:
                raise HTTPException(status_code=404, detail=f"No Excel/CSV found inside: {p}")
            return pick
        if not _is_excel_name(p.name):
            raise HTTPException(status_code=400, detail="Unsupported file type (must be .xlsx, .xls, .csv).")
        return p

    # 2) Relative -> search under chat folder then root UPLOAD_DIR
    bases: List[Path] = []
    if chat_id:
        bases.append(UPLOAD_DIR / chat_id)
    bases.append(UPLOAD_DIR)

    name_has_ext = any(str(p).lower().endswith(ext) for ext in ALLOWED_EXCEL_EXTS)

    for b in bases:
        cand = (b / file_path).resolve()
        if cand.exists():
            if cand.is_dir():
                pick = _find_excel_in_dir(cand)
                if pick:
                    return pick
            elif _is_excel_name(cand.name):
                return cand

        # try with implicit extensions if none provided
        if not name_has_ext:
            for ext in ALLOWED_EXCEL_EXTS:
                cand2 = (b / f"{file_path}{ext}").resolve()
                if cand2.exists() and cand2.is_file():
                    return cand2

    search_roots = ", ".join(str(b) for b in bases)
    raise HTTPException(status_code=404, detail=f"Could not resolve '{file_path}' in: {search_roots}")

# ---------- question parsing: sheet name ----------
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
            s = (m.group(1) or "").strip()
            if s and s.lower() not in {"the", "first", "sheet"}:
                return s
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

# ---------- Safe table reader (now honors sheet in question) ----------
def _sniff_delimiter(path: Path) -> Optional[str]:
    try:
        sample = path.open("r", encoding="utf-8", errors="ignore").read(8192)
        return csv.Sniffer().sniff(sample).delimiter
    except Exception:
        return None

def _read_table_safely(path: Path, question: Optional[str] = None) -> pd.DataFrame:
    """
    Read CSV/XLS/XLSX robustly for quick validation.
    Honors sheet selection parsed from the question for Excel files.
    """
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    suf = path.suffix.lower()
    if suf == ".csv":
        sep = _sniff_delimiter(path)
        try:
            df = pd.read_csv(
                path,
                sep=sep,
                engine="python",
                on_bad_lines="skip",
                encoding="utf-8-sig",
            )
        except UnicodeDecodeError:
            df = pd.read_csv(
                path,
                sep=sep,
                engine="python",
                on_bad_lines="skip",
                encoding="latin1",
            )
    elif suf in (".xlsx", ".xls"):
        xl = pd.ExcelFile(path)
        if not xl.sheet_names:
            raise ValueError("Excel file has no sheets.")
        want_sheet = _parse_sheet_from_question(question or "")
        sname = _pick_sheet(xl, want_sheet)
        df = pd.read_excel(xl, sheet_name=sname)
    else:
        raise ValueError(f"Unsupported file type: {suf}")

    # guard for truly empty frames
    if df is None or df.empty or df.shape[0] == 0 or df.shape[1] == 0:
        raise ValueError("Parsed table is empty.")
    return df

# ---------- Optional DB persistence helpers (best-effort) ----------
def _db_save_plot_meta(meta: Dict[str, Any], chat_id: Optional[str]) -> None:
    """
    Save plot metadata to DB if DBHandler is available and has a suitable method.
    This function fails softly and never raises.
    """
    if not DBHandler:
        return
    try:
        db = DBHandler()
        if hasattr(db, "save_plot_meta"):
            try:
                db.save_plot_meta(meta, chat_id=chat_id)  # type: ignore[attr-defined]
                return
            except TypeError:
                db.save_plot_meta(meta)  # type: ignore[attr-defined]
                return

        if hasattr(db, "save_message_and_images"):
            pid = meta.get("id") or meta.get("plot_id")
            img_path = pipeline.get_image_path(pid) if pid else None
            db.save_message_and_images(  # type: ignore[attr-defined]
                chat_id=chat_id,
                source="viz",
                role="assistant",
                text=f"[Visualization] {meta.get('title') or meta.get('question') or pid}",
                image_urls=[img_path] if img_path else [],
            )
    except Exception:
        pass

# -------- List visualizations (with filters + URLs) --------
@router.get("/list")
def list_visualizations(
    request: Request,
    chat_id: Optional[str] = Query(None, description="Filter by chat_id"),
    q: Optional[str] = Query(None, description="Search in title or question"),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    order: str = Query("desc", pattern="^(asc|desc)$"),
):
    try:
        items = pipeline.list_meta() or []

        chat_ids = sorted({(i.get("chat_id") or "").strip() for i in items if i.get("chat_id")})

        # filters
        if chat_id:
            items = [m for m in items if m.get("chat_id") == chat_id]

        if q:
            ql = q.lower()

            def hit(m):
                return (
                    (m.get("title") or "").lower().find(ql) >= 0
                    or (m.get("question") or "").lower().find(ql) >= 0
                    or (m.get("chat_id") or "").lower().find(ql) >= 0
                )

            items = [m for m in items if hit(m)]

        # sort + paginate
        items.sort(key=lambda m: m.get("created_at") or "", reverse=(order == "desc"))
        total = len(items)
        items = items[offset : offset + limit]

        # add URLs FE can use directly
        for m in items:
            pid = m.get("id") or m.get("plot_id")
            m["image_url"] = f"/api/visualizations/{pid}/image"
            m["thumb_url"] = f"/api/visualizations/{pid}/thumb"

        return {"items": items, "total": total, "chat_ids": chat_ids}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# -------- Single-file visualization --------
@router.post("/generate")
def generate_visualization(
    question: str = Form(...),
    title: Optional[str] = Form(None),
    chat_id: Optional[str] = Form(None),
    file: UploadFile = File(None),
    file_path: Optional[str] = Form(None),
):
    """
    Accept either a file upload (CSV/XLS/XLSX) or a server-side file_path.
    If file_path is a relative name, it will be resolved under:
      uploaded_excels/<chat_id>/  then  uploaded_excels/
    Returns metadata including image/thumb URLs.
    """
    if not file and not file_path:
        raise HTTPException(status_code=400, detail="Either 'file' or 'file_path' is required.")

    tmp_path = None
    try:
        # handle upload
        if file:
            ext = os.path.splitext(file.filename or "")[-1].lower()
            if ext not in ALLOWED_EXCEL_EXTS:
                raise HTTPException(status_code=400, detail="Only .xlsx, .xls, or .csv files are supported.")
            tmp_path = UPLOAD_DIR / "_tmp" / f"{uuid.uuid4().hex}_{file.filename}"
            with open(tmp_path, "wb") as f:
                shutil.copyfileobj(file.file, f)
            resolved = tmp_path
        else:
            # resolve relative 'file_path' using chat_id
            resolved = _resolve_excel_path(file_path, chat_id)

        # pre-validate with the SAME sheet selection logic as pipeline
        try:
            _ = _read_table_safely(Path(resolved), question=question)
        except (FileNotFoundError, ValueError) as e:
            raise HTTPException(status_code=422, detail=f"Cannot read the selected file: {e}")

        # pipeline call (pipeline will also honor sheet selection)
        try:
            meta = pipeline.generate_and_store(
                file_path=str(resolved),
                question=question,
                title=title,
                chat_id=chat_id,
            )
        except Exception as e:
            msg = str(e)
            if "No objects to concatenate" in msg or "empty" in msg.lower():
                raise HTTPException(
                    status_code=422,
                    detail="No usable table found in the file. Check delimiter/columns.",
                )
            raise

        # add URLs
        pid = meta.get("id") or meta.get("plot_id")
        meta["image_url"] = f"/api/visualizations/{pid}/image"
        meta["thumb_url"] = f"/api/visualizations/{pid}/thumb"

        # best-effort DB persistence
        _db_save_plot_meta(meta, chat_id)

        return meta

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Visualization failed: {e}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass

# -------- Multi-file (combine) visualization --------
@router.post("/generate-combined")
def generate_visualization_combined(
    question: str = Form(...),
    title: Optional[str] = Form(None),
    chat_id: Optional[str] = Form(None),
    file_paths: Optional[str] = Form(None),
):
    """
    Combine multiple server-side files.
    Provide file_paths as a JSON list string or a comma-separated string.
    Each entry may be absolute, or a relative name resolved under:
      uploaded_excels/<chat_id>/  then  uploaded_excels/
    """
    if not file_paths:
        raise HTTPException(status_code=400, detail="'file_paths' is required for combined generation.")

    # parse list
    try:
        if file_paths.strip().startswith("["):
            raw_paths: List[str] = json.loads(file_paths)
        else:
            raw_paths = [p.strip() for p in file_paths.split(",") if p.strip()]
    except Exception:
        raise HTTPException(
            status_code=400, detail="Invalid 'file_paths' format. Send JSON array or comma-separated list."
        )

    if len(raw_paths) < 2:
        raise HTTPException(status_code=400, detail="Provide at least two file paths to combine.")

    # resolve + validate each before pipeline (so we return 422, not 500)
    usable: List[str] = []
    errs: List[str] = []
    for p in raw_paths:
        try:
            resolved = _resolve_excel_path(p, chat_id)
            _ = _read_table_safely(Path(resolved), question=question)  # validate honoring sheet selection
            usable.append(str(resolved))
        except Exception as e:
            errs.append(f"{p}: {e}")

    if len(usable) == 0:
        raise HTTPException(status_code=422, detail="No usable tables found across the selected files.")
    if len(usable) == 1:
        # fallback to single-file behaviour
        try:
            meta = pipeline.generate_and_store(
                file_path=usable[0],
                question=question,
                title=title,
                chat_id=chat_id,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Visualization failed: {e}")
    else:
        try:
            meta = pipeline.generate_and_store_combine(
                file_paths=usable,
                question=question,
                title=title,
                chat_id=chat_id,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Visualization failed: {e}")

    pid = meta.get("id") or meta.get("plot_id")
    meta["image_url"] = f"/api/visualizations/{pid}/image"
    meta["thumb_url"] = f"/api/visualizations/{pid}/thumb"
    if errs:
        # Include non-fatal file errors so caller can show a warning
        meta["warnings"] = errs

    # best-effort DB persistence
    _db_save_plot_meta(meta, chat_id)

    return meta

# -------- Image endpoints (full + thumb) --------
@router.get("/{plot_id}/image")
def get_image(plot_id: str):
    path = pipeline.get_image_path(plot_id)
    if not path or not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Image not found.")
    return FileResponse(path, media_type="image/png")

@router.get("/{plot_id}/thumb")
def get_thumb(plot_id: str):
    path = pipeline.get_thumb_path(plot_id)
    if not path or not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Thumbnail not found.")
    return FileResponse(path, media_type="image/png")
