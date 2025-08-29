# final_backend/src/routes/plot_routes.py

from typing import Optional, List
from fastapi import APIRouter, UploadFile, File, Form, Query, HTTPException, Request
from fastapi.responses import FileResponse
from pathlib import Path
import os, uuid, shutil, csv, json

import pandas as pd
from src.pipeline.plot_pipeline import PlotGenerationPipeline

router = APIRouter(prefix="/api/visualizations", tags=["visualizations"])
pipeline = PlotGenerationPipeline()

ALLOWED_EXCEL_EXTS = (".xlsx", ".xls", ".csv")
UPLOAD_DIR = Path("uploaded_excels")  # adjust if your project uses another folder


# ---------- Safe table reader (no separate io.py needed) ----------
def _sniff_delimiter(path: Path) -> Optional[str]:
    try:
        sample = path.open("r", encoding="utf-8", errors="ignore").read(8192)
        return csv.Sniffer().sniff(sample).delimiter
    except Exception:
        return None

def _read_table_safely(path: Path) -> pd.DataFrame:
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
        df = pd.read_excel(xl, sheet_name=0)
    else:
        raise ValueError(f"Unsupported file type: {suf}")

    # guard for truly empty frames
    if df is None or df.empty or df.shape[0] == 0 or df.shape[1] == 0:
        raise ValueError("Parsed table is empty.")
    return df
# -------------------------------------------------------------------


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
        items = items[offset: offset + limit]

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
            (UPLOAD_DIR / "_tmp").mkdir(parents=True, exist_ok=True)
            tmp_path = UPLOAD_DIR / "_tmp" / f"{uuid.uuid4().hex}_{file.filename}"
            with open(tmp_path, "wb") as f:
                shutil.copyfileobj(file.file, f)
            file_path = str(tmp_path)
        else:
            ext = os.path.splitext(file_path or "")[-1].lower()
            if ext not in ALLOWED_EXCEL_EXTS:
                raise HTTPException(status_code=400, detail="file_path must be .xlsx, .xls, or .csv")

        # pre-validate (prevents pandas concat empty errors bubbling as 500)
        try:
            _ = _read_table_safely(Path(file_path))
        except (FileNotFoundError, ValueError) as e:
            raise HTTPException(status_code=422, detail=f"Cannot read the selected file: {e}")

        # pipeline call
        try:
            meta = pipeline.generate_and_store(
                file_path=file_path,
                question=question,
                title=title,
                chat_id=chat_id,
            )
        except Exception as e:
            msg = str(e)
            if "No objects to concatenate" in msg or "empty" in msg.lower():
                raise HTTPException(status_code=422, detail="No usable table found in the file. Check delimiter/columns.")
            raise

        # add URLs
        pid = meta.get("id") or meta.get("plot_id")
        meta["image_url"] = f"/api/visualizations/{pid}/image"
        meta["thumb_url"] = f"/api/visualizations/{pid}/thumb"
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
    """
    if not file_paths:
        raise HTTPException(status_code=400, detail="'file_paths' is required for combined generation.")

    # parse list
    try:
        if file_paths.strip().startswith("["):
            paths: List[str] = json.loads(file_paths)
        else:
            paths = [p.strip() for p in file_paths.split(",") if p.strip()]
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid 'file_paths' format. Send JSON array or comma-separated list.")

    if len(paths) < 2:
        raise HTTPException(status_code=400, detail="Provide at least two file paths to combine.")

    # validate each before pipeline (so we return 422, not 500)
    usable = []
    errs = []
    for p in paths:
        try:
            _ = _read_table_safely(Path(p))
            usable.append(p)
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
