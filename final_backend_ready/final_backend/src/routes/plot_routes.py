from typing import Optional
from fastapi import APIRouter, UploadFile, File, Form, Query, HTTPException, Request
from fastapi.responses import FileResponse
import os, uuid, shutil

from src.pipeline.plot_pipeline import PlotGenerationPipeline

router = APIRouter(prefix="/api/visualizations", tags=["visualizations"])
pipeline = PlotGenerationPipeline()

ALLOWED_EXCEL_EXTS = (".xlsx", ".xls", ".csv")

# -------- List visualizations (with filters + URLs your UI can use) --------
@router.get("/list")
def list_visualizations(
    request: Request,
    chat_id: Optional[str] = Query(None, description="Filter by chat_id"),
    q: Optional[str] = Query(None, description="Search in title or question"),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    order: str = Query("desc", pattern="^(asc|desc)$"),
):
    """
    Returns:
      {
        "items": [{...,"image_url":"/api/visualizations/{id}/image","thumb_url":"/api/visualizations/{id}/thumb"}],
        "total": <int>,
        "chat_ids": ["chat-a","chat-b", ...]
      }
    """
    try:
        items = pipeline.list_meta() or []

        # collect available chats for filter dropdown
        chat_ids = sorted({(i.get("chat_id") or "").strip() for i in items if i.get("chat_id")})

        # filter
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

        # sort by created_at if present
        items.sort(key=lambda m: m.get("created_at") or "", reverse=(order == "desc"))

        total = len(items)
        items = items[offset: offset + limit]

        # add URLs the FE can render directly
        for m in items:
            pid = m.get("id") or m.get("plot_id")
            m["image_url"] = f"/api/visualizations/{pid}/image"
            m["thumb_url"] = f"/api/visualizations/{pid}/thumb"

        return {"items": items, "total": total, "chat_ids": chat_ids}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# -------- Generate a visualization from Excel/CSV path or uploaded file --------
@router.post("/generate")
def generate_visualization(
    question: str = Form(...),
    title: Optional[str] = Form(None),
    chat_id: Optional[str] = Form(None),
    file: UploadFile = File(None),
    file_path: Optional[str] = Form(None),
):
    """
    You can send either a file upload (Excel/CSV) or an existing file_path on disk.
    Returns the saved metadata with image/thumb URLs.
    """
    if not file and not file_path:
        raise HTTPException(status_code=400, detail="Either 'file' or 'file_path' is required.")

    tmp_path = None
    try:
        # If uploaded, save to a temp path
        if file:
            ext = os.path.splitext(file.filename or "")[-1].lower()
            if ext not in ALLOWED_EXCEL_EXTS:
                raise HTTPException(status_code=400, detail="Only .xlsx, .xls, or .csv files are supported.")
            os.makedirs("uploaded_excels/_tmp", exist_ok=True)
            tmp_path = os.path.join("uploaded_excels/_tmp", f"{uuid.uuid4().hex}_{file.filename}")
            with open(tmp_path, "wb") as f:
                shutil.copyfileobj(file.file, f)
            file_path = tmp_path
        else:
            # validate extension if path was provided
            ext = os.path.splitext(file_path or "")[-1].lower()
            if ext not in ALLOWED_EXCEL_EXTS:
                raise HTTPException(status_code=400, detail="file_path must be .xlsx, .xls, or .csv")

        meta = pipeline.generate_and_store(
            file_path=file_path,
            question=question,
            title=title,
            chat_id=chat_id
        )

        # enrich with URLs
        pid = meta.get("id") or meta.get("plot_id")
        meta["image_url"] = f"/api/visualizations/{pid}/image"
        meta["thumb_url"] = f"/api/visualizations/{pid}/thumb"

        return meta
    finally:
        # best-effort cleanup for uploaded tmp file
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass

# -------- Image endpoints (full + thumb) --------
@router.get("/{plot_id}/image")
def get_image(plot_id: str):
    path = pipeline.get_image_path(plot_id)
    if not path or not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Image not found.")
    return FileResponse(path, media_type="image/png")

@router.get("/{plot_id}/thumb}")
def get_thumb(plot_id: str):
    path = pipeline.get_thumb_path(plot_id)
    if not path or not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Thumbnail not found.")
    return FileResponse(path, media_type="image/png")
