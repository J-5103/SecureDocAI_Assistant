# src/routes/ploe.py
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse
import os
import shutil
from typing import Optional, Dict, Any, List
from src.pipeline.plot_pipeline import PlotGenerationPipeline

router = APIRouter(prefix="/api/visualizations", tags=["visualizations"])
plot_pipeline = PlotGenerationPipeline()

UPLOAD_EXCEL = "uploaded_excels"
os.makedirs(UPLOAD_EXCEL, exist_ok=True)

@router.post("/generate")
async def generate_plot(
    question: str = Form(...),
    title: Optional[str] = Form(None),
    file_path: Optional[str] = Form(None),
    file: UploadFile = File(None),
) -> Dict[str, Any]:
    """
    Generate a plot and persist it to the visualization store.
    Accepts either:
      - file_path (returned by /api/excel/upload/)
      - or an uploaded file
    """
    try:
        resolved_path: Optional[str] = None

        if file is not None:
            ext = os.path.splitext(file.filename)[-1].lower()
            if ext not in [".csv", ".xls", ".xlsx"]:
                raise HTTPException(status_code=400, detail="Only CSV/XLS/XLSX allowed.")
            save_path = os.path.join(UPLOAD_EXCEL, file.filename)
            with open(save_path, "wb") as f:
                shutil.copyfileobj(file.file, f)
            resolved_path = save_path

        if not resolved_path and file_path:
            if not os.path.exists(file_path):
                raise HTTPException(status_code=404, detail="file_path not found.")
            resolved_path = file_path

        if not resolved_path:
            raise HTTPException(status_code=400, detail="Provide file or file_path.")

        meta = plot_pipeline.generate_and_store(resolved_path, question, title)
        return meta

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/list")
def list_plots() -> List[Dict[str, Any]]:
    return plot_pipeline.list_meta()

@router.get("/{plot_id}/image")
def get_image(plot_id: str):
    path = plot_pipeline.get_image_path(plot_id)
    if not path:
        raise HTTPException(status_code=404, detail="Not found")
    return FileResponse(path)

@router.get("/{plot_id}/thumb")
def get_thumb(plot_id: str):
    path = plot_pipeline.get_thumb_path(plot_id)
    if not path:
        raise HTTPException(status_code=404, detail="Not found")
    return FileResponse(path)
