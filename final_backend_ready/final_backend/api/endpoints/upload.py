# src/routes/upload_routes.py

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
import os
import shutil
from src.pipeline.document_pipeline import DocumentPipeline

router = APIRouter(prefix="/upload")  # Final path: /api/upload/*

document_pipeline = DocumentPipeline()

UPLOAD_FOLDER = "uploaded_docs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@router.post("/upload_file")  # Final endpoint: /api/upload/upload_file
async def upload_file(chat_id: str = Form(...), file: UploadFile = File(...)):
    try:
        if not file.filename.endswith(".pdf", ".doc", ".docx", ".xls", ".xlsx", ".csv", ".png", ".jpg", ".jpeg"):
            raise HTTPException(status_code=400, detail="Only PDF files allowed.")

        # Normalize filename
        filename = file.filename.replace(" ", "_")
        chat_folder = os.path.join(UPLOAD_FOLDER, chat_id)
        os.makedirs(chat_folder, exist_ok=True)

        file_path = os.path.join(chat_folder, filename)
        with open(file_path, "wb") as f_out:
            shutil.copyfileobj(file.file, f_out)

        # Pass chat_id to pipeline
        document_pipeline.run(file_path, filename, chat_id=chat_id)

        return {
            "message": "File processed and saved.",
            "filename": filename,
            "chat_id": chat_id,
            "status": "success"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload error: {str(e)}")
