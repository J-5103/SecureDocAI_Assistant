from fastapi import APIRouter, UploadFile, File, Form, HTTPException
import os

router = APIRouter()

UPLOAD_FOLDER = "uploaded_docs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@router.post("/upload/upload_file")
async def upload_file(file: UploadFile = File(...), chat_id: str = Form(...)):
    try:
        file_path = os.path.join(UPLOAD_FOLDER, f"{chat_id}_{file.filename}")
        with open(file_path, "wb") as f_out:
            content = await file.read()
            f_out.write(content)

        return {
            "message": f"File '{file.filename}' uploaded for chat_id '{chat_id}'",
            "status": "success",
            "filename": file.filename,
            "chat_id": chat_id,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")
