from fastapi import APIRouter , UploadFile , File , HTTPException
from src.pipeline.document_pipeline import DocumentPipeline

router = APIRouter()
document_pipeline = DocumentPipeline()

@router.post("/upload/")
async def upload_file(file :UploadFile = File(...)):
    try:
        content = await file.read()
        result= document_pipeline.process(file.filename,content)
        return {"message": "File processed and saved.", "status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500,detail=str(e))