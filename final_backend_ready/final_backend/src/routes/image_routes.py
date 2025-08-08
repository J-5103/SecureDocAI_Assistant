from fastapi import APIRouter, UploadFile, File
from src.pipeline.image_question_pipeline import ImageQuestionPipeline

router = APIRouter()

@router.post("/ask-image")
async def ask_image(
    front_image: UploadFile = File(...),
    back_image: UploadFile = File(None)
):
    try:
        front_bytes = await front_image.read()
        back_bytes = await back_image.read() if back_image else None

        pipeline = ImageQuestionPipeline()
        result = pipeline.run(front_bytes, back_bytes)

        return {"status": "success", "data": result}
    except Exception as e:
        return {"status": "error", "message": str(e)}
