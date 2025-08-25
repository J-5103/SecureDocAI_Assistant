from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any
import uuid

from src.pipeline.image_question_pipeline import ImageQuestionPipeline

router = APIRouter(prefix="/api", tags=["vision"])
pipeline = ImageQuestionPipeline()

# simple in-memory chat store
CHAT: Dict[str, list] = {}

class AskImageOut(BaseModel):
    status: str
    data: Dict[str, Any]
    session_id: str

@router.post("/ask-image", response_model=AskImageOut)
async def ask_image(
    front_image: UploadFile = File(...),
    back_image: Optional[UploadFile] = File(None),
    session_id: Optional[str] = Form(None)
):
    try:
        fb = await front_image.read()
        bb = await back_image.read() if back_image else None
        out = pipeline.run(fb, bb)
    except Exception as e:
        raise HTTPException(500, f"Vision extraction failed: {e}")

    sid = session_id or str(uuid.uuid4())
    CHAT.setdefault(sid, []).append({
        "role": "user",
        "type": "image",
        "front_name": front_image.filename,
        "back_name": back_image.filename if back_image else None
    })
    CHAT[sid].append({
        "role": "assistant",
        "whatsapp": out["whatsapp"],
        "vcard": out["vcard"],
        "json": out["json"]
    })

    return AskImageOut(status="success", data=out, session_id=sid)

@router.get("/chat/{session_id}")
async def get_chat(session_id: str):
    return {"session_id": session_id, "history": CHAT.get(session_id, [])}
