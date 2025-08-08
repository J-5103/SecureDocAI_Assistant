from fastapi import APIRouter
from uuid import uuid4

router = APIRouter(prefix="/chat")

@router.post("/create")
async def create_chat():
    chat_id = str(uuid4())
    return {
        "message": "Chat created successfully",
        "chat_id": chat_id
    }
