from fastapi import APIRouter

router = APIRouter()

@router.post("/create_chat")
async def create_chat():
    return {"message": "Chat created"}
