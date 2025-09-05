# src/routes/chat_routes.py
from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Literal, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from uuid import uuid4

# Import your DB helper
from src.components.db_handler import DBHandler

router = APIRouter(prefix="/chat", tags=["chat"])

# ---------- Pydantic models ----------

SourceType = Literal["chat", "viz"]


class CreateChatReq(BaseModel):
    name: Optional[str] = Field(default=None, description="Human-friendly chat name")
    source: SourceType = Field(default="chat", description="Bucket: chat or viz")


class CreateChatRes(BaseModel):
    message: str
    chat_id: str
    source: SourceType


class MessageIn(BaseModel):
    chat_id: str = Field(..., description="Existing chat id")
    sender: Literal["user", "ai"]
    text: Optional[str] = None
    imageUrls: Optional[List[str]] = Field(default=None, description="Attachment URLs")
    timestamp: Optional[datetime] = None
    source: SourceType = Field(default="chat")


class BulkMessagesIn(BaseModel):
    chat_id: str
    source: SourceType = Field(default="chat")
    messages: List[MessageIn]


# ---------- Routes ----------

@router.post("/create", response_model=CreateChatRes)
async def create_chat(payload: CreateChatReq | None = None):
    """
    Create a new chat session and persist it in Postgres.
    """
    data = payload or CreateChatReq()
    chat_id = str(uuid4())

    try:
        db = DBHandler()
        # Persist session (UPSERT inside DBHandler)
        db.upsert_chat_session(
            chat_id=chat_id,
            source=data.source,
            name=data.name or f"{data.source.capitalize()} Chat {datetime.now().isoformat(timespec='seconds')}",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create chat: {e}")

    return CreateChatRes(message="Chat created successfully", chat_id=chat_id, source=data.source)


@router.post("/message")
async def post_message(msg: MessageIn):
    """
    Store a single message (user/ai) for a chat.
    Accepts text and/or imageUrls. Timestamp defaults to now().
    """
    try:
        ts = msg.timestamp or datetime.now(timezone.utc)
        db = DBHandler()
        db.upsert_messages(
            chat_id=msg.chat_id,
            source=msg.source,
            messages=[
                {
                    "id": str(uuid4()),
                    "sender": msg.sender,
                    "text": (msg.text or "").strip(),
                    "image_urls": msg.imageUrls or [],
                    "timestamp": ts,
                }
            ],
        )
        return {"ok": True, "chat_id": msg.chat_id, "stored": 1}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save message: {e}")


@router.post("/messages/bulk")
async def post_messages_bulk(payload: BulkMessagesIn):
    """
    Store multiple messages in one call.
    """
    try:
        now = datetime.now(timezone.utc)
        msgs = []
        for m in payload.messages:
            ts = m.timestamp or now
            msgs.append(
                {
                    "id": str(uuid4()),
                    "sender": m.sender,
                    "text": (m.text or "").strip(),
                    "image_urls": m.imageUrls or [],
                    "timestamp": ts,
                }
            )

        db = DBHandler()
        db.upsert_messages(chat_id=payload.chat_id, source=payload.source, messages=msgs)
        return {"ok": True, "chat_id": payload.chat_id, "stored": len(msgs)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save messages: {e}")


# --- Optional convenience reads (use only if your DBHandler exposes getters) ---

@router.get("/{chat_id}")
async def get_chat(chat_id: str):
    """
    Return chat session metadata if your DBHandler implements a getter.
    """
    try:
        db = DBHandler()
        if hasattr(db, "get_chat"):
            return db.get_chat(chat_id)
        raise HTTPException(status_code=501, detail="Get chat not implemented in DBHandler")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch chat: {e}")


@router.get("/{chat_id}/messages")
async def get_chat_messages(chat_id: str, source: SourceType = "chat"):
    """
    Return stored messages for a chat if your DBHandler implements a getter.
    """
    try:
        db = DBHandler()
        if hasattr(db, "get_messages"):
            return {"chat_id": chat_id, "source": source, "messages": db.get_messages(chat_id, source)}
        raise HTTPException(status_code=501, detail="Get messages not implemented in DBHandler")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch messages: {e}")
