# src/routes/chat_routes.py
from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Literal, Optional

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from uuid import uuid4

# Optional: get image dimensions if Pillow is available
try:
    from PIL import Image  # type: ignore
except Exception:  # pragma: no cover
    Image = None  # type: ignore

# Import your DB helper
from src.components.db_handler import DBHandler

router = APIRouter(prefix="/chat", tags=["chat"])

# ---------- Config / paths ----------
UPLOADS_ROOT = Path(os.environ.get("UPLOADS_DIR", "uploads")).resolve()
CHAT_UPLOADS_DIR = UPLOADS_ROOT / "chat"
CHAT_UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

MANIFESTS_DIR = CHAT_UPLOADS_DIR / "_manifests"
MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)

# Allowed (match frontend api.js)
IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tiff", ".tif"}
VISION_EXTS = IMG_EXTS | {".pdf"}


def _ext_of(name: str) -> str:
    i = name.rfind(".")
    return name[i:].lower() if i >= 0 else ""


def _ensure_allowed(filename: str):
    ext = _ext_of(filename or "")
    if ext not in VISION_EXTS:
        raise HTTPException(
            400,
            f"Only image/PDF files are allowed "
            f"({', '.join(sorted(VISION_EXTS))}). Got: {filename}",
        )


def _safe_name(name: str) -> str:
    return re.sub(r"[^\w.\-]+", "_", (name or "upload.bin")).strip("._") or "upload.bin"


def _chat_dir(chat_id: str) -> Path:
    safe = re.sub(r"[^a-zA-Z0-9_\-]", "_", chat_id or "default")
    p = CHAT_UPLOADS_DIR / safe
    p.mkdir(parents=True, exist_ok=True)
    return p


def _manifest_path(chat_id: str) -> Path:
    safe = re.sub(r"[^a-zA-Z0-9_\-]", "_", chat_id or "default")
    return MANIFESTS_DIR / f"{safe}.json"


def _save_file(chat_id: str, up: UploadFile) -> dict:
    _ensure_allowed(up.filename or "")
    base_dir = _chat_dir(chat_id)
    unique = f"{uuid4().hex[:8]}-{_safe_name(up.filename or 'upload')}"
    abs_path = base_dir / unique

    # Save file
    with abs_path.open("wb") as f:
        f.write(up.file.read())

    # Try to get dims for images
    width = height = None
    if Image is not None and _ext_of(unique) in IMG_EXTS:
        try:
            with Image.open(abs_path) as im:
                width, height = im.size
        except Exception:
            pass

    rel_url = f"/uploads/chat/{_safe_name(chat_id)}/{unique}"

    return {
        "name": up.filename or unique,
        "size": abs_path.stat().st_size,
        "content_type": up.content_type or "",
        "url": rel_url,
        "width": width,
        "height": height,
        "saved_path": str(abs_path),
    }


def _append_manifest(chat_id: str, entries: List[dict]):
    mp = _manifest_path(chat_id)
    try:
        cur = json.loads(mp.read_text("utf-8"))
    except Exception:
        cur = {"messages": []}

    # We store each upload as one "message" with its attachments
    cur["messages"].append(
        {
            "id": uuid4().hex,
            "sender": "user",
            "text": "(image attached)",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "attachments": [
                {k: v for k, v in e.items() if k != "saved_path"} for e in entries
            ],
        }
    )
    mp.write_text(json.dumps(cur, ensure_ascii=False, indent=2), encoding="utf-8")


# ---------- Pydantic models (existing) ----------
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


# ---------- New: upload endpoint to match frontend chatUploadImages() ----------

@router.post("")
async def upload_to_chat(
    chat_id: str = Form(...),
    text: str = Form(default=""),
    files: List[UploadFile] = File(default=None),
):
    """
    Accepts multiple files, saves under /uploads/chat/{chat_id}/, and returns:
    {
      status: "ok",
      chat_id,
      text,
      attachments: [{name, url, size, content_type, width?, height?}, ...],
      image_urls: ["..."]
    }
    Also appends a simple manifest row and (optionally) upserts a DB message.
    """
    try:
        saved: List[dict] = []
        for up in files or []:
            saved.append(_save_file(chat_id, up))

        # Persist lightweight manifest for /chat/history
        if saved:
            _append_manifest(chat_id, saved)

        # Optional: upsert a single message in DB with the normalized URLs (matches your existing schema)
        try:
            if saved:
                db = DBHandler()
                db.upsert_messages(
                    chat_id=chat_id,
                    source="chat",
                    messages=[
                        {
                            "id": str(uuid4()),
                            "sender": "user",
                            "text": (text or "(image attached)").strip(),
                            "image_urls": [e["url"] for e in saved],
                            "timestamp": datetime.now(timezone.utc),
                        }
                    ],
                )
        except Exception:
            # DB is optional for this path; ignore failures
            pass

        # Shape expected by frontend api.js (it re-normalizes urls to absolute)
        attachments = [
            {k: v for k, v in e.items() if k != "saved_path"} for e in saved
        ]
        image_urls = [e["url"] for e in saved]

        return JSONResponse(
            {
                "status": "ok",
                "chat_id": chat_id,
                "text": text,
                "attachments": attachments,
                "image_urls": image_urls,
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {e}")


@router.get("/history")
async def chat_history(chat_id: str = Query(..., description="Chat id to list uploads for")):
    """
    Return a simple history view reconstructed from the manifest created by /chat uploads.
    The frontend normalizer maps each message's attachments and absolute-izes the urls.
    """
    mp = _manifest_path(chat_id)
    if not mp.exists():
        return {"chat_id": chat_id, "messages": []}

    try:
        data = json.loads(mp.read_text("utf-8"))
        messages = data.get("messages", [])
    except Exception:
        messages = []

    return {"chat_id": chat_id, "messages": messages}


# ---------- Existing routes (kept) ----------

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
