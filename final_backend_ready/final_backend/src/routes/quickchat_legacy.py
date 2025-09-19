# src/routes/quickchat_legacy.py
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

# Flexible import (run as 'src.main:app' or from inside 'src/')
try:
    from src.routes.quick_chats import (
        CHATS,
        MESSAGES,
        create_quick_chat as create_new,
        list_quick_chats as list_new,
        get_messages as get_messages_new,
        send_message as send_message_new,
        delete_chat as delete_chat_new,
        get_quick_chat as get_quick_chat_new,
        rename_quick_chat as rename_quick_chat_new,
        ChatCreate,
        SendMessage,
        ChatPatch,
    )
except Exception:
    from .quick_chats import (
        CHATS,
        MESSAGES,
        create_quick_chat as create_new,
        list_quick_chats as list_new,
        get_messages as get_messages_new,
        send_message as send_message_new,
        delete_chat as delete_chat_new,
        get_quick_chat as get_quick_chat_new,
        rename_quick_chat as rename_quick_chat_new,
        ChatCreate,
        SendMessage,
        ChatPatch,
    )

router = APIRouter(prefix="/api/quickchat", tags=["quick-chats-legacy"])

# ----- Request bodies (legacy) -----
class CreateBody(BaseModel):
    seedPrompt: str | None = None
    seedGreeting: bool | None = True
    greetingText: str | None = None

class SendBody(BaseModel):
    text: str

class RenameBody(BaseModel):
    title: str


# ----- Legacy routes mapping to new handlers -----

@router.get("/list")
def list_legacy():
    """
    Legacy: GET /api/quickchat/list
    Mirrors GET /api/quick-chats
    """
    # mirror plural list and include chatId alias for each item
    items = sorted(CHATS.values(), key=lambda c: c["createdAt"], reverse=True)
    enriched = [{**it, "chatId": it["id"]} for it in items]
    return {"items": enriched}

@router.get("/{chat_id}")
def messages_legacy(chat_id: str):
    """
    Legacy: GET /api/quickchat/{id}
    Mirrors GET /api/quick-chats/{id}/messages
    """
    if chat_id not in CHATS:
        raise HTTPException(status_code=404, detail="Chat not found")
    return {
        "chatId": chat_id,
        "title": CHATS[chat_id]["title"],
        "createdAt": CHATS[chat_id]["createdAt"],
        "items": MESSAGES.get(chat_id, []),
    }

@router.get("/{chat_id}/meta")
def meta_legacy(chat_id: str):
    """
    Legacy: GET /api/quickchat/{id}/meta
    Mirrors GET /api/quick-chats/{id}
    """
    # Ensure chatId alias is present
    resp = get_quick_chat_new(chat_id)
    if isinstance(resp, dict) and "chatId" not in resp and "id" in resp:
        resp = {**resp, "chatId": resp["id"]}
    return resp

@router.post("", status_code=201)
def create_legacy(body: CreateBody):
    """
    Legacy: POST /api/quickchat
    Mirrors POST /api/quick-chats
    """
    res = create_new(ChatCreate(**body.model_dump()))
    # normalize to include chatId
    if isinstance(res, dict):
        cid = res.get("id") or res.get("chatId")
        return {**res, "chatId": cid}
    return {"id": res, "chatId": res}

@router.post("/{chat_id}/message")
def send_legacy(chat_id: str, body: SendBody):
    """
    Legacy: POST /api/quickchat/{id}/message
    Mirrors POST /api/quick-chats/{id}/messages
    """
    res = send_message_new(chat_id, SendMessage(text=body.text))
    if isinstance(res, dict):
        return {**res, "chatId": chat_id}
    # if new handler returned a plain string, wrap it
    return {"chatId": chat_id, "reply": str(res)}

@router.patch("/{chat_id}")
def rename_legacy(chat_id: str, body: RenameBody):
    """
    Legacy: PATCH /api/quickchat/{id}
    Mirrors PATCH /api/quick-chats/{id}
    """
    res = rename_quick_chat_new(chat_id, ChatPatch(title=body.title))
    if isinstance(res, dict):
        # ensure chatId + title present
        return {"ok": bool(res.get("ok", True)), "chatId": chat_id, "title": res.get("title", body.title)}
    return {"ok": True, "chatId": chat_id, "title": body.title}

@router.delete("/{chat_id}")
def delete_legacy(chat_id: str):
    """
    Legacy: DELETE /api/quickchat/{id}
    Mirrors DELETE /api/quick-chats/{id}
    """
    res = delete_chat_new(chat_id)
    if isinstance(res, dict):
        return {**res, "chatId": chat_id}
    return {"ok": True, "chatId": chat_id}
