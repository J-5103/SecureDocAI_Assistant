import json, os
from nanoid import generate
from .config import settings

def _load():
    os.makedirs(os.path.dirname(settings.DATA_FILE), exist_ok=True)
    if not os.path.exists(settings.DATA_FILE):
        return {"chats": []}
    with open(settings.DATA_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def _save(data):
    os.makedirs(os.path.dirname(settings.DATA_FILE), exist_ok=True)
    with open(settings.DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def list_chats():
    d = _load()
    items = [{"id": c["id"], "title": c.get("title","Untitled"), "createdAt": c["createdAt"]} for c in d["chats"]]
    items.sort(key=lambda x: x["createdAt"], reverse=True)
    return items

def get_chat(chat_id):
    d = _load()
    for c in d["chats"]:
        if c["id"] == chat_id:
            return c
    return None

def create_chat(seed_prompt=None, seed_greeting=None, greeting_text=None):
    import datetime as dt
    d = _load()
    cid = generate()
    chat = {"id": cid, "title": (seed_prompt or "New quick chat")[:64],
            "createdAt": dt.datetime.utcnow().isoformat()+"Z",
            "messages": []}
    if seed_greeting:
        chat["messages"].append({
            "id": generate(),
            "sender": "ai",
            "text": greeting_text or "Hello! I'm ready to help. What would you like to know?",
            "at": dt.datetime.utcnow().isoformat()+"Z",
        })
    if seed_prompt:
        chat["messages"].append({
            "id": generate(), "sender": "user", "text": seed_prompt,
            "at": dt.datetime.utcnow().isoformat()+"Z"
        })
    d["chats"].append(chat)
    _save(d)
    return cid

def append_message(chat_id, sender, text):
    import datetime as dt
    d = _load()
    for c in d["chats"]:
        if c["id"] == chat_id:
            m = {"id": generate(), "sender": sender, "text": text,
                 "at": dt.datetime.utcnow().isoformat()+"Z"}
            c["messages"].append(m)
            _save(d)
            return m
    return None

def delete_chat(chat_id):
    d = _load()
    d["chats"] = [c for c in d["chats"] if c["id"] != chat_id]
    _save(d)
    return True
