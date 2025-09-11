# routers/cards.py
# FastAPI router for business-card extraction + listing + export
# --------------------------------------------------------------
# main.py:
#   from routers import cards
#   app.include_router(cards.router)        # /api/cards/*
#   app.include_router(cards.chats_router)  # /api/chats/*
#   app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")
#
# Optional deps:
#   pip install pillow pytesseract pdfplumber openpyxl

from __future__ import annotations

import csv
import io
import json
import os
import re
import uuid
import zipfile
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

from fastapi import (
    APIRouter,
    File,
    Form,
    HTTPException,
    Query,
    UploadFile,
    Request,
)
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

# ---- Optional imports (best-effort) ----
try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None  # type: ignore

try:
    import pytesseract  # type: ignore
except Exception:  # pragma: no cover
    pytesseract = None  # type: ignore

try:
    import pdfplumber  # type: ignore
except Exception:  # pragma: no cover
    pdfplumber = None  # type: ignore

try:
    import openpyxl  # type: ignore
    from openpyxl.workbook import Workbook  # type: ignore
except Exception:  # pragma: no cover
    openpyxl = None  # type: ignore
    Workbook = None  # type: ignore


# =============================================================================
# Config / Paths
# =============================================================================

UPLOADS_ROOT = Path(os.environ.get("UPLOADS_DIR", "uploads")).resolve()
CARDS_DIR = UPLOADS_ROOT / "cards"
CARDS_DIR.mkdir(parents=True, exist_ok=True)

def chat_cards_path(chat_id: str) -> Path:
    safe = re.sub(r"[^a-zA-Z0-9_\-]", "_", chat_id or "default")
    return CARDS_DIR / f"{safe}.json"

IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tiff", ".tif"}
VISION_EXTS = IMG_EXTS | {".pdf"}


# =============================================================================
# Models
# =============================================================================

class Address(BaseModel):
    street: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    postal_code: Optional[str] = Field(default=None, alias="postalCode")
    country: Optional[str] = None

    class Config:
        populate_by_name = True

class CardItem(BaseModel):
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    organization: Optional[str] = None
    job_title: Optional[str] = None
    phones: List[str] = Field(default_factory=list)
    emails: List[str] = Field(default_factory=list)
    websites: List[str] = Field(default_factory=list)
    address: Optional[Address] = None
    source_url: Optional[str] = None  # relative path to uploaded image/pdf

class CardsResponse(BaseModel):
    items: List[CardItem] = Field(default_factory=list)

# Stored row shape (adds meta)
class CardRecord(CardItem):
    id: str
    chat_id: Optional[str] = None
    created_at: str
    raw_text: Optional[str] = None
    vcard_url: Optional[str] = None
    payload_json: Optional[dict] = None


# =============================================================================
# Routers
# =============================================================================

router = APIRouter(prefix="/api/cards", tags=["cards"])
chats_router = APIRouter(prefix="/api/chats", tags=["cards"])


# =============================================================================
# Helpers
# =============================================================================

def _ext_of(name: str) -> str:
    i = name.rfind(".")
    return name[i:].lower() if i >= 0 else ""

def _ensure_allowed(filename: str):
    ext = _ext_of(filename)
    if ext not in VISION_EXTS:
        raise HTTPException(
            400,
            f"Only image/PDF files are allowed "
            f"({', '.join(sorted(VISION_EXTS))}). Got: {filename}",
        )

def _save_upload(file: UploadFile) -> Tuple[str, Path]:
    """Save uploaded file under /uploads/cards/<uuid>-<name> and return (rel_url, abs_path)."""
    _ensure_allowed(file.filename or "")
    safe_name = re.sub(r"[^\w.\-]+", "_", file.filename or "upload.bin")
    unique = f"{uuid.uuid4().hex[:8]}-{safe_name}"
    abs_path = CARDS_DIR / unique
    with abs_path.open("wb") as f:
        f.write(file.file.read())
    rel_url = f"/uploads/cards/{unique}"
    return rel_url, abs_path

def _read_text_from_pdf(path: Path) -> str:
    if pdfplumber is None:
        return ""
    try:
        text_chunks = []
        with pdfplumber.open(str(path)) as pdf:
            for page in pdf.pages[:2]:
                t = page.extract_text() or ""
                if t.strip():
                    text_chunks.append(t)
        return "\n".join(text_chunks).strip()
    except Exception:
        return ""

def _ocr_image(path: Path) -> str:
    if pytesseract is None or Image is None:
        return ""
    try:
        img = Image.open(path)
        if img.mode not in ("L", "RGB"):
            img = img.convert("RGB")
        return pytesseract.image_to_string(img) or ""
    except Exception:
        return ""

EMAIL_RE = re.compile(r"[A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,}", re.I)
PHONE_RE = re.compile(r"(?:\+?\d{1,3}[\s\-]?)?(?:\(?\d{2,4}\)?[\s\-]?)?\d{3,4}[\s\-]?\d{3,4}")
URL_RE = re.compile(
    r"(?:https?://|www\.)[A-Z0-9._\-\/~%+#?=&]+|[A-Z0-9._\-]+\.[A-Z]{2,}(?:/[^\s]*)?",
    re.I,
)

def _first_two_nonempty_lines(text: str) -> Tuple[str, str]:
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    a = lines[0] if len(lines) > 0 else ""
    b = lines[1] if len(lines) > 1 else ""
    return a, b

def _guess_name_and_org(text: str) -> Tuple[str, str, str]:
    a, b = _first_two_nonempty_lines(text)
    def looks_like_name(s: str) -> bool:
        parts = [p for p in re.split(r"[\s,]+", s) if p]
        caps = sum(1 for p in parts if p[:1].isupper())
        return 1 <= len(parts) <= 4 and caps >= max(1, len(parts) - 1)
    first_name, last_name, org = "", "", ""
    if looks_like_name(a):
        parts = a.split()
        if parts:
            first_name = parts[0]
            last_name = " ".join(parts[1:]) if len(parts) > 1 else ""
        if b and (b.isupper() or b.istitle() or len(b) > 3):
            org = b
    else:
        if b and looks_like_name(b):
            parts = b.split()
            first_name = parts[0]
            last_name = " ".join(parts[1:]) if len(parts) > 1 else ""
        if a:
            org = a
    return first_name, last_name, org

def _dedup(seq: List[str]) -> List[str]:
    seen = set()
    out = []
    for s in seq:
        t = re.sub(r"\s+", " ", (s or "").strip())
        if not t:
            continue
        k = t.lower()
        if k not in seen:
            seen.add(k)
            out.append(t)
    return out

def _parse_text_to_card(text: str) -> CardItem:
    emails = _dedup(EMAIL_RE.findall(text or ""))
    phones = _dedup(PHONE_RE.findall(text or ""))
    webs = _dedup(URL_RE.findall(text or ""))

    fn, ln, org = _guess_name_and_org(text or "")
    title = ""
    for line in (text or "").splitlines():
        l = line.strip()
        if re.search(r"\b(owner|ceo|cto|coo|founder|director|manager|lead|head|vp|president)\b", l, re.I):
            title = l
            break

    return CardItem(
        first_name=fn or None,
        last_name=ln or None,
        organization=org or None,
        job_title=title or None,
        phones=phones,
        emails=emails,
        websites=webs,
        address=None,
    )

def _store_card(chat_id: Optional[str], rec: CardRecord) -> None:
    p = chat_cards_path(chat_id or "default")
    try:
        if p.exists():
            data = json.loads(p.read_text("utf-8"))
        else:
            data = {"items": []}
    except Exception:
        data = {"items": []}
    items = data.get("items", [])
    items.append(rec.model_dump(by_alias=True))
    data["items"] = items
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

def _make_vcard(item: CardItem) -> str:
    fn = f"{(item.first_name or '').strip()} {(item.last_name or '').strip()}".strip()
    n_line = f"N:{(item.last_name or '').strip()};{(item.first_name or '').strip()};;;"
    lines = ["BEGIN:VCARD", "VERSION:3.0", n_line, f"FN:{fn}"]
    if item.organization:
        lines.append(f"ORG:{item.organization}")
    if item.job_title:
        lines.append(f"TITLE:{item.job_title}")
    for ph in item.phones or []:
        lines.append(f"TEL;TYPE=WORK,VOICE:{ph}")
    for em in item.emails or []:
        lines.append(f"EMAIL;TYPE=INTERNET:{em}")
    for w in item.websites or []:
        lines.append(f"URL:{w}")
    if item.address:
        a = item.address
        adr = f";;{a.street or ''};{a.city or ''};{a.state or ''};{a.postal_code or ''};{a.country or ''}"
        lines.append(f"ADR;TYPE=WORK:{adr}")
    lines.append("END:VCARD")
    return "\n".join(lines)

def _write_vcard_file(item: CardItem, base_name: str) -> str:
    safe = re.sub(r"[^\w.\-]+", "_", base_name or "contact")
    vcf_name = f"{safe}.vcf"
    vcf_path = CARDS_DIR / vcf_name
    vcf_path.write_text(_make_vcard(item), encoding="utf-8")
    return f"/uploads/cards/{vcf_name}"

def _now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


# =============================================================================
# Endpoints
# =============================================================================

@router.post("/extract")
async def cards_extract(
    request: Request,
    file: UploadFile = File(...),
    prompt: str = Form(""),
    return_vcard: bool = Query(True),
    chat_id: Optional[str] = Form(default=None),
):
    """
    Extract contact details from an uploaded business-card image/PDF.
    Returns: { status, data:{ items:[CardItem] }, vcard_url? }
    Also persists a normalized record in per-chat JSON.
    """
    rel_url, abs_path = _save_upload(file)

    ext = _ext_of(file.filename or "")
    text = ""
    if ext == ".pdf":
        text = _read_text_from_pdf(abs_path)
    else:
        text = _ocr_image(abs_path) if pytesseract and Image else ""

    parsed = _parse_text_to_card(text) if text else CardItem()
    parsed.source_url = rel_url

    rec = CardRecord(
        **parsed.model_dump(),
        id=uuid.uuid4().hex,
        chat_id=chat_id,
        created_at=_now_iso(),
        raw_text=text or None,
        vcard_url=None,
        payload_json={"prompt": prompt} if prompt else None,
    )

    vcard_url = None
    if return_vcard:
        base_name = (
            f"{parsed.first_name or ''}-{parsed.last_name or ''}".strip("-_")
            or Path(file.filename or "contact").stem
        )
        vcard_url = _write_vcard_file(parsed, base_name)
        rec.vcard_url = vcard_url

    try:
        _store_card(chat_id or "default", rec)
    except Exception as e:
        print("[cards] persist failed:", e)

    return JSONResponse(
        {
            "status": "ok",
            "data": {"items": [parsed.model_dump(by_alias=True)]},
            **({"vcard_url": vcard_url} if vcard_url else {}),
        }
    )

@router.post("/from-json")
async def cards_from_json(card: CardItem):
    """Build a .vcf file from a provided JSON card object."""
    base = f"{(card.first_name or '').strip()}_{(card.last_name or '').strip()}".strip("_") or "contact"
    vcard_url = _write_vcard_file(card, base)
    return {"vcard_url": vcard_url}

# ---- Back-compat list endpoint used by some clients -------------------------
@router.get("/list")
async def cards_list(chat_id: str = Query(...)):
    """Return cards for a chat via /api/cards/list?chat_id=... (compat shim)."""
    p = chat_cards_path(chat_id)
    if not p.exists():
        return {"items": []}
    try:
        data = json.loads(p.read_text("utf-8"))
        items = data.get("items", [])
    except Exception:
        items = []
    # Normalize to CardItem shape
    out = []
    for it in items:
        out.append({
            "first_name": it.get("first_name"),
            "last_name": it.get("last_name"),
            "organization": it.get("organization"),
            "job_title": it.get("job_title"),
            "phones": it.get("phones") or [],
            "emails": it.get("emails") or [],
            "websites": it.get("websites") or [],
            "address": it.get("address"),
            "source_url": it.get("source_url"),
        })
    return {"items": out}

# ---- Chat-scoped listing ----------------------------------------------------

@chats_router.get("/{chat_id}/cards")
async def chat_cards_list(chat_id: str):
    p = chat_cards_path(chat_id)
    if not p.exists():
        return {"items": []}
    try:
        data = json.loads(p.read_text("utf-8"))
        items = data.get("items", [])
    except Exception:
        items = []
    normalized = []
    for it in items:
        normalized.append({
            "first_name": it.get("first_name"),
            "last_name": it.get("last_name"),
            "organization": it.get("organization"),
            "job_title": it.get("job_title"),
            "phones": it.get("phones") or [],
            "emails": it.get("emails") or [],
            "websites": it.get("websites") or [],
            "address": it.get("address"),
            "source_url": it.get("source_url"),
        })
    return {"items": normalized}

# ---- Export helpers ---------------------------------------------------------

def _flatten_for_rows(item: CardItem) -> dict:
    addr = item.address or Address()
    return {
        "first_name": item.first_name or "",
        "last_name": item.last_name or "",
        "organization": item.organization or "",
        "job_title": item.job_title or "",
        "phones": ", ".join(item.phones or []),
        "emails": ", ".join(item.emails or []),
        "websites": ", ".join(item.websites or []),
        "street": addr.street or "",
        "city": addr.city or "",
        "state": addr.state or "",
        "postal_code": (addr.postal_code or ""),
        "country": addr.country or "",
    }

def _load_cards_raw(chat_id: str) -> List[dict]:
    p = chat_cards_path(chat_id)
    if not p.exists():
        return []
    try:
        return json.loads(p.read_text("utf-8")).get("items", [])
    except Exception:
        return []

def _load_cards_filtered(chat_id: str, include_ids: Optional[set[str]] = None, indices: Optional[set[int]] = None) -> List[CardItem]:
    """
    Load CardItem list optionally filtered by record 'id' or 1-based index.
    """
    raw = _load_cards_raw(chat_id)
    items: List[CardItem] = []
    for idx, it in enumerate(raw, start=1):
        if include_ids and it.get("id") not in include_ids:
            continue
        if indices and idx not in indices:
            continue
        # normalize into CardItem
        try:
            items.append(CardItem.model_validate({k: it.get(k) for k in CardItem.model_fields.keys()}))
        except Exception:
            pass
    return items

def _make_csv_bytes(items: List[CardItem]) -> bytes:
    buf = io.StringIO()
    writer = csv.DictWriter(
        buf,
        fieldnames=[
            "first_name","last_name","organization","job_title",
            "phones","emails","websites",
            "street","city","state","postal_code","country",
        ],
    )
    writer.writeheader()
    for it in items:
        writer.writerow(_flatten_for_rows(it))
    return buf.getvalue().encode("utf-8")

def _make_xlsx_bytes(items: List[CardItem]) -> bytes:
    if openpyxl is None or Workbook is None:
        return _make_csv_bytes(items)
    wb = Workbook()
    ws = wb.active
    ws.title = "Contacts"
    headers = [
        "first_name","last_name","organization","job_title",
        "phones","emails","websites","street","city","state","postal_code","country",
    ]
    ws.append(headers)
    for it in items:
        row = _flatten_for_rows(it)
        ws.append([row[h] for h in headers])
    out = io.BytesIO()
    wb.save(out)
    out.seek(0)
    return out.read()

def _make_vcf_bytes(items: List[CardItem]) -> bytes:
    vcards = [_make_vcard(it) for it in items]
    return ("\n".join(vcards)).encode("utf-8")

# ---- Export (whole chat or a single card) -----------------------------------

@chats_router.get("/{chat_id}/cards/export")
async def chat_cards_export(
    chat_id: str,
    format: str = Query("xlsx", regex="^(csv|xlsx|vcf|zip)$"),
    # NEW: filter to a single/selected card(s)
    card_id: Optional[str] = Query(default=None, description="Export only this card id"),
    id: Optional[List[str]] = Query(default=None, description="Repeatable query param: ?id=...&id=..."),
    index: Optional[int] = Query(default=None, ge=1, description="1-based index of the card to export"),
):
    """
    Export the chat's extracted cards as CSV / XLSX / VCF / ZIP.
    By default exports **all** cards in the chat.
    You can export a specific card using either:
      - ?card_id=<id>
      - ?id=<id>&id=<id2> (repeatable)
      - ?index=1         (1-based)
    """

    include_ids: Optional[set[str]] = None
    indices: Optional[set[int]] = None

    ids_from_params: List[str] = []
    if card_id:
        ids_from_params.append(card_id)
    if id:
        ids_from_params.extend([x for x in id if x])

    if ids_from_params:
        include_ids = set(ids_from_params)
    if index is not None:
        indices = {index}

    # load (optionally filtered)
    items = _load_cards_filtered(chat_id, include_ids=include_ids, indices=indices)

    safe_chat = re.sub(r"[^a-zA-Z0-9_\-]", "_", chat_id)
    dt = datetime.utcnow().strftime("%Y%m%d-%H%M%S")

    # Graceful empty export
    if not items:
        items = []  # produce an empty file with headers

    if format == "csv":
        data = _make_csv_bytes(items)
        fname = f"business-cards-{safe_chat}-{dt}.csv"
        return StreamingResponse(
            io.BytesIO(data),
            media_type="text/csv; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="{fname}"'},
        )

    if format == "xlsx":
        data = _make_xlsx_bytes(items)
        if openpyxl is None:
            fname = f"business-cards-{safe_chat}-{dt}.csv"
            media = "text/csv; charset=utf-8"
        else:
            fname = f"business-cards-{safe_chat}-{dt}.xlsx"
            media = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        return StreamingResponse(
            io.BytesIO(data),
            media_type=media,
            headers={"Content-Disposition": f'attachment; filename="{fname}"'},
        )

    if format == "vcf":
        data = _make_vcf_bytes(items)
        fname = f"business-cards-{safe_chat}-{dt}.vcf"
        return StreamingResponse(
            io.BytesIO(data),
            media_type="text/vcard; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="{fname}"'},
        )

    # format == "zip"
    mem = io.BytesIO()
    with zipfile.ZipFile(mem, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("business-cards.csv", _make_csv_bytes(items))
        for idx, it in enumerate(items, 1):
            base = (it.first_name or "") + "-" + (it.last_name or "")
            base = re.sub(r"[^\w.\-]+", "_", base).strip("_") or f"contact-{idx}"
            zf.writestr(f"{base}.vcf", _make_vcard(it))
    mem.seek(0)
    fname = f"business-cards-{safe_chat}-{dt}.zip"
    return StreamingResponse(
        mem,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{fname}"'},
    )
