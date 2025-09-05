# backend/routes/image_routes.py
from __future__ import annotations

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from typing import List, Optional, Dict, Any
from pathlib import Path
import mimetypes

# --- (best-effort) load .env so GEMINI_API_KEY is available ---
try:
    from dotenv import load_dotenv  # type: ignore
    here = Path(__file__).resolve()
    for p in [
        Path.cwd() / ".env",
        here.with_name(".env"),
        here.parent / ".env",
        here.parent.parent / ".env",
        here.parent.parent / "src" / ".env",
    ]:
        if p.exists():
            load_dotenv(p, override=False)
except Exception:
    pass

# --- import Gemini extractor (your shared module) ---
try:
    from src.pipeline.card_extractor import extract_business_card, BusinessCard  # type: ignore
except Exception:
    from pipeline.card_extractor import extract_business_card, BusinessCard  # type: ignore

# DB
try:
    from src.db_handler import DBHandler  # type: ignore
except Exception:
    DBHandler = None  # type: ignore

router = APIRouter(tags=["image"])

# ---------- helpers ----------
def _looks_image(name: str, ctype: Optional[str]) -> bool:
    n = (name or "").lower()
    ct = (ctype or "").lower()
    return ct.startswith("image/") or n.endswith(
        (".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tiff", ".tif")
    )

def _looks_pdf(name: str, ctype: Optional[str]) -> bool:
    n = (name or "").lower()
    ct = (ctype or "").lower()
    return n.endswith(".pdf") or ct in ("application/pdf", "application/x-pdf")

def _guess_mime(name: str, ctype: Optional[str]) -> str:
    return ctype or mimetypes.guess_type(name or "")[0] or "image/jpeg"

def _merge_lists(a: Optional[List[str]], b: Optional[List[str]]) -> List[str]:
    out: List[str] = []
    for v in (a or []):
        if v and v not in out:
            out.append(v)
    for v in (b or []):
        if v and v not in out:
            out.append(v)
    return out

def _format_addr(addr: Optional[Dict[str, Any]]) -> str:
    if not addr:
        return ""
    parts = [
        addr.get("street"),
        addr.get("city"),
        addr.get("state"),
        addr.get("postal_code"),
        addr.get("country"),
    ]
    return ", ".join([p for p in parts if p])

def _format_card_markdown(card: Dict[str, Any]) -> str:
    """Simple, clean markdown summary."""
    lines: List[str] = []
    # names (support multiple)
    names = card.get("names") or []
    full_name = card.get("full_name")
    if names:
        for nm in names:
            if nm:
                lines.append(f"**Name:** {nm}")
    elif full_name:
        lines.append(f"**Name:** {full_name}")

    # titles/companies (support multiple)
    for t in card.get("titles") or ([] if not card.get("job_title") else [card["job_title"]]):
        if t:
            lines.append(f"**Title:** {t}")
    orgs = card.get("organizations") or []
    if card.get("organization"):
        orgs = [card["organization"], *orgs]
    for c in orgs:
        if c:
            lines.append(f"**Company:** {c}")

    for p in card.get("phones") or []:
        lines.append(f"**Phone:** {p}")
    for e in card.get("emails") or []:
        lines.append(f"**Email:** {e}")
    for u in card.get("websites") or []:
        lines.append(f"**Website:** {u}")

    adr = _format_addr(card.get("address"))
    if adr:
        lines.append(f"**Address:** {adr}")

    social = card.get("social") or {}
    for k, v in social.items():
        if v:
            k2 = k[:1].upper() + k[1:]
            lines.append(f"**{k2}:** {v}")

    return "\n".join(lines).strip() or "No text detected."

def _vcard_of(card: Dict[str, Any]) -> str:
    """Very small vCard 3.0 generator (best-effort)."""
    def _safe(v: Optional[str]) -> str:
        return (v or "").replace("\n", " ").replace(";", " ").strip()

    names = card.get("names") or ([card.get("full_name")] if card.get("full_name") else [])
    org = None
    if card.get("organizations"):
        org = card["organizations"][0]
    elif card.get("organization"):
        org = card["organization"]

    lines = ["BEGIN:VCARD", "VERSION:3.0"]
    if names:
        lines.append(f"FN:{_safe(names[0])}")
    if org:
        lines.append(f"ORG:{_safe(org)}")
    for t in card.get("titles") or ([] if not card.get("job_title") else [card["job_title"]]):
        if t:
            lines.append(f"TITLE:{_safe(t)}")
    for p in card.get("phones") or []:
        lines.append(f"TEL;TYPE=CELL:{_safe(p)}")
    for e in card.get("emails") or []:
        lines.append(f"EMAIL;TYPE=INTERNET:{_safe(e)}")
    for u in card.get("websites") or []:
        lines.append(f"URL:{_safe(u)}")
    adr = card.get("address") or {}
    if adr:
        parts = [
            "",  # PO box
            "",  # extended
            _safe(adr.get("street")),
            _safe(adr.get("city")),
            _safe(adr.get("state")),
            _safe(adr.get("postal_code")),
            _safe(adr.get("country")),
        ]
        lines.append("ADR;TYPE=WORK:" + ";".join(parts))
    lines.append("END:VCARD")
    return "\n".join(lines)

def _pick_longer(a: Optional[str], b: Optional[str]) -> Optional[str]:
    a = a or ""
    b = b or ""
    return a if len(a) >= len(b) else b

def _merge_cards(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    """Merge two sides of same person (keep arrays, de-dup)."""
    if not a:
        return b or {}
    if not b:
        return a or {}
    addr_a = a.get("address") or {}
    addr_b = b.get("address") or {}

    merged = {
        "full_name": _pick_longer(a.get("full_name"), b.get("full_name")),
        "first_name": a.get("first_name") or b.get("first_name") or None,
        "last_name": a.get("last_name") or b.get("last_name") or None,
        "organization": _pick_longer(a.get("organization"), b.get("organization")),
        "job_title": _pick_longer(a.get("job_title"), b.get("job_title")),
        "phones": _merge_lists(a.get("phones"), b.get("phones")),
        "emails": _merge_lists(a.get("emails"), b.get("emails")),
        "websites": _merge_lists(a.get("websites"), b.get("websites")),
        "address": None,
        "social": {**(a.get("social") or {}), **(b.get("social") or {})},
        "notes": a.get("notes") or b.get("notes") or None,
        "raw_text": "\n".join(
            [x for x in [(a.get("raw_text") or "").strip(), (b.get("raw_text") or "").strip()] if x]
        ).strip() or None,
        # carry multi fields if extractor provides them
        "names": _merge_lists(a.get("names"), b.get("names")),
        "titles": _merge_lists(a.get("titles"), b.get("titles")),
        "organizations": _merge_lists(a.get("organizations"), b.get("organizations")),
        "companies": _merge_lists(a.get("companies"), b.get("companies")),
    }
    if addr_a or addr_b:
        merged["address"] = {
            "street": addr_a.get("street") or addr_b.get("street") or None,
            "city": addr_a.get("city") or addr_b.get("city") or None,
            "state": addr_a.get("state") or addr_b.get("state") or None,
            "postal_code": addr_a.get("postal_code") or addr_b.get("postal_code") or None,
            "country": addr_a.get("country") or addr_b.get("country") or None,
        }
    return merged

def _norm(s: str) -> str:
    return (s or "").lower().replace("&", "and").strip()

def _primary_name(card: Dict[str, Any]) -> str:
    names = card.get("names") or []
    if names:
        return _norm(names[0])
    return _norm(card.get("full_name") or "")

def _intersect(a: List[str], b: List[str]) -> bool:
    sa = {str(x).lower() for x in (a or [])}
    sb = {str(x).lower() for x in (b or [])}
    return len(sa.intersection(sb)) > 0

def _same_person(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    # shared email or phone => same
    if _intersect(a.get("emails") or [], b.get("emails") or []):
        return True
    if _intersect(a.get("phones") or [], b.get("phones") or []):
        return True
    # or identical normalized name
    return _primary_name(a) and _primary_name(a) == _primary_name(b)

def _cluster_and_merge(cards: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Merge cards that look like the same person; keep different people separate.
    """
    clusters: List[Dict[str, Any]] = []
    for c in cards:
        placed = False
        for i in range(len(clusters)):
            if _same_person(clusters[i], c):
                clusters[i] = _merge_cards(clusters[i], c)
                placed = True
                break
        if not placed:
            clusters.append(c)
    return clusters

def _split_single_card_if_multi(card: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
    """
    If one card has multiple names (and possibly parallel phones/emails), split into many contacts (best-effort).
    """
    names = card.get("names") or ([card.get("full_name")] if card.get("full_name") else [])
    names = [n for n in names if n]
    if len(names) < 2:
        return None
    emails = card.get("emails") or []
    phones = card.get("phones") or []
    result: List[Dict[str, Any]] = []
    for i, nm in enumerate(names[:4]):
        result.append(
            {
                **card,
                "full_name": nm,
                "names": [nm],
                "emails": [emails[i]] if len(emails) == len(names) else emails,
                "phones": [phones[i]] if len(phones) == len(names) else phones,
            }
        )
    return result

# ---------- core processor ----------
async def _process_extract(
    uploads: List[UploadFile],
    chat_id: Optional[str],
    source: str,
    attachment_urls: Optional[List[str]],
    return_vcard: bool,
) -> Dict[str, Any]:
    if not uploads:
        raise HTTPException(status_code=400, detail="No file provided. Attach image(s) or a PDF.")

    image_like = [f for f in uploads if _looks_image(f.filename or "", f.content_type)]
    candidates = image_like if image_like else [f for f in uploads if _looks_pdf(f.filename or "", f.content_type)]
    if not candidates:
        raise HTTPException(status_code=415, detail="Unsupported file type. Upload image(s) or PDF.")
    candidates = candidates[:2]  # front/back or two cards

    # run gemini on each file
    per_file_cards: List[Dict[str, Any]] = []
    for up in candidates:
        raw = await up.read()
        if not raw:
            continue
        mime = _guess_mime(up.filename or "", up.content_type)
        try:
            bc: BusinessCard = extract_business_card(raw, mime_type=mime)  # should be Pydantic model
            cdict = bc.model_dump()
            per_file_cards.append(cdict)
        finally:
            try:
                await up.close()
            except Exception:
                pass

    if not per_file_cards:
        raise HTTPException(status_code=422, detail="No usable image/PDF content found.")

    # If single file has multiple names, split
    if len(per_file_cards) == 1:
        split = _split_single_card_if_multi(per_file_cards[0])
        if split:
            per_file_cards = split

    # Merge same person across files; keep distinct people separate
    persons = _cluster_and_merge(per_file_cards)

    # Save to DB (best-effort)
    saved_ids: List[str] = []
    if DBHandler is not None:
        try:
            db = DBHandler()
            saved_ids = db.save_business_contacts(
                contacts=persons,
                chat_id=chat_id,
                source=source or "viz",
                attachment_urls=attachment_urls or [],
            )
        except Exception:
            # Do not fail the API if DB write fails—just proceed
            saved_ids = []

    # Build response
    if len(persons) == 1:
        card = persons[0]
        md = _format_card_markdown(card)
        resp: Dict[str, Any] = {
            "status": "ok",
            "message": md,  # <-- primary chat text
            "data": {
                "message": md,   # mirror for UIs expecting it inside data
                "whatsapp": md,
                "text": md,
                "json": card,
                "image_url": None,
            },
            "answer": md,
            "meta": {
                "provider": "gemini",
                "model": "gemini-2.5-flash",
                "num_images": len(candidates),
                "contact_ids": saved_ids,
            },
            "card": card,
        }
        if return_vcard:
            resp["vcard"] = _vcard_of(card)  # only when explicitly asked
        return resp

    # multiple contacts
    md_all = []
    for i, c in enumerate(persons, 1):
        md_all.append(f"### Contact {i}\n{_format_card_markdown(c)}")
    md_join = "\n\n---\n\n".join(md_all)

    resp: Dict[str, Any] = {
        "status": "ok",
        "message": md_join,  # <-- primary chat text
        "data": {
            "message": md_join,
            "whatsapp": md_join,
            "text": md_join,
            "json": {"contacts": persons},
            "image_url": None,
        },
        "answer": md_join,
        "meta": {
            "provider": "gemini",
            "model": "gemini-2.5-flash",
            "num_images": len(candidates),
            "contact_ids": saved_ids,
        },
        "cards": persons,
    }
    if return_vcard:
        resp["vcards"] = [_vcard_of(c) for c in persons]  # only when explicitly asked
    return resp


# ---------- New main endpoint ----------
@router.post("/extract-business-card")
async def extract_business_card_route(
    # text params
    chat_id: Optional[str] = Form(None),
    source: str = Form("viz"),
    # IMPORTANT: default False -> no vCard returned unless explicitly requested
    return_vcard: bool = Form(False),

    # optional URLs of already-uploaded images (for traceability)
    attachment_urls: Optional[List[str]] = Form(None),

    # preferred fastapi file params
    images: Optional[List[UploadFile]] = File(None, description="images"),
    files: Optional[List[UploadFile]] = File(None, description="files"),

    # aliases often used by frontends
    images_bracket: Optional[List[UploadFile]] = File(None, alias="images[]"),
    files_bracket: Optional[List[UploadFile]] = File(None, alias="files[]"),

    # legacy single/paired
    front_image: Optional[UploadFile] = File(None),
    back_image: Optional[UploadFile] = File(None),
    front_file_camel: Optional[UploadFile] = File(None, alias="frontFile"),
    back_file_camel: Optional[UploadFile] = File(None, alias="backFile"),
    single_file: Optional[UploadFile] = File(None, alias="file"),
):
    """
    Extract business-card details (1–2 uploads). Saves contacts to DB.
    Returns:
      - one person: {"message": "...", "card": {...}} (no vCard by default)
      - multiple:   {"message": "...", "cards": [{...}, {...}]}
      - always includes: status/data/answer/meta
    """
    # collect uploads (keep order)
    incoming: List[UploadFile] = []
    for group in (images, files, images_bracket, files_bracket):
        if group:
            incoming.extend(group)
    for single in (front_image, back_image, front_file_camel, back_file_camel, single_file):
        if single:
            incoming.append(single)

    return await _process_extract(
        uploads=incoming,
        chat_id=chat_id,
        source=source or "viz",
        attachment_urls=attachment_urls,
        return_vcard=bool(return_vcard),
    )


# ---------- Legacy route kept for backward compatibility ----------
@router.post("/ask-image")
async def ask_image_legacy(
    prompt: str = Form(""),
    question: Optional[str] = Form(None),
    text: Optional[str] = Form(None),

    images: Optional[List[UploadFile]] = File(None, description="images"),
    files: Optional[List[UploadFile]] = File(None, description="files"),
    images_bracket: Optional[List[UploadFile]] = File(None, alias="images[]"),
    files_bracket: Optional[List[UploadFile]] = File(None, alias="files[]"),
    front_image: Optional[UploadFile] = File(None),
    back_image: Optional[UploadFile] = File(None),
    front_file_camel: Optional[UploadFile] = File(None, alias="frontFile"),
    back_file_camel: Optional[UploadFile] = File(None, alias="backFile"),
    single_file: Optional[UploadFile] = File(None, alias="file"),
):
    """
    Old endpoint name. Behaves like /extract-business-card (no DB params).
    """
    incoming: List[UploadFile] = []
    for group in (images, files, images_bracket, files_bracket):
        if group:
            incoming.extend(group)
    for single in (front_image, back_image, front_file_camel, back_file_camel, single_file):
        if single:
            incoming.append(single)

    # Legacy keeps vCard OFF by default too (clean chat text)
    return await _process_extract(
        uploads=incoming,
        chat_id=None,
        source="viz",
        attachment_urls=None,
        return_vcard=False,
    )
