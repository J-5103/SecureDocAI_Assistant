# backend/routes/image_routes.py
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from typing import List, Optional
import base64
import io
import os
import importlib
import httpx

router = APIRouter()

# --------- Config ----------
# Point these at your Ollama host / model (your host is 192.168.0.88:11434)
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://192.168.0.88:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llava:13b")
OLLAMA_ENDPOINT = f"{OLLAMA_URL}/api/generate"

VISION_MAX_IMAGES = int(os.getenv("VISION_MAX_IMAGES", "6"))   # safety cap
MAX_IMG_SIDE      = int(os.getenv("VISION_MAX_SIDE",  "1600")) # soft downscale

# ---------- Helpers (lazy imports for optional deps) ----------
def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("utf-8")

def _is_image_filename(name: str) -> bool:
    n = (name or "").lower()
    return n.endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".gif"))

def _is_pdf_filename(name: str) -> bool:
    return (name or "").lower().endswith(".pdf")

def _is_image_ct(ct: Optional[str]) -> bool:
    return (ct or "").lower().startswith("image/")

def _is_pdf_ct(ct: Optional[str]) -> bool:
    ct = (ct or "").lower()
    return ct in ("application/pdf", "application/x-pdf")

def _require_pillow_image_module():
    try:
        return importlib.import_module("PIL.Image")
    except Exception:
        raise HTTPException(status_code=500, detail="Pillow not installed. Please `pip install pillow`.")

def _require_fitz_module():
    try:
        return importlib.import_module("fitz")  # PyMuPDF
    except Exception:
        raise HTTPException(status_code=500, detail="PyMuPDF not installed. Please `pip install pymupdf`.")

def _image_bytes_to_png_b64(raw: bytes) -> str:
    Image = _require_pillow_image_module()
    img = Image.open(io.BytesIO(raw)).convert("RGB")

    # Downscale very large images (keeps aspect)
    try:
        w, h = img.size
        if MAX_IMG_SIDE and max(w, h) > MAX_IMG_SIDE:
            if w >= h:
                nh = int(h * (MAX_IMG_SIDE / float(w)))
                img = img.resize((MAX_IMG_SIDE, max(1, nh)))
            else:
                nw = int(w * (MAX_IMG_SIDE / float(h)))
                img = img.resize((max(1, nw), MAX_IMG_SIDE))
    except Exception:
        pass

    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return _b64(buf.getvalue())

def _pdf_first_page_to_png_b64(raw: bytes, dpi: int = 180) -> str:
    fitz = _require_fitz_module()
    doc = fitz.open(stream=raw, filetype="pdf")
    if doc.page_count == 0:
        raise HTTPException(status_code=400, detail="Empty PDF.")
    page = doc.load_page(0)
    pix = page.get_pixmap(dpi=dpi)
    return _b64(pix.tobytes("png"))

async def _to_b64_for_vision(file: UploadFile) -> Optional[str]:
    """
    Normalize any supported upload to a PNG base64 string:
      - image/* or image file extension -> PNG base64
      - application/pdf or .pdf -> first page PNG base64
    Unsupported types return None.
    """
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail=f"Empty file: {file.filename}")

    ct = (file.content_type or "").lower()
    name = file.filename or ""

    if _is_image_ct(ct) or _is_image_filename(name):
        return _image_bytes_to_png_b64(raw)
    if _is_pdf_ct(ct) or _is_pdf_filename(name):
        return _pdf_first_page_to_png_b64(raw)

    return None

# ---------- Route ----------
@router.post("/ask-image")
async def ask_image(
    # Prompt is optional; defaults to extraction intent
    prompt: str = Form("Extract key information from this image. Provide a concise summary and a JSON if possible."),

    # Preferred multi-upload names
    images: Optional[List[UploadFile]] = File(None, description="images"),
    files: Optional[List[UploadFile]] = File(None, description="files"),

    # Accept bracketed aliases from some UIs (images[] / files[])
    images_bracket: Optional[List[UploadFile]] = File(None, alias="images[]"),
    files_bracket: Optional[List[UploadFile]] = File(None, alias="files[]"),

    # Back-compat BOTH snake_case and camelCase
    front_image: Optional[UploadFile] = File(None),
    back_image: Optional[UploadFile] = File(None),
    front_file_camel: Optional[UploadFile] = File(None, alias="frontFile"),
    back_file_camel: Optional[UploadFile] = File(None, alias="backFile"),

    # Some callers send a single 'file'
    single_file: Optional[UploadFile] = File(None, alias="file"),
):
    """
    Rule: If any images are attached with the question → ALWAYS call LLaVA and extract info
    from those images (ignore any currently selected document).
    If only PDFs are attached, render first page and send to LLaVA.
    """

    # 1) Collect uploads from all supported field names
    incoming: List[UploadFile] = []
    for group in (images, files, images_bracket, files_bracket):
        if group:
            incoming.extend(group)
    for single in (front_image, back_image, front_file_camel, back_file_camel, single_file):
        if single:
            incoming.append(single)

    if not incoming:
        raise HTTPException(status_code=400, detail="No file provided. Attach image(s) or PDF.")

    # 2) Prefer real images; else PDFs; allow up to VISION_MAX_IMAGES
    image_like: List[UploadFile] = [
        f for f in incoming
        if _is_image_ct(f.content_type) or _is_image_filename(f.filename or "")
    ]
    candidates: List[UploadFile] = image_like if image_like else [
        f for f in incoming if _is_pdf_ct(f.content_type) or _is_pdf_filename(f.filename or "")
    ]
    if not candidates:
        raise HTTPException(status_code=415, detail="Unsupported file type. Upload image(s) or PDF.")

    candidates = candidates[:VISION_MAX_IMAGES]

    # 3) Normalize → PNG base64 list (deduplicate)
    b64_list: List[str] = []
    seen = set()
    for f in candidates:
        try:
            b64 = await _to_b64_for_vision(f)
            if b64 and b64 not in seen:
                seen.add(b64)
                b64_list.append(b64)
        finally:
            try:
                await f.close()
            except Exception:
                pass

    if not b64_list:
        raise HTTPException(status_code=415, detail="Could not prepare any image for the vision model.")

    # 4) Call Ollama LLaVA (bubble up Ollama's error message/status)
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt or "Describe the image.",
        "images": b64_list,   # raw base64 strings (no data: prefix)
        "stream": False,
    }

    try:
        async with httpx.AsyncClient(timeout=180) as client:
            resp = await client.post(OLLAMA_ENDPOINT, json=payload)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Failed to reach Ollama at {OLLAMA_URL}: {e}")

    if resp.status_code >= 400:
        # Surface Ollama's exact message so the UI shows it (not just {}).
        raise HTTPException(status_code=resp.status_code, detail=resp.text or f"Ollama error {resp.status_code}")

    try:
        data = resp.json()
    except Exception:
        raise HTTPException(status_code=502, detail="Invalid JSON from Ollama /api/generate.")

    # 5) Normalize response for UI
    text = (data.get("response") or "").strip()
    return {
        "status": "ok",
        "data": {
            "whatsapp": text if text else "Processed the image(s).",
            "json": None,
            "image_urls": [],
        },
        "meta": {
            "provider": "ollama",
            "model": data.get("model", OLLAMA_MODEL),
            "num_images": len(b64_list),
        },
    }
