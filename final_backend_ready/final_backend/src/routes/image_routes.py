# backend/routes/image_routes.py
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from typing import List, Optional
import base64
import os
import httpx

router = APIRouter()

# Configure Ollama target (can be overridden via env)
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://192.168.0.88:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llava:13b")  # vision-capable model
OLLAMA_ENDPOINT = f"{OLLAMA_URL.rstrip('/')}/api/generate"


async def _file_to_b64(file: UploadFile) -> str:
    """Read an UploadFile and return a base64-encoded string."""
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail=f"Empty file: {file.filename}")
    return base64.b64encode(content).decode("utf-8")


@router.post("/ask-image")
async def ask_image(
    # New: prompt comes from the form
    prompt: str = Form(...),

    # New: support multiple images via images[]
    images: Optional[List[UploadFile]] = File(None),

    # Back-compat: support the old two-file style
    front_image: Optional[UploadFile] = File(None),
    back_image: Optional[UploadFile] = File(None),
):
    """
    Accept a prompt + one or more images, call Ollama's /api/generate with
    a vision model (e.g., llava), and return the model's answer.
    """
    # Collect all incoming images (any of the supported fields)
    files: List[UploadFile] = []
    if images:
        files.extend(images)
    if front_image:
        files.append(front_image)
    if back_image:
        files.append(back_image)

    if not files:
        raise HTTPException(status_code=400, detail="No image provided.")

    # Convert images to base64 strings for Ollama
    try:
        b64_images = [await _file_to_b64(f) for f in files]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read image(s): {e}")

    # Build Ollama payload
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "images": b64_images,
        "stream": False,  # return a single JSON response
    }

    # Call Ollama
    try:
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(OLLAMA_ENDPOINT, json=payload)
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPStatusError as e:
        detail = e.response.text if e.response is not None else str(e)
        raise HTTPException(
            status_code=502, detail=f"Ollama error ({e.response.status_code if e.response else 'unknown'}): {detail}"
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Failed to reach Ollama: {e}")

    # Ollama returns {"model": "...", "created_at": "...", "response": "...", ...}
    answer = data.get("response", "").strip()
    if not answer:
        # Return raw for debugging if no 'response' field
        return {
            "status": "error",
            "message": "No response from model.",
            "ollama_raw": data,
        }

    return {
        "status": "success",
        "answer": answer,
        "meta": {
            "model": data.get("model", OLLAMA_MODEL),
            "num_images": len(b64_images),
        },
    }
