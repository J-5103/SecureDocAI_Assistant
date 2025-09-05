# src/pipeline/card_extractor.py
"""
Business-card extractor using Google's Gemini.

Priority:
  1) google-genai (pip install google-genai)
  2) google-generativeai (pip install google-generativeai)
  3) REST fallback (no SDK needed)

Env:
  GEMINI_API_KEY=<key>          # preferred
  GOOGLE_API_KEY=<key>          # optional
  GENAI_API_KEY=<key>           # optional
  GOOGLE_GENAI_API_KEY=<key>    # optional
"""

from __future__ import annotations

import os
import json
import re
import base64
import importlib
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path
import logging

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# --- Load .env (best effort) -------------------------------------------------
def _load_envs() -> None:
    try:
        from dotenv import load_dotenv  # type: ignore
    except Exception:
        return
    here = Path(__file__).resolve()
    candidates = [
        Path.cwd() / ".env",                         # where uvicorn was started
        here.parent / ".env",                        # src/pipeline/.env
        here.parent.parent / ".env",                 # src/.env
        here.parent.parent.parent / ".env",          # project root
        here.parent.parent.parent / "backend/.env",  # backend/.env (if split)
    ]
    for p in candidates:
        try:
            if p.exists():
                load_dotenv(p, override=False)
        except Exception:
            pass

_load_envs()

# ----------------------------- Pydantic models -------------------------------
class PostalAddress(BaseModel):
    street: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    postal_code: Optional[str] = None
    country: Optional[str] = None


class BusinessCard(BaseModel):
    full_name: Optional[str] = Field(None, description="Person full name")
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    organization: Optional[str] = None
    job_title: Optional[str] = None
    phones: List[str] = Field(default_factory=list, description="Phone numbers")
    emails: List[str] = Field(default_factory=list, description="Email addresses")
    websites: List[str] = Field(default_factory=list, description="Web/URLs")
    address: Optional[PostalAddress] = None
    social: Dict[str, Optional[str]] = Field(
        default_factory=dict,
        description="e.g. linkedin, twitter, instagram, facebook, github",
    )
    notes: Optional[str] = None
    raw_text: Optional[str] = None


__all__ = ["PostalAddress", "BusinessCard", "extract_business_card"]


# ----------------------------- SDK import helpers ----------------------------
def _import_google_genai() -> Tuple[Optional[Any], Optional[Any]]:
    """
    Import google-genai dynamically.
    Returns (genai_module, types_module) or (None, None) if not installed/importable.
    """
    try:
        genai = importlib.import_module("google.genai")
        genai_types = importlib.import_module("google.genai.types")
        return genai, genai_types
    except Exception:
        return None, None


def _import_legacy_generativeai():
    """Optional fallback to the legacy google-generativeai SDK."""
    try:
        return importlib.import_module("google.generativeai")
    except Exception:
        return None


def _namespace_conflict_hint() -> str:
    """
    Detect the legacy 'google' distribution which breaks the 'google.*' namespace.
    If both 'google' and 'google-genai' are installed, imports can fail.
    """
    try:
        try:
            import importlib.metadata as md  # py3.8+
        except Exception:
            import importlib_metadata as md  # type: ignore

        names = {d.metadata.get("Name", "").lower() for d in md.distributions()}
        if "google" in names and ("google-genai" in names or "google-generativeai" in names):
            return (
                "\nDetected the legacy 'google' package alongside a Gemini SDK. "
                "This breaks namespace imports. Uninstall it:\n"
                "  pip uninstall -y google\n"
            )
    except Exception:
        pass
    return ""


# ----------------------------- Client builders -------------------------------
def _get_api_key() -> str:
    candidates = [
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "GENAI_API_KEY",
        "GOOGLE_GENAI_API_KEY",
    ]
    for k in candidates:
        v = (os.getenv(k) or "").strip()
        if v:
            if k != "GEMINI_API_KEY":
                logger.info(f"Using API key from {k}")
            return v
    raise RuntimeError(
        "Gemini API key missing. Set GEMINI_API_KEY (or GOOGLE_API_KEY / GENAI_API_KEY) "
        "in your environment or .env (e.g., src/.env)."
    )


def _build_genai_client():
    """Build client for google-genai (preferred)."""
    genai, genai_types = _import_google_genai()
    if not genai or not genai_types:
        return None, None
    client = genai.Client(api_key=_get_api_key())
    return client, genai_types


def _build_legacy_client():
    """Build client for google-generativeai (fallback)."""
    legacy = _import_legacy_generativeai()
    if not legacy:
        return None
    legacy.configure(api_key=_get_api_key())
    return legacy.GenerativeModel("gemini-1.5-flash")  # broad availability


# ----------------------------- REST fallback ---------------------------------
def _http_post_json(url: str, payload: Dict[str, Any], timeout: int = 60) -> Dict[str, Any]:
    # Prefer requests, fallback to urllib
    try:
        import requests  # type: ignore

        r = requests.post(url, json=payload, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception:
        import json as _json
        from urllib.request import Request, urlopen
        from urllib.error import HTTPError

        req = Request(
            url,
            data=_json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(req, timeout=timeout) as resp:  # type: ignore
                return _json.loads(resp.read().decode("utf-8"))
        except HTTPError as e:
            body = e.read().decode("utf-8", "ignore")
            raise RuntimeError(f"REST call failed: {e.status} {e.reason} :: {body}") from e


def _rest_generate_content(
    api_key: str,
    model: str,
    mime_type: str,
    image_bytes: bytes,
    prompt: str,
) -> str:
    """
    Use the Google Generative Language REST API directly.
    Works even if SDKs cannot be imported (e.g., 'google' package conflict).
    """
    # Some experimental model names are not in v1beta; fall back to 1.5-flash if needed.
    rest_model = model or "gemini-1.5-flash"
    if "2.5" in rest_model or "2.0" in rest_model:
        rest_model = "gemini-1.5-flash"

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{rest_model}:generateContent?key={api_key}"

    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {
                        "inline_data": {
                            "mime_type": mime_type,
                            "data": base64.b64encode(image_bytes).decode("ascii"),
                        }
                    },
                    {"text": prompt},
                ],
            }
        ],
        "generationConfig": {"responseMimeType": "application/json"},
    }

    data = _http_post_json(url, payload)
    # Extract text from the first candidate
    try:
        candidates = data.get("candidates") or []
        if not candidates:
            raise RuntimeError(f"Empty REST response: {data!r}")
        parts = (
            (candidates[0].get("content") or {}).get("parts")
            or candidates[0].get("content", {}).get("parts", [])
        )
        # Join all text parts just in case
        texts = []
        for p in parts or []:
            t = p.get("text")
            if t:
                texts.append(t)
        return "\n".join(texts).strip()
    except Exception as e:
        raise RuntimeError(f"Failed to parse REST response: {e} :: {data!r}") from e


# ----------------------------- Prompt / JSON helpers -------------------------
_SCHEMA_HINT = {
    "full_name": "string|null",
    "first_name": "string|null",
    "last_name": "string|null",
    "organization": "string|null",
    "job_title": "string|null",
    "phones": ["string"],
    "emails": ["string"],
    "websites": ["string"],
    "address": {
        "street": "string|null",
        "city": "string|null",
        "state": "string|null",
        "postal_code": "string|null",
        "country": "string|null",
    },
    "social": {
        "linkedin": "string|null",
        "twitter": "string|null",
        "instagram": "string|null",
        "facebook": "string|null",
        "github": "string|null",
    },
    "notes": "string|null",
    "raw_text": "string|null",
}


def _instruction_text() -> str:
    return (
        "You are a precise information extraction engine for business cards. "
        "Return ONLY a strict JSON object that matches the schema. "
        "If a field is missing on the card, use null or [] accordingly. "
        "Preserve visible punctuation/country codes in phone numbers. "
        "Extract websites (http/https or bare domain). "
        "Include a best-effort full OCR in `raw_text`."
        f"\nJSON schema (types, not examples): {json.dumps(_SCHEMA_HINT)}"
    )


def _extract_first_json(text: str) -> Optional[str]:
    """Extract the first top-level JSON object from text using a brace stack."""
    s = (text or "").strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    if s.startswith("{") and s.endswith("}"):
        return s
    start = s.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(s)):
        c = s[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return s[start : i + 1]
    return None


def _safe_json_from_text(text: str) -> Dict[str, Any]:
    """Parse JSON safely; if the model wrapped JSON in prose, extract with a stack parser."""
    try:
        return json.loads(text or "")
    except Exception:
        pass
    obj = _extract_first_json(text or "")
    if obj:
        return json.loads(obj)
    raise RuntimeError("Model did not return valid JSON.")


# ----------------------------- Public API ------------------------------------
def extract_business_card(
    image_bytes: bytes,
    mime_type: str = "image/jpeg",
    *,
    model: str = "gemini-2.5-flash",
) -> BusinessCard:
    """
    Extract structured contact info from one business-card image.

    Strategy:
      1) Try google-genai (new SDK) with plain JSON output.
      2) Fall back to google-generativeai (legacy).
      3) Fall back to REST (no SDK required).
    """
    api_key = _get_api_key()
    instruction = _instruction_text()

    # -------- Preferred path: google-genai (new SDK) --------
    client, genai_types = _build_genai_client()
    new_sdk_error = None
    if client and genai_types:
        try:
            try:
                img_part = genai_types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
            except Exception:
                img_part = {"mime_type": mime_type, "data": image_bytes}

            try:
                txt_part = genai_types.Part.from_text(instruction)
            except Exception:
                txt_part = instruction

            try:
                contents = [genai_types.Content(role="user", parts=[img_part, txt_part])]
            except Exception:
                contents = [img_part, txt_part]

            cfg = None
            try:
                cfg = genai_types.GenerateContentConfig(response_mime_type="application/json")
            except Exception:
                cfg = None

            if cfg is not None:
                resp = client.models.generate_content(model=model, contents=contents, config=cfg)
            else:
                resp = client.models.generate_content(model=model, contents=contents)

            text = getattr(resp, "text", "") or ""
            data = _safe_json_from_text(text)
            return BusinessCard(**data)
        except Exception as e:
            new_sdk_error = e  # try legacy/REST before failing

    # -------- Fallback path: google-generativeai (legacy) --------
    legacy_model = _build_legacy_client()
    legacy_error = None
    if legacy_model:
        try:
            result = legacy_model.generate_content(
                [
                    {"mime_type": mime_type, "data": image_bytes},
                    instruction,
                ],
                generation_config={"response_mime_type": "application/json"},
            )
            text = getattr(result, "text", None) or ""
            data = _safe_json_from_text(text)
            return BusinessCard(**data)
        except Exception as e:
            legacy_error = e  # try REST next

    # -------- REST fallback (no SDK) -----------------------------------------
    try:
        text = _rest_generate_content(api_key, model, mime_type, image_bytes, instruction)
        data = _safe_json_from_text(text)
        return BusinessCard(**data)
    except Exception as e:
        rest_error = e

    # -------- Nothing worked: raise a helpful error --------------------------
    hint = _namespace_conflict_hint()
    if new_sdk_error or legacy_error:
        raise RuntimeError(
            "Gemini extraction failed.\n"
            f"- google-genai error: {new_sdk_error!r}\n"
            f"- google-generativeai error: {legacy_error!r}\n"
            f"- REST error: {rest_error!r}\n"
            + hint
            + "\nIf the SDK import keeps failing, uninstall the legacy 'google' package:\n"
            "  pip uninstall -y google\n"
            "Then install ONE of:\n"
            "  pip install -U google-genai\n"
            "  or\n"
            "  pip install -U google-generativeai\n"
        )
    # SDKs not available and REST failed (bad key / network)
    raise RuntimeError(
        "No Gemini SDK found and REST call failed. "
        "Check your GEMINI_API_KEY / network.\n"
        f"Details: {rest_error!r}"
        + hint
    )
