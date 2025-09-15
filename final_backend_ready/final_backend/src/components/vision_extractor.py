# src/components/vision_extractor.py
# Hybrid extractor for image-based PDFs (IDs/forms/invoices):
# 1) PDF -> cleaned images (deskew/denoise)
# 2) OCR text (multi-language)
# 3) Regex/rules for high-confidence fields (Aadhaar, PAN, dates, mobile)
# 4) VLM (Ollama) fills remaining fields from page images
# 5) Final normalization & simple validations

from __future__ import annotations

import os
import re
import io
import json
import base64
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import fitz  # PyMuPDF
import cv2
import numpy as np
from dateutil import parser as dateparser
from paddleocr import PaddleOCR

# ---------------------------
# Patterns & light validators
# ---------------------------

AADHAAR_RX = re.compile(r"\b(\d{4}\s?\d{4}\s?\d{4})\b")
PAN_RX     = re.compile(r"\b([A-Z]{5}[0-9]{4}[A-Z])\b")
DATE_RX    = re.compile(r"\b(?:\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|[A-Za-z]{3,9}\s+\d{1,2},\s*\d{4})\b")
MOBILE_RX  = re.compile(r"\b(?:\+?91[-\s]?)?\d{10}\b")
EMAIL_RX   = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")

def _normalize_date(s: str) -> Optional[str]:
    try:
        dt = dateparser.parse(s, dayfirst=True, fuzzy=True)
        return dt.strftime("%Y-%m-%d")
    except Exception:
        return None

def _verhoeff_validate_aadhaar(a12: str) -> bool:
    # Verhoeff checksum table (for Aadhaar)
    d = [
        [0,1,2,3,4,5,6,7,8,9],
        [1,2,3,4,0,6,7,8,9,5],
        [2,3,4,0,1,7,8,9,5,6],
        [3,4,0,1,2,8,9,5,6,7],
        [4,0,1,2,3,9,5,6,7,8],
        [5,9,8,7,6,0,4,3,2,1],
        [6,5,9,8,7,1,0,4,3,2],
        [7,6,5,9,8,2,1,0,4,3],
        [8,7,6,5,9,3,2,1,0,4],
        [9,8,7,6,5,4,3,2,1,0],
    ]
    p = [
        [0,1,2,3,4,5,6,7,8,9],
        [1,5,7,6,2,8,3,0,9,4],
        [5,8,0,3,7,9,6,1,4,2],
        [8,9,1,6,0,4,3,5,2,7],
        [9,4,5,3,1,2,6,8,7,0],
        [4,2,8,6,5,7,3,9,0,1],
        [2,7,9,3,8,0,5,4,1,6],
        [7,0,4,6,9,1,2,3,5,8],
    ]
    inv = [0,4,3,2,1,5,6,7,8,9]
    c = 0
    s = a12.replace(" ", "")
    if len(s) != 12 or not s.isdigit():
        return False
    for i, ch in enumerate(reversed(s)):
        c = d[c][p[(i + 1) % 8][int(ch)]]
    return c == 0

# ---------------------------
# Image helpers (deskew/clean)
# ---------------------------

def _deskew(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    inv = cv2.bitwise_not(gray)
    thr = cv2.threshold(inv, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thr > 0))
    if coords.size < 2:
        return img
    angle = cv2.minAreaRect(coords)[-1]
    angle = -(90 + angle) if angle < -45 else -angle
    (h, w) = img.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

def _enhance(img: np.ndarray) -> np.ndarray:
    x = cv2.fastNlMeansDenoisingColored(img, None, 5, 5, 7, 21)
    k = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]], dtype=np.float32)  # sharpen
    x = cv2.filter2D(x, -1, k)
    return x

def _im_encode_png(img: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", img)
    return buf.tobytes() if ok else b""

def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")

# ---------------------------
# Core extractor
# ---------------------------

@dataclass
class ExtractResult:
    # KYC-ish superset; add fields as needed
    name: Optional[str] = None
    gender: Optional[str] = None
    aadhaar_number: Optional[str] = None
    pan_number: Optional[str] = None
    address: Optional[str] = None
    mobile: Optional[str] = None
    email: Optional[str] = None
    dob: Optional[str] = None
    ration_card_number: Optional[str] = None
    form_title: Optional[str] = None
    signatures_present: Optional[bool] = None
    photo_present: Optional[bool] = None
    dates: Optional[List[str]] = None
    raw_text: str = ""

class VisionExtractor:
    """
    OCR-first + VLM-verify pipeline.
    Config via ENV:
      OCR_LANGS=en,hi,gu
      OCR_DPI=220
      OCR_MIN_CONF=0.60
      EXTRACTOR_USE_VLM=true
    """
    def __init__(self):
        langs = (os.getenv("OCR_LANGS") or "en").split(",")
        langs = [l.strip() for l in langs if l.strip()]
        primary = langs[0] if langs else "en"
        self.ocr = PaddleOCR(use_angle_cls=True, lang=primary)
        self.ocr_fallback = None
        if len(langs) > 1:
            # try a second language as fallback (e.g., hi or gu)
            self.ocr_fallback = PaddleOCR(use_angle_cls=True, lang=langs[1])

        self.min_conf = float(os.getenv("OCR_MIN_CONF", "0.60"))
        self.dpi = int(os.getenv("OCR_DPI", "220"))
        self.use_vlm = (os.getenv("EXTRACTOR_USE_VLM", "true").lower() == "true")

    # ---------- PDF -> images ----------
    def pdf_to_images(self, pdf_path: str) -> List[np.ndarray]:
        doc = fitz.open(pdf_path)
        out: List[np.ndarray] = []
        for p in doc:
            pix = p.get_pixmap(dpi=self.dpi)
            # pix is BGRA; convert to BGR
            arr = np.frombuffer(pix.tobytes(), dtype=np.uint8)
            img = arr.reshape(pix.h, pix.w, 4)
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            out.append(img)
        return out

    # ---------- OCR one page ----------
    def ocr_page(self, img: np.ndarray) -> Tuple[str, List[Tuple[str, float, Tuple[int,int,int,int]]]]:
        img2 = _deskew(img)
        img2 = _enhance(img2)

        lines: List[Tuple[str, float, Tuple[int,int,int,int]]] = []
        res = self.ocr.ocr(img2)
        if res and res[0]:
            for (box, (txt, conf)) in res[0]:
                if conf >= self.min_conf:
                    x1, y1 = map(int, box[0]); x2, y2 = map(int, box[2])
                    lines.append((txt, float(conf), (x1, y1, x2, y2)))

        if not lines and self.ocr_fallback:
            res2 = self.ocr_fallback.ocr(img2)
            if res2 and res2[0]:
                for (box, (txt, conf)) in res2[0]:
                    if conf >= self.min_conf:
                        x1, y1 = map(int, box[0]); x2, y2 = map(int, box[2])
                        lines.append((txt, float(conf), (x1, y1, x2, y2)))

        full_text = "\n".join(t for (t, _, __) in lines)
        return full_text, lines

    # ---------- Regex/rules pass ----------
    def regex_pass(self, text: str) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        # Aadhaar
        a = AADHAAR_RX.search(text)
        if a:
            raw = re.sub(r"\D", "", a.group(1))
            if len(raw) == 12 and _verhoeff_validate_aadhaar(raw):
                out["aadhaar_number"] = f"{raw[:4]} {raw[4:8]} {raw[8:]}"
        # PAN
        p = PAN_RX.search(text)
        if p:
            out["pan_number"] = p.group(1)
        # Mobile
        m = MOBILE_RX.search(text)
        if m:
            out["mobile"] = re.sub(r"\D", "", m.group(0))[-10:]
        # Email
        e = EMAIL_RX.search(text)
        if e:
            out["email"] = e.group(0)
        # Dates (normalized)
        found_dates = []
        for d in DATE_RX.findall(text):
            norm = _normalize_date(d)
            if norm:
                found_dates.append(norm)
        if found_dates:
            out["dates"] = sorted(set(found_dates))
        return out

    def _pages_to_base64(self, pages: List[np.ndarray]) -> List[str]:
        return [_b64(_im_encode_png(p)) for p in pages]

    # ---------- VLM call (Ollama /api/chat) ----------
    def _vlm_fill(self, ollama_base: str, model: str, prompt: str, images_b64: List[str], num_ctx=3072) -> Dict[str, Any]:
        import requests
        url = ollama_base.rstrip("/") + "/api/chat"
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": "You extract fields from ID forms and scanned documents. Output STRICT JSON only, no prose."},
                {"role": "user", "content": prompt, "images": images_b64}
            ],
            "stream": False,
            "options": {"num_ctx": int(num_ctx)}
        }
        r = requests.post(url, json=payload, timeout=180)
        r.raise_for_status()
        j = r.json()
        content = (j.get("message") or {}).get("content") or j.get("response") or ""
        # try JSON decode (tolerant)
        try:
            return json.loads(content)
        except Exception:
            content = content.strip()
            # remove code fences if any
            content = re.sub(r"^```json\s*|\s*```$", "", content, flags=re.IGNORECASE | re.MULTILINE)
            return json.loads(content)

    # ---------- Public entry ----------
    def extract(self, pdf_path: str, ollama_base: Optional[str] = None, vlm_model: Optional[str] = None) -> Dict[str, Any]:
        """
        Returns a dict with fields:
        name, gender, aadhaar_number, pan_number, address, mobile, email, dob, ration_card_number,
        form_title, signatures_present, photo_present, dates[], raw_text
        """
        base = (ollama_base or os.getenv("OLLAMA_BASE_URL") or "http://127.0.0.1:11434").strip()
        model = (vlm_model or os.getenv("OLLAMA_MODEL") or "qwen2.5vl:latest").strip()

        pages = self.pdf_to_images(pdf_path)
        ocr_texts: List[str] = []
        for img in pages:
            t, _ = self.ocr_page(img)
            ocr_texts.append(t)

        joined = "\n".join(ocr_texts)
        rules = self.regex_pass(joined)

        result: Dict[str, Any] = {
            "name": None,
            "gender": None,
            "aadhaar_number": rules.get("aadhaar_number"),
            "pan_number": rules.get("pan_number"),
            "address": None,
            "mobile": rules.get("mobile"),
            "email": rules.get("email"),
            "dob": None,
            "ration_card_number": None,
            "form_title": None,
            "signatures_present": None,
            "photo_present": None,
            "dates": rules.get("dates", []),
            "raw_text": joined[:5000],
        }

        if not self.use_vlm:
            return result

        # Ask VLM to fill the gaps using both OCR text (context) + page images
        prompt = (
            "Extract these fields as STRICT JSON (null if absent/unreadable): "
            "name, gender, aadhaar_number, pan_number, address, mobile, email, "
            "dob(YYYY-MM-DD), ration_card_number, form_title, signatures_present(bool), "
            "photo_present(bool), dates(list of YYYY-MM-DD).\n"
            "Rules:\n"
            "- Prefer OCR_TEXT when clear; otherwise read directly from the images.\n"
            "- Do not guess; prefer null over guessing.\n"
            "- Aadhaar must be 12 digits (Verhoeff valid); PAN must match AAAAA9999A.\n"
            f"OCR_TEXT (truncated):\n{result['raw_text']}"
        )

        try:
            images_b64 = self._pages_to_base64(pages)
            filled = self._vlm_fill(base, model, prompt, images_b64, num_ctx=int(os.getenv("OLLAMA_NUM_CTX", "3072")))
            # Merge conservatively (keep regex-validated IDs if present)
            for k, v in filled.items():
                if k in ("aadhaar_number", "pan_number") and result.get(k):
                    continue
                result[k] = v
        except Exception as e:
            # If VLM fails, return OCR/rules-only result
            pass

        # Final normalization
        if result.get("dob"):
            norm = _normalize_date(str(result["dob"]))
            result["dob"] = norm or result["dob"]

        if result.get("aadhaar_number"):
            raw = re.sub(r"\D", "", str(result["aadhaar_number"]))
            if len(raw) == 12 and _verhoeff_validate_aadhaar(raw):
                result["aadhaar_number"] = f"{raw[:4]} {raw[4:8]} {raw[8:]}"
            else:
                # invalidate obviously wrong aadhaar
                result["aadhaar_number"] = None

        if result.get("pan_number"):
            if not PAN_RX.match(str(result["pan_number"])):
                result["pan_number"] = None

        return result
