# no PyMuPDF. Uses pypdfium2 only.
import io, base64, mimetypes, requests
import pypdfium2 as pdfium

def _to_data_uri_bytes(b: bytes, mime: str) -> str:
    return f"data:{mime};base64," + base64.b64encode(b).decode()

def _pdf_to_text_or_images(pdf_bytes: bytes, text_min_len: int = 800, scale: float = 2.0):
    pdf = pdfium.PdfDocument(io.BytesIO(pdf_bytes))
    n = len(pdf)

    # --- text first ---
    texts = []
    for i in range(n):
        page = pdf[i]
        tp = page.get_textpage()
        try:
            txt = tp.get_text_range()  # full text of the page
        finally:
            tp.close()
        if txt:
            texts.append(txt)
    full = "\n".join(t.strip() for t in texts if t).strip()
    if len(full) >= text_min_len:
        return {"mode": "text", "text": full}

    # --- else render to images ---
    images = []
    for i in range(n):
        page = pdf[i]
        pil_img = page.render(scale=scale).to_pil()  # PNG via PIL
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        images.append(_to_data_uri_bytes(buf.getvalue(), "image/png"))
    return {"mode": "images", "images": images}

def _opts(num_ctx=4096, num_predict=800, temperature=0, num_gpu=None):
    o = {"num_ctx": num_ctx, "num_predict": num_predict, "temperature": temperature}
    if num_gpu is not None:
        o["num_gpu"] = num_gpu
    return o

def generate_minicpm(base_url: str, prompt: str, images=None, prefer_gpu=True, timeout=180):
    payload = {
        "model": "minicpm-v",
        "prompt": prompt,
        "stream": False,
        "options": _opts(num_ctx=4096, num_predict=800, temperature=0,
                         num_gpu=None if prefer_gpu else 0)
    }
    if images:
        payload["images"] = images

    r = requests.post(f"{base_url}/api/generate", json=payload, timeout=timeout)
    if r.status_code == 200:
        return r.json()

    # fallback to CPU with tighter limits (prevents 500 on Windows GPUs)
    payload["options"] = _opts(num_ctx=2048, num_predict=400, temperature=0, num_gpu=0)
    r2 = requests.post(f"{base_url}/api/generate", json=payload, timeout=timeout)
    r2.raise_for_status()
    return r2.json()

def summarize_pdf_bytes(base_url: str, pdf_bytes: bytes, user_task: str = "Give me a concise summary."):
    sel = _pdf_to_text_or_images(pdf_bytes)
    if sel["mode"] == "text":
        prompt = (
            "You are a precise document summarizer. Use ONLY the text between <<<DOC>>> markers. "
            "Return 5–8 bullets, and list any dates/amounts if present.\n\n"
            "<<<DOC>>>\n" + sel["text"] + "\n<<<DOC>>>"
        )
        return generate_minicpm(base_url, prompt, images=None, prefer_gpu=True)

    prompt = (
        "You are a precise KYC/forms summarizer. Pages are scans/photos/handwriting. "
        "Summarize in 5–8 bullets. If IDs/forms appear, also extract key fields concisely: "
        "Aadhaar: name, dob_or_yob, gender, aadhaar_number, address; "
        "PAN: name, father_name, dob, pan_number; "
        "Ration: card_number, state, holder_name, family_members, address. "
        "If something is unreadable, say Unreadable."
    )
    return generate_minicpm(base_url, prompt, images=sel["images"], prefer_gpu=True)
