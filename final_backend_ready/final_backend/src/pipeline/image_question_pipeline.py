import base64
import json
import re
import requests
from typing import Optional, Dict, Any, List


class ImageQuestionPipeline:
    """
    Works with Ollama (LLaVA) and returns a structured dict:
      {
        "whatsapp": "<formatted whatsapp message>",
        "vcard": "BEGIN:VCARD...END:VCARD",
        "json": {...}
      }

    Usage:
        pipe = ImageQuestionPipeline(ollama_base="http://192.168.0.88:11434", model_name="llava:13b")
        out = pipe.run(front_bytes, back_bytes_or_None)
    """

    def __init__(
        self,
        ollama_base: Optional[str] = None,
        ollama_url: Optional[str] = None,
        model_name: str = "llava:13b",
        timeout_sec: int = 200,
    ):
        # Prefer base (we build /api/chat and /api/generate). Keep legacy ollama_url for back-compat.
        self.model_name = model_name
        self.timeout = timeout_sec
        if ollama_base:
            base = ollama_base.rstrip("/")
            self.chat_url = f"{base}/api/chat"
            self.generate_url = f"{base}/api/generate"
        else:
            u = (ollama_url or "").rstrip("/")
            self.chat_url = u.replace("/api/generate", "/api/chat")
            self.generate_url = u if u else "http://192.168.0.88:11434/api/generate"

    # ------------------------ public ------------------------

    def run(self, front_image_bytes: bytes, back_image_bytes: Optional[bytes] = None) -> Dict[str, Any]:
        """
        Returns a dict with keys: whatsapp, vcard, json.
        """
        front_b64 = base64.b64encode(front_image_bytes).decode("utf-8")
        back_b64 = base64.b64encode(back_image_bytes).decode("utf-8") if back_image_bytes else None

        system_prompt = (
            "You are an AI contact extraction agent. Analyze one or two images of a business card and "
            "extract ONLY visible contact details (no guessing). "
            "Return the result in EXACTLY three sections, in this order:\n\n"
            "1) WhatsApp Message\n"
            "2) vCard 3.0\n"
            "3) JSON\n\n"
            "Formatting rules:\n"
            "- WhatsApp: short, clean Markdown; use emojis sparingly; include only fields you have.\n"
            "- vCard 3.0: BEGIN:VCARD..END:VCARD, FN, ORG, TITLE, TEL (international), EMAIL, URL, "
            "  ADR (single line), NOTE, and X-SOCIALPROFILE for social links if present.\n"
            "- JSON: a VALID JSON object with keys (omit missing): "
            "  name, company, title, phones (array), emails (array), website, address, city, state, postal_code, country, "
            "  linkedin, twitter, instagram, facebook, github, other_social (array), notes\n"
            "- No placeholders or empty fields. No extra commentary beyond the 3 sections."
        )
        user_instructions = (
            "You are given one or two visiting card images (front and optionally back). "
            "Extract and return all available contact information in exactly three formats as specified."
        )

        # ---- Try /api/chat (multimodal messages) ----
        try:
            msgs = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_instructions},
                {"role": "user", "content": "Front image"},
                {"role": "user", "content": {"type": "image", "image": front_b64}},
            ]
            if back_b64:
                msgs.append({"role": "user", "content": "Back image"})
                msgs.append({"role": "user", "content": {"type": "image", "image": back_b64}})

            chat_payload = {
                "model": self.model_name,
                "messages": self._to_ollama_chat_messages(msgs),
                "stream": False,
                # keep options conservative; tweak as needed
                "options": {"temperature": 0.2, "num_predict": 1200},
            }
            r = requests.post(self.chat_url, json=chat_payload, timeout=self.timeout)
            if r.ok:
                text = self._extract_text_from_chat(r.json())
                parsed = self._parse_three_blocks(text)
                return self._postprocess_fill_missing(parsed)
        except Exception:
            # fall through to /api/generate
            pass

        # ---- Fallback: /api/generate (images under "images") ----
        try:
            images = [front_b64] + ([back_b64] if back_b64 else [])
            prompt = system_prompt + "\n\n" + user_instructions
            gen_payload = {
                "model": self.model_name,
                "prompt": prompt,
                "images": images if images else None,
                "stream": False,
                "options": {"temperature": 0.2, "num_predict": 1200},
            }
            r = requests.post(self.generate_url, json=gen_payload, timeout=self.timeout)
            r.raise_for_status()
            text = self._extract_text_from_generate(r.json())
            parsed = self._parse_three_blocks(text)
            return self._postprocess_fill_missing(parsed)
        except Exception as e:
            return {"whatsapp": f"❌ Error: {e}", "vcard": "", "json": {}}

    # ------------------------ helpers: Ollama payloads ------------------------

    @staticmethod
    def _to_ollama_chat_messages(msgs: list) -> list:
        """
        Convert a flat list of {role, content} into Ollama chat 'messages'.
        content can be str or {"type":"image","image":b64}.
        Ollama expects: {"role":"user","content":[{"type":"text","text":"..."}, {"type":"image","image":"..."}]}
        We merge consecutive items with the same role.
        """
        merged: List[Dict[str, Any]] = []
        for m in msgs:
            role = m["role"]
            content = m["content"]
            as_obj = (
                {"type": "text", "text": content}
                if isinstance(content, str)
                else {"type": "image", "image": content.get("image", "")}
            )
            if merged and merged[-1]["role"] == role:
                merged[-1]["content"].append(as_obj)
            else:
                merged.append({"role": role, "content": [as_obj]})
        return merged

    @staticmethod
    def _extract_text_from_chat(resp: Dict[str, Any]) -> str:
        """
        Ollama /api/chat returns {"message":{"content": "..."}}
        or sometimes {"messages":[...]} (aggregate). Handle both.
        """
        if not isinstance(resp, dict):
            return ""
        if "message" in resp and isinstance(resp["message"], dict):
            return resp["message"].get("content", "") or ""
        if "messages" in resp and isinstance(resp["messages"], list):
            parts = []
            for m in resp["messages"]:
                if m.get("role") == "assistant":
                    c = m.get("content")
                    if isinstance(c, str):
                        parts.append(c)
                    elif isinstance(c, list):
                        for it in c:
                            if isinstance(it, dict) and it.get("type") == "text":
                                parts.append(it.get("text", ""))
            return "\n".join(p for p in parts if p)
        return ""

    @staticmethod
    def _extract_text_from_generate(resp: Dict[str, Any]) -> str:
        """
        Ollama /api/generate returns {"response":"..."} (non-stream).
        """
        if isinstance(resp, dict):
            return resp.get("response", "") or resp.get("message", {}).get("content", "") or ""
        return ""

    # ------------------------ parsing & fallbacks ------------------------

    def _parse_three_blocks(self, raw: str) -> Dict[str, Any]:
        """
        Parse model text into whatsapp / vcard / json.
        Defensive against formatting variations.
        """
        text = (raw or "").strip()

        vcard = self._grab_vcard(text)
        json_obj = self._grab_json(text)
        whatsapp = self._grab_whatsapp(text, vcard_text=vcard, json_obj=json_obj)

        return {
            "whatsapp": (whatsapp or "").strip(),
            "vcard": (vcard or "").strip(),
            "json": json_obj if isinstance(json_obj, dict) else {},
        }

    @staticmethod
    def _grab_vcard(text: str) -> str:
        m = re.search(r"(BEGIN:VCARD[\s\S]+?END:VCARD)", text, re.IGNORECASE)
        return m.group(1).strip() if m else ""

    @staticmethod
    def _grab_json(text: str) -> Any:
        # Try codefence ```json ... ```
        fence = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
        candidates = []
        if fence:
            candidates.append(fence.group(1))
        # Try last {...} block (often the JSON section)
        braces = list(re.finditer(r"\{[\s\S]*\}", text))
        if braces:
            candidates.append(braces[-1].group(0))
        for c in candidates:
            try:
                return json.loads(c)
            except Exception:
                continue
        return {}

    @staticmethod
    def _strip_block(big: str, sub: str) -> str:
        if not sub:
            return big
        return big.replace(sub, " ")

    def _grab_whatsapp(self, text: str, vcard_text: str, json_obj: Any) -> str:
        # Remove vcard + json chunks to isolate the WhatsApp-like section
        t = self._strip_block(text, vcard_text)
        if isinstance(json_obj, dict) and json_obj:
            try:
                t = self._strip_block(t, json.dumps(json_obj))
            except Exception:
                pass

        m = re.search(
            r"(?:^|\n)\s*(?:📲|WhatsApp Message|1\.\s*WhatsApp)[\s\S]*?(?=(?:BEGIN:VCARD|```|\Z))",
            t,
            re.IGNORECASE,
        )
        if m:
            return m.group(0).strip()

        # Fallback: take first ~40 non-empty lines
        lines = [ln for ln in t.splitlines() if ln.strip()]
        return "\n".join(lines[:40]).strip()

    # ------------------------ post-process: fill missing sections ------------------------

    def _postprocess_fill_missing(self, data: Dict[str, Any]) -> Dict[str, Any]:
        j = data.get("json") or {}
        # If JSON is empty, attempt a quick regex scrape from the whole text we had (not available here).
        # As a pragmatic fallback, try to extract from WhatsApp block or vCard if present:
        if not j:
            j = self._json_from_text_fallback(data.get("whatsapp", "") + "\n" + data.get("vcard", ""))
            data["json"] = j

        # If vCard missing, build from JSON
        if not data.get("vcard"):
            vc = self._vcard_from_json(j)
            data["vcard"] = vc

        # If WhatsApp missing/too short, build from JSON
        if not data.get("whatsapp") or len(data["whatsapp"]) < 16:
            data["whatsapp"] = self._whatsapp_from_json(j)

        # Ensure JSON has arrays for phones/emails
        if isinstance(data.get("json"), dict):
            if "phones" in data["json"] and isinstance(data["json"]["phones"], str):
                data["json"]["phones"] = [data["json"]["phones"]]
            if "emails" in data["json"] and isinstance(data["json"]["emails"], str):
                data["json"]["emails"] = [data["json"]["emails"]]

        return data

    # ------------------------ builders & regex fallbacks ------------------------

    @staticmethod
    def _norm_phone(p: str) -> str:
        if not p:
            return ""
        s = re.sub(r"[^\d+]", "", p)
        if s and not s.startswith("+") and len(re.sub(r"\D", "", s)) in (10, 11, 12):
            # heuristics; do not guess country code aggressively
            return s
        return s

    def _json_from_text_fallback(self, text: str) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        if not text:
            return out

        # emails
        emails = re.findall(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text)
        if emails:
            out["emails"] = list(dict.fromkeys(emails))

        # phones
        phones = re.findall(r"(?:\+?\d[\d\s().-]{7,}\d)", text)
        phones = [self._norm_phone(p) for p in phones]
        phones = [p for p in phones if len(re.sub(r"\D", "", p)) >= 10]
        if phones:
            out["phones"] = list(dict.fromkeys(phones))

        # website
        mweb = re.search(r"(https?://[^\s)]+|www\.[^\s)]+)", text, re.I)
        if mweb:
            out["website"] = mweb.group(1)

        # name (very rough: FN in vCard or beginning of text before role/org)
        mfn = re.search(r"FN:(.+)", text)
        if mfn:
            out["name"] = mfn.group(1).strip()

        # org/title from vCard
        morg = re.search(r"\nORG:(.+)", text)
        if morg:
            out["company"] = morg.group(1).strip()
        mtitle = re.search(r"\nTITLE:(.+)", text)
        if mtitle:
            out["title"] = mtitle.group(1).strip()

        # simple social links
        for key, pat in [
            ("linkedin", r"(https?://(www\.)?linkedin\.com/[^\s)]+)"),
            ("twitter", r"(https?://(www\.)?twitter\.com/[^\s)]+|https?://x\.com/[^\s)]+)"),
            ("instagram", r"(https?://(www\.)?instagram\.com/[^\s)]+)"),
            ("facebook", r"(https?://(www\.)?facebook\.com/[^\s)]+)"),
            ("github", r"(https?://(www\.)?github\.com/[^\s)]+)"),
        ]:
            m = re.search(pat, text, re.I)
            if m:
                out[key] = m.group(1)

        return out

    def _vcard_from_json(self, j: Dict[str, Any]) -> str:
        if not j:
            return ""
        parts = ["BEGIN:VCARD", "VERSION:3.0"]
        if j.get("name"): parts.append(f"FN:{j['name']}")
        if j.get("company"): parts.append(f"ORG:{j['company']}")
        if j.get("title"): parts.append(f"TITLE:{j['title']}")
        for p in j.get("phones", []) or []:
            norm = self._norm_phone(p)
            if norm: parts.append(f"TEL;TYPE=CELL:{norm}")
        for e in j.get("emails", []) or []:
            parts.append(f"EMAIL;TYPE=INTERNET:{e}")
        if j.get("website"): parts.append(f"URL:{j['website']}")
        adr_line = j.get("address") or ""
        if adr_line:
            # Keep simple single-line ADR; vCard ADR is semicolon-separated; we use NOTE for free text as well
            parts.append(f"ADR:;;{adr_line};;;;")
        note_bits = []
        for k in ("city", "state", "postal_code", "country", "notes"):
            if j.get(k):
                note_bits.append(f"{k.title()}: {j[k]}")
        if note_bits:
            parts.append("NOTE:" + " | ".join(note_bits))
        # Social profiles
        for k in ("linkedin", "twitter", "instagram", "facebook", "github"):
            if j.get(k):
                parts.append(f"X-SOCIALPROFILE;type={k}:{j[k]}")
        for other in j.get("other_social", []) or []:
            parts.append(f"X-SOCIALPROFILE:{other}")
        parts.append("END:VCARD")
        return "\n".join(parts)

    def _whatsapp_from_json(self, j: Dict[str, Any]) -> str:
        if not j:
            return "Contact details extracted."
        name = j.get("name", "Contact")
        org = j.get("company")
        title = j.get("title")
        phones = j.get("phones") or []
        emails = j.get("emails") or []
        website = j.get("website")
        lines = [f"📇 *{name}*"]
        if title or org:
            lines.append(f"👔 {title+' · ' if title else ''}{org or ''}".strip())
        if phones:
            lines.append("📞 " + " | ".join(phones))
        if emails:
            lines.append("✉️ " + " | ".join(emails))
        if website:
            lines.append(f"🌐 {website}")
        for k in ("linkedin", "twitter", "instagram", "facebook", "github"):
            if j.get(k):
                lines.append(f"🔗 {k.title()}: {j[k]}")
        if j.get("address"):
            lines.append(f"📍 {j['address']}")
        return "\n".join(lines)
