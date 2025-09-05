# src/utils/vcard.py
from __future__ import annotations

from datetime import datetime
from typing import Iterable, Optional, Dict, List

FOLD_LIMIT = 75  # vCard 3.0: lines SHOULD be folded at 75 octets; we approximate by chars.


def _esc(s: Optional[str]) -> str:
    """
    Escape text values for vCard:
    - Backslash, comma, semicolon, newline
    """
    if not s:
        return ""
    return (
        s.replace("\\", "\\\\")
        .replace(";", r"\;")
        .replace(",", r"\,")
        .replace("\n", r"\n")
    )


def _fold(line: str) -> str:
    """
    Fold a single vCard line to the FOLD_LIMIT (approximate by char count).
    Continuation lines begin with a single space.
    """
    if len(line) <= FOLD_LIMIT:
        return line
    parts: List[str] = []
    s = line
    while len(s) > FOLD_LIMIT:
        parts.append(s[:FOLD_LIMIT])
        s = s[FOLD_LIMIT:]
    parts.append(s)
    # join with CRLF + space (folding whitespace)
    return "\r\n ".join(parts)


def _dedup_keep_order(items: Iterable[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for it in items or []:
        if not it:
            continue
        key = it.strip()
        if key and key not in seen:
            seen.add(key)
            out.append(key)
    return out


def _norm_url(u: str) -> str:
    u = (u or "").strip()
    if not u:
        return ""
    if u.startswith(("http://", "https://")):
        return u
    # Some cards will have bare domains (e.g., yesbank.in). Add http:// for compatibility.
    return f"http://{u}"


def make_vcard(
    full_name: str,
    last_name: str = "",
    first_name: str = "",
    org: Optional[str] = None,
    title: Optional[str] = None,
    phones: Iterable[str] = (),
    emails: Iterable[str] = (),
    websites: Iterable[str] = (),
    address: Optional[Dict] = None,  # {street, city, state, postal_code, country}
    notes: Optional[str] = None,
) -> str:
    """
    Build a vCard 3.0 string. UTF-8 safe.

    Params are identical to your original function, so existing callers continue to work.
    """
    lines: List[str] = []

    def add(prop: str, value: str):
        """Append a folded vCard property line if value is non-empty."""
        if value is None:
            return
        raw = f"{prop}:{value}"
        lines.append(_fold(raw))

    # Header
    lines.append("BEGIN:VCARD")
    lines.append("VERSION:3.0")

    # N: Last;First;Middle;Prefix;Suffix
    add("N", f"{_esc(last_name)};{_esc(first_name)};;;")

    # FN (Formatted Name) – required by most clients
    add("FN", _esc(full_name or f"{first_name} {last_name}".strip()))

    # ORG, TITLE
    if org:
        add("ORG", _esc(org))
    if title:
        add("TITLE", _esc(title))

    # TEL (dedup + preserve order); we keep TYPE=CELL by default (unknown labels from OCR can vary)
    for p in _dedup_keep_order(phones or []):
        add("TEL;TYPE=CELL", _esc(p))

    # EMAIL (dedup)
    for e in _dedup_keep_order(emails or []):
        add("EMAIL;TYPE=INTERNET", _esc(e))

    # URL (dedup + normalize)
    for u in _dedup_keep_order(websites or []):
        add("URL", _esc(_norm_url(u)))

    # ADR;TYPE=WORK: PO Box;Extended;Street;City;Region;Postal;Country
    if address:
        street = _esc(address.get("street"))
        city = _esc(address.get("city"))
        state = _esc(address.get("state"))
        postal = _esc(address.get("postal_code"))
        country = _esc(address.get("country"))
        add("ADR;TYPE=WORK", f";;{street};{city};{state};{postal};{country}")

    # NOTE
    if notes:
        add("NOTE", _esc(notes))

    # Revision timestamp
    add("REV", datetime.utcnow().strftime("%Y%m%dT%H%M%SZ"))

    # Footer
    lines.append("END:VCARD")

    # Join with CRLF as per spec
    return "\r\n".join(lines)
