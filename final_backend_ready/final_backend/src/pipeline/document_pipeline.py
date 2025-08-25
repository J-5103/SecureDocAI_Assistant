# src/pipeline/document_pipeline.py
import os
from typing import Optional

from src.components.rag_pipeline import RAGPipeline

UPLOAD_BASE = os.path.abspath(os.path.join(os.getcwd(), "uploaded_docs"))
VSTORE_BASE = os.path.abspath(os.path.join(os.getcwd(), "vectorstores"))


def _safe_filename(name: str) -> str:
    """Return just the filename (strip any client-side paths/backslashes)."""
    return os.path.basename((name or "").replace("\\", "/"))


def _resolve_pdf_path(chat_id: str, pdf_path: str, document_id: Optional[str]) -> str:
    """
    Resolve the absolute path to the uploaded PDF.
    Tries:
      1) given absolute path (if exists)
      2) uploaded_docs/<chat_id>/<document_id>
      3) uploaded_docs/<chat_id>/<stem>.pdf
    """
    # 1) absolute path already correct?
    if pdf_path and os.path.isabs(pdf_path) and os.path.exists(pdf_path):
        return pdf_path

    # 2) try exact filename under chat folder
    filename = _safe_filename(document_id or os.path.basename(pdf_path))
    cand1 = os.path.join(UPLOAD_BASE, chat_id, filename)

    # 3) fallback to <stem>.pdf
    stem, ext = os.path.splitext(filename)
    cand2 = os.path.join(UPLOAD_BASE, chat_id, f"{stem}.pdf")

    if os.path.exists(cand1):
        return os.path.abspath(cand1)
    if os.path.exists(cand2):
        return os.path.abspath(cand2)

    # last resort: return most likely (cand2 if ext != .pdf)
    return os.path.abspath(cand2 if ext.lower() != ".pdf" else cand1)


class DocumentPipeline:
    def __init__(self):
        self.rag = RAGPipeline()

    def run(self, pdf_path: str, document_id: Optional[str], chat_id: str):
        """
        Build/refresh the vectorstore for a single PDF inside
        vectorstores/<chat_id>/<doc-stem>.

        Args:
            pdf_path: path from upload handler (may be relative)
            document_id: original filename (used to derive stem)
            chat_id: per-chat namespace

        Returns:
            dict with vectorstore_path and document_key (stem)
        """
        if not chat_id:
            raise ValueError("chat_id is required")

        safe_name = _safe_filename(document_id or os.path.basename(pdf_path))
        stem, _ = os.path.splitext(safe_name)

        pdf_abs = _resolve_pdf_path(chat_id, pdf_path, safe_name)
        if not os.path.exists(pdf_abs):
            raise FileNotFoundError(f"Document not found at {pdf_abs}")

        vs_dir = os.path.join(VSTORE_BASE, chat_id, stem)
        os.makedirs(vs_dir, exist_ok=True)

        # Build (or rebuild) the vectorstore
        self.rag.create_vectorstore(pdf_path=pdf_abs, vector_store_path=vs_dir)

        return {
            "vectorstore_path": vs_dir,
            "document_key": stem,
            "pdf_path": pdf_abs,
        }
