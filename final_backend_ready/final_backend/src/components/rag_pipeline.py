# src/components/rag_pipeline.py
import os
import time
from typing import List, Tuple, Optional, Dict, Any
import io

# --- EDIT START: New imports for smart PDF extraction ---
import fitz  # PyMuPDF
import pytesseract
# --- EDIT END ---

import pandas as pd
import docx  # python-docx
from PIL import Image
import requests

# Optional torch (don't crash if it's not installed)
try:
    import torch  # type: ignore
    _TORCH_AVAILABLE = True
except Exception:
    torch = None  # type: ignore
    _TORCH_AVAILABLE = False

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

from src.utils.synonym_expander import SynonymExpander

DEFAULT_K = 8
CHUNK_SIZE = 700
CHUNK_OVERLAP = 100


class RAGPipeline:
    """
    RAG helper that:
      - builds FAISS vectorstores per (chat_id, doc_id)
      - retrieves top contexts from one or many docs
      - calls Ollama for grounded answers
    Vectorstore layout (new):
        vectorstores/<chat_id>/<doc_id>/index.faiss
    Legacy layout still works:
        vectorstores/<chat_id>/<file_basename>/index.faiss
    """

    def __init__(
        self,
        vector_store_path: str = "vectorstores",
        ollama_url: str = "http://192.168.0.88:11434/api/generate",
        model_name: str = "qwen2.5vl-gpu:latest",
    ):
        device = "cuda" if (_TORCH_AVAILABLE and torch.cuda.is_available()) else "cpu"  # type: ignore[attr-defined]
        print(f"Initializing RAGPipeline with device: {device}")

        self.embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": device},
        )
        self.vector_store_path = vector_store_path
        self.ollama_url = ollama_url
        self.model_name = model_name

    # -------------------- path utils --------------------

    def _chat_folder(self, chat_id: str) -> str:
        return os.path.join(self.vector_store_path, chat_id)

    def _doc_folder(self, chat_id: str, doc_name_or_id: str) -> str:
        """
        Accepts either a modern doc_id ('file-slug-1a2b3c4d') or a legacy file name.
        We strip extension if present so both 'abc.pdf' and 'abc' map to same folder.
        """
        base = os.path.splitext(doc_name_or_id)[0]
        return os.path.join(self._chat_folder(chat_id), base)

    def _load_vs(self, folder_path: str):
        index_file = os.path.join(folder_path, "index.faiss")
        if not os.path.exists(index_file):
            return None
        return FAISS.load_local(
            folder_path,
            self.embedding_model,
            allow_dangerous_deserialization=True,
        )

    # -------------------- small helpers --------------------

    def sanitize_text(self, text: str) -> str:
        return text.encode("utf-8", "replace").decode("utf-8")

    def detect_question_type(self, question: str) -> str:
        q = (question or "").lower()
        if any(k in q for k in ["cost", "price", "budget", "amount", "charges"]):
            return "cost"
        if any(k in q for k in ["summary", "tell me about", "describe", "overview"]):
            return "summary"
        return "default"

    # -------------------- extraction --------------------

    def extract_text_from_doc(self, file_path: str) -> List[Document]:
        """Extract text from various document types with progress logging."""
        filename = os.path.basename(file_path).replace(os.path.splitext(file_path)[1], "")
        documents: List[Document] = []
        try:
            lower = file_path.lower()

            # --- EDIT START: Replaced old PDF logic with new smart extraction ---
            if lower.endswith(".pdf"):
                print(f"Starting Smart PDF text extraction from: {filename}")
                doc = fitz.open(file_path)
                for i, page in enumerate(doc):
                    page_text = ""
                    # 1. Text-First Approach
                    text = page.get_text().strip()
                    if text:
                        page_text = text
                        print(f"Extracted text directly from page {i + 1}")
                    # 2. OCR Fallback for image-based pages
                    else:
                        print(f"No text found on page {i + 1}, trying OCR...")
                        pix = page.get_pixmap(dpi=300)
                        img_data = pix.tobytes("png")
                        image = Image.open(io.BytesIO(img_data))
                        
                        # Tesseract OCR to extract text
                        ocr_text = pytesseract.image_to_string(image, lang='eng')
                        if ocr_text.strip():
                            page_text = ocr_text
                            print(f"Successfully extracted text via OCR from page {i + 1}")
                        else:
                            print(f"OCR could not find any text on page {i + 1}")
                    
                    if page_text.strip():
                        safe_text = self.sanitize_text(page_text)
                        documents.append(
                            Document(
                                page_content=safe_text,
                                metadata={"source": filename, "page": i + 1},
                            )
                        )
                doc.close()
            # --- EDIT END ---

            elif lower.endswith((".docx", ".doc")):
                print(f"Starting DOCX text extraction from: {filename}")
                d = docx.Document(file_path)
                full_text = [para.text.strip() for para in d.paragraphs if para.text.strip()]
                if full_text:
                    safe_text = self.sanitize_text("\n".join(full_text))
                    documents.append(Document(page_content=safe_text, metadata={"source": filename}))
                print(f"Extracted {len(full_text)} paragraphs")
            elif lower.endswith((".xlsx", ".xls", ".csv")):
                print(f"Starting Excel/CSV text extraction from: {filename}")
                if lower.endswith((".xlsx", ".xls")):
                    df = pd.read_excel(file_path)
                else:
                    df = pd.read_csv(file_path)
                text = df.to_string(index=False)
                if text.strip():
                    safe_text = self.sanitize_text(text)
                    documents.append(Document(page_content=safe_text, metadata={"source": filename}))
                print(f"Extracted {len(df)} rows")
            elif lower.endswith((".png", ".jpg", ".jpeg")):
                print(f"Starting image text extraction from: {filename}")
                # You can add OCR logic here too if needed for standalone images
                image = Image.open(file_path)
                text = pytesseract.image_to_string(image, lang='eng')
                if text.strip():
                    safe_text = self.sanitize_text(text)
                    documents.append(Document(page_content=safe_text, metadata={"source": filename}))
                print("Image extraction via OCR completed")
            else:
                raise ValueError(f"Unsupported file type: {file_path}")
        except Exception as e:
            print(f"❌ Error reading {filename}: {e}")
            raise
        print(f"Text extraction complete for {filename}. Total documents: {len(documents)}")
        return documents

    def _infer_chat_id_from_pdf_path(self, file_path: Optional[str]) -> Optional[str]:
        if not file_path:
            return None
        parts = os.path.normpath(file_path).split(os.sep)
        try:
            up_idx = parts.index("uploaded_docs")
            return parts[up_idx + 1] if up_idx + 1 < len(parts) else None
        except ValueError:
            return None

    # -------------------- indexing --------------------

    def create_vectorstore(
        self,
        file_path: Optional[str] = None,
        combined_text: Optional[str] = None,
        vector_store_path: Optional[str] = None,
        metadata: Optional[dict] = None,
    ):
        """
        Create a FAISS vectorstore at vector_store_path.
        If not provided, we infer <vectorstores>/<chat_id>/<filename-base>.
        """
        filename = os.path.basename(file_path or "manual_input").replace(
            os.path.splitext(file_path or "")[1], ""
        )
        try:
            if not combined_text and not file_path:
                raise ValueError("Either 'file_path' or 'combined_text' must be provided.")

            inferred_chat = self._infer_chat_id_from_pdf_path(file_path)
            default_folder = (
                os.path.join(self.vector_store_path, inferred_chat, filename)
                if inferred_chat
                else os.path.join(self.vector_store_path, filename)
            )
            vectorstore_folder = vector_store_path or default_folder
            os.makedirs(vectorstore_folder, exist_ok=True)

            print(f"Starting vectorstore creation for: {filename} at {vectorstore_folder}")
            start_time = time.time()

            if combined_text:
                safe_text = self.sanitize_text(combined_text)
                docs = [Document(page_content=safe_text, metadata={"source": filename})]
            else:
                docs = self.extract_text_from_doc(file_path)  # type: ignore[arg-type]

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
            )
            chunks = splitter.split_documents(docs)
            print(f"Text splitting complete. Total chunks: {len(chunks)}")

            # Batch embedding
            batch_size = 100
            all_embeddings = []
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i : i + batch_size]
                embeddings = self.embedding_model.embed_documents([c.page_content for c in batch])
                all_embeddings.extend(embeddings)
                print(f"Embedded batch {i // batch_size + 1} of {(len(chunks) + batch_size - 1) // batch_size}")
                time.sleep(0.05)  # slight yield; remove in production

            texts = [c.page_content for c in chunks]
            text_embeddings = list(zip(texts, all_embeddings))
            metadatas = [{"source": filename, "page": c.metadata.get("page", 1)} for c in chunks]

            vectorstore = FAISS.from_embeddings(
                text_embeddings=text_embeddings,
                embedding=self.embedding_model,
                metadatas=metadatas,
            )
            vectorstore.save_local(vectorstore_folder)

            elapsed = time.time() - start_time
            print(f"✅ Vectorstore created at: {vectorstore_folder} in {elapsed:.2f} seconds")

            return {
                "status": "ready",
                "document_id": os.path.splitext(filename)[0],
                "vectorstore_path": vectorstore_folder,
            }
        except Exception as e:
            print(f"❌ create_vectorstore failed for {filename}: {e}")
            raise

    # Convenience API (optional, used by some variants of main.py)
    def index_document(self, file_path: str, namespace: str, out_dir: str) -> Dict[str, Any]:
        """
        Kept for compatibility with alternative wiring.
        We ignore 'namespace' here since our layout already scopes by chat/doc.
        """
        return self.create_vectorstore(file_path=file_path, vector_store_path=out_dir)

    def search(self, query: str, namespace: str, index_dir: str, k: int = DEFAULT_K) -> List[Dict[str, Any]]:
        """
        Convenience search that returns a list of {text, score, source, page}.
        """
        vs = self._load_vs(index_dir)
        if not vs:
            return []
        docs_scores = vs.similarity_search_with_score(query, k=k)
        out = []
        for d, s in docs_scores:
            out.append({
                "text": d.page_content,
                "score": float(s),
                "source": d.metadata.get("source"),
                "page": d.metadata.get("page"),
            })
        return out

    # -------------------- context retrieval --------------------

    def _expanded_query(self, question: str) -> str:
        expander = SynonymExpander()
        return expander.find_similar_words(question)

    def _topk_from_vs(self, vs, query: str, k: int) -> List[Tuple[Document, float]]:
        docs_with_scores = vs.similarity_search_with_score(query, k=max(1, k))
        # No strict threshold; we’ll sort later across docs
        return docs_with_scores

    def get_context_from_single_doc(
        self,
        question: str,
        chat_id: str,
        document_id: str,
        top_k: int = DEFAULT_K,
    ) -> Tuple[str, List[str], str]:
        question_type = self.detect_question_type(question)
        expanded_query = self._expanded_query(question)

        folder_path = self._doc_folder(chat_id, document_id)
        vs = self._load_vs(folder_path)
        if not vs:
            print(f"❌ Vectorstore not found for: {folder_path}")
            return "", [], question_type

        docs_with_scores = self._topk_from_vs(vs, expanded_query, top_k)
        if not docs_with_scores:
            return "", [], question_type

        # sort best-first (lower score is better for FAISS distances)
        docs_with_scores.sort(key=lambda x: x[1])
        docs_with_scores = docs_with_scores[:top_k]

        numbered = []
        for idx, (doc, _) in enumerate(docs_with_scores, start=1):
            doc_text = self.sanitize_text(doc.page_content.strip())
            numbered.append(f"{idx}. {doc_text}")

        ctx = "\n".join(numbered)
        return ctx, [os.path.basename(folder_path)], question_type

    def get_context_from_multiple_docs(
        self,
        question: str,
        chat_id: str,
        document_names: Optional[List[str]] = None,
        top_k: int = DEFAULT_K,
    ) -> Tuple[str, List[str], str]:
        """
        Retrieve from multiple docs and keep the BEST 'top_k' chunks GLOBALLY.
        'document_names' are usually new doc_ids. If None, we search across all docs in the chat.
        """
        print("🔍 Loading vectorstores across multiple docs...")
        question_type = self.detect_question_type(question)
        base_folder = self._chat_folder(chat_id)

        if not os.path.exists(base_folder):
            print(f"❌ Chat folder not found: {base_folder}")
            return "", [], question_type

        expanded_query = self._expanded_query(question)

        # Build list of folders to query
        target_folders: List[str] = []
        if document_names:
            for name in document_names:
                target_folders.append(self._doc_folder(chat_id, name))
        else:
            for folder_name in os.listdir(base_folder):
                target_folders.append(os.path.join(base_folder, folder_name))

        # Collect results from each doc, then rank globally
        heap: List[Tuple[float, Document, str]] = []  # (score, doc, folder_path)
        used_folders: set = set()

        # We over-fetch per doc (2x) then trim globally
        per_doc_k = max(1, min(top_k * 2, 20))

        for folder_path in target_folders:
            vs = self._load_vs(folder_path)
            if not vs:
                continue
            try:
                pairs = self._topk_from_vs(vs, expanded_query, per_doc_k)
                for doc, score in pairs:
                    heap.append((float(score), doc, folder_path))
                    used_folders.add(folder_path)
            except Exception as e:
                print(f"❌ Failed search in {folder_path}: {e}")

        if not heap:
            return "", [], question_type

        # global sort by score and take top_k
        heap.sort(key=lambda t: t[0])
        best = heap[:top_k]

        numbered: List[str] = []
        used_docs: List[str] = []
        for idx, (score, doc, fldr) in enumerate(best, start=1):
            doc_text = self.sanitize_text(doc.page_content.strip())
            numbered.append(f"{idx}. {doc_text}")
            used_docs.append(os.path.basename(fldr))

        # dedupe used_docs but keep order
        seen = set()
        used_docs = [d for d in used_docs if not (d in seen or seen.add(d))]

        merged_context = "\n".join(numbered)
        return merged_context, used_docs, question_type

    def get_context(
        self,
        question: str,
        chat_id: str,
        document_id: Optional[str] = None,
        combine_docs: Optional[List[str]] = None,
        k: int = DEFAULT_K,
    ) -> Tuple[str, List[str]]:
        if document_id:
            ctx, used, _ = self.get_context_from_single_doc(
                question, chat_id, document_id, top_k=k
            )
            return ctx, used
        ctx, used, _ = self.get_context_from_multiple_docs(
            question, chat_id, document_names=combine_docs, top_k=k
        )
        return ctx, used

    # -------------------- answer --------------------

    def answer_question(
        self,
        question: str,
        chat_id: str,
        document_id: Optional[str] = None,
        combine_docs: Optional[List[str]] = None,
    ) -> str:
        try:
            context, used_docs = self.get_context(
                question=question,
                chat_id=chat_id,
                document_id=document_id,
                combine_docs=combine_docs,
                k=DEFAULT_K,
            )

            if not context:
                raise Exception("No relevant context retrieved for the selected documents.")

            prompt = (
                "You are a helpful assistant. Answer strictly from the context. "
                "If the answer is not present, say you don't know.\n\n"
                f"Context from documents {used_docs}:\n{context}\n\n"
                f"Question: {question}\nAnswer:"
            )

            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "keep_alive": "10m",
                "options": {
                    "temperature": 0.2,
                    "num_predict": 400,
                    "num_ctx": 3072,
                },
            }
            response = requests.post(self.ollama_url, json=payload, timeout=2000)
            response.raise_for_status()

            answer = response.json().get("response", "") or ""
            if not answer.strip():
                raise Exception("Model returned empty response.")
            return answer.strip()

        except requests.RequestException as e:
            print(f"❌ Error calling Ollama API: {e}")
            raise
        except Exception as e:
            print(f"❌ Error in answer_question: {e}")
            raise