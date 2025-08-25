# src/components/rag_pipeline.py
import os
import time
from typing import List, Tuple, Optional

import pdfplumber
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
    def __init__(
        self,
        vector_store_path: str = "vectorstores",  # plural to match main.py
        ollama_url: str = "http://192.168.0.88:11434/api/generate",
        model_name: str = "llama3:8b",
    ):
        # Choose device only if torch is present
        device = "cuda" if (_TORCH_AVAILABLE and torch.cuda.is_available()) else "cpu"  # type: ignore[attr-defined]
        print(f"Initializing RAGPipeline with device: {device}")

        self.embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": device},
        )
        self.vector_store_path = vector_store_path
        self.ollama_url = ollama_url
        self.model_name = model_name

    # -------------------- utils --------------------

    def _chat_folder(self, chat_id: str) -> str:
        return os.path.join(self.vector_store_path, chat_id)

    def _doc_folder(self, chat_id: str, doc_name_or_file: str) -> str:
        base = os.path.splitext(doc_name_or_file)[0]
        return os.path.join(self._chat_folder(chat_id), base)

    def _load_vs(self, folder_path: str):
        index_file = os.path.join(folder_path, "index.faiss")
        if not os.path.exists(index_file):
            return None
        # langchain_community 0.2+: load_local(path, embeddings, allow_dangerous_deserialization=...)
        return FAISS.load_local(
            folder_path,
            self.embedding_model,
            allow_dangerous_deserialization=True,
        )

    def sanitize_text(self, text: str) -> str:
        return text.encode("utf-8", "replace").decode("utf-8")

    def detect_question_type(self, question: str) -> str:
        q = (question or "").lower()
        if any(k in q for k in ["cost", "price", "budget", "amount", "charges"]):
            return "cost"
        if any(k in q for k in ["summary", "tell me about", "describe", "overview"]):
            return "summary"
        return "default"

    # -------------------- document extraction --------------------

    def extract_text_from_doc(self, file_path: str) -> List[Document]:
        """Extract text from various document types with progress logging."""
        filename = os.path.basename(file_path).replace(os.path.splitext(file_path)[1], "")
        documents: List[Document] = []
        try:
            lower = file_path.lower()
            if lower.endswith(".pdf"):
                print(f"Starting PDF text extraction from: {filename}")
                with pdfplumber.open(file_path) as pdf:
                    for i, page in enumerate(pdf.pages):
                        text = page.extract_text() or ""
                        tables = page.extract_tables()
                        for table in tables or []:
                            table_text = "\n".join(
                                [
                                    " | ".join((cell.strip() if cell else "") for cell in row)
                                    for row in table
                                    if any(row)
                                ]
                            )
                            if table_text.strip():
                                text += f"\n[Extracted Table - Page {i + 1}]\n{table_text}"
                        if text.strip():
                            safe_text = self.sanitize_text(text)
                            documents.append(
                                Document(
                                    page_content=safe_text,
                                    metadata={"source": filename, "page": i + 1},
                                )
                            )
                        print(f"Extracted text from page {i + 1} of {len(pdf.pages)}")
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
                # Placeholder (OCR not implemented here)
                with Image.open(file_path):
                    text = "Image content (OCR required)"
                    if text.strip():
                        safe_text = self.sanitize_text(text)
                        documents.append(Document(page_content=safe_text, metadata={"source": filename}))
                print("Image extraction completed (OCR placeholder)")
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

    def create_vectorstore(
        self,
        file_path: Optional[str] = None,
        combined_text: Optional[str] = None,
        vector_store_path: Optional[str] = None,
        metadata: Optional[dict] = None,
    ):
        """Create vectorstore with batch processing and progress logging."""
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

            # Create FAISS vectorstore with (text, embedding) pairs
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

    # -------------------- context retrieval --------------------

    def get_context_from_single_doc(
        self,
        question: str,
        chat_id: str,
        document_id: str,
        top_k: int = DEFAULT_K,
    ) -> Tuple[str, List[str], str]:
        question_type = self.detect_question_type(question)
        expander = SynonymExpander()
        expanded_query = expander.find_similar_words(question)

        folder_path = self._doc_folder(chat_id, document_id)
        vs = self._load_vs(folder_path)
        if not vs:
            print(f"❌ Vectorstore not found for: {folder_path}")
            return "", [], question_type

        docs_with_scores = vs.similarity_search_with_score(expanded_query, k=top_k)
        top_docs = [doc for doc, score in docs_with_scores if score < 1.0] or [
            doc for doc, score in docs_with_scores
        ]

        if not top_docs:
            return "", [], question_type

        numbered = []
        for idx, doc in enumerate(top_docs, start=1):
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
        print("🔍 Loading vectorstores...")
        question_type = self.detect_question_type(question)
        base_folder = self._chat_folder(chat_id)

        if not os.path.exists(base_folder):
            print(f"❌ Chat folder not found: {base_folder}")
            return "", [], question_type

        expander = SynonymExpander()
        expanded_query = expander.find_similar_words(question)

        used_docs: List[str] = []
        contexts: List[str] = []

        target_folders: List[str] = []
        if document_names:
            for name in document_names:
                target_folders.append(self._doc_folder(chat_id, name))
        else:
            for folder_name in os.listdir(base_folder):
                target_folders.append(os.path.join(base_folder, folder_name))

        for folder_path in target_folders:
            vs = self._load_vs(folder_path)
            if not vs:
                continue
            try:
                docs_with_scores = vs.similarity_search_with_score(expanded_query, k=top_k)
                top_docs = [doc for doc, score in docs_with_scores if score < 1.0] or [
                    doc for doc, score in docs_with_scores
                ]
                if not top_docs:
                    continue

                numbered = []
                for idx, doc in enumerate(top_docs, start=1):
                    doc_text = self.sanitize_text(doc.page_content.strip())
                    numbered.append(f"{idx}. {doc_text}")
                contexts.append("\n".join(numbered))
                used_docs.append(os.path.basename(folder_path))
            except Exception as e:
                print(f"❌ Failed loading {folder_path}: {e}")

        merged_context = "\n\n".join(contexts)
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
            response = requests.post(self.ollama_url, json=payload, timeout=120)
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
