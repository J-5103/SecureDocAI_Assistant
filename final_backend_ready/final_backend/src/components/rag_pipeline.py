import os
import pdfplumber
from typing import List, Tuple, Optional

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

from src.utils.synonym_expander import SynonymExpander
import requests


DEFAULT_K = 8
CHUNK_SIZE = 700
CHUNK_OVERLAP = 100


class RAGPipeline:
    def __init__(
        self,
        vector_store_path: str = "vectorstores",
        ollama_url: str = "http://192.168.0.88:11434/api/generate",
        model_name: str = "llama3:8b",  # faster quantized default
    ):
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.vector_store_path = vector_store_path
        self.ollama_url = ollama_url
        self.model_name = model_name

    # -------------------- utils --------------------

    def _chat_folder(self, chat_id: str) -> str:
        return os.path.join(self.vector_store_path, chat_id)

    def _doc_folder(self, chat_id: str, doc_name_or_file: str) -> str:
        # accept "MyDoc.pdf" or "MyDoc"
        base = os.path.splitext(doc_name_or_file)[0]
        return os.path.join(self._chat_folder(chat_id), base)

    def _load_vs(self, folder_path: str):
        index_file = os.path.join(folder_path, "index.faiss")
        if not os.path.exists(index_file):
            return None
        return FAISS.load_local(
            folder_path, self.embedding_model, allow_dangerous_deserialization=True
        )

    def sanitize_text(self, text: str) -> str:
        return text.encode("utf-8", "replace").decode("utf-8")

    def detect_question_type(self, question: str) -> str:
        q = question.lower()
        if any(k in q for k in ["cost", "price", "budget", "amount", "charges"]):
            return "cost"
        if any(k in q for k in ["summary", "tell me about", "describe", "overview"]):
            return "summary"
        return "default"

    # -------------------- building VS --------------------

    def extract_pdf_with_tables(self, pdf_path: str) -> List[Document]:
        documents = []
        filename = os.path.basename(pdf_path).replace(".pdf", "")
        try:
            with pdfplumber.open(pdf_path) as pdf:
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
        except Exception as e:
            print(f"❌ Error reading PDF: {e}")
            raise
        return documents

    def _infer_chat_id_from_pdf_path(self, pdf_path: Optional[str]) -> Optional[str]:
        """
        Try to infer chat_id from a path like: uploaded_docs/<chat_id>/<file>.pdf
        """
        if not pdf_path:
            return None
        parts = os.path.normpath(pdf_path).split(os.sep)
        try:
            up_idx = parts.index("uploaded_docs")
            return parts[up_idx + 1] if up_idx + 1 < len(parts) else None
        except ValueError:
            return None

    def create_vectorstore(
        self,
        pdf_path: Optional[str] = None,
        combined_text: Optional[str] = None,
        vector_store_path: Optional[str] = None,
        metadata: Optional[dict] = None,
    ):
        try:
            if not combined_text and not pdf_path:
                raise ValueError("Either 'pdf_path' or 'combined_text' must be provided.")

            filename = os.path.basename(pdf_path or "manual_input").replace(".pdf", "")
            # If a VS path isn't provided, infer chat_id from uploaded_docs/<chat_id>/...
            inferred_chat = self._infer_chat_id_from_pdf_path(pdf_path)
            default_folder = (
                os.path.join(self.vector_store_path, inferred_chat, filename)
                if inferred_chat
                else os.path.join(self.vector_store_path, filename)
            )
            vectorstore_folder = vector_store_path or default_folder
            os.makedirs(vectorstore_folder, exist_ok=True)

            if combined_text:
                safe_text = self.sanitize_text(combined_text)
                docs = [Document(page_content=safe_text, metadata={"source": filename})]
            else:
                docs = self.extract_pdf_with_tables(pdf_path)

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
            )
            chunks = splitter.split_documents(docs)

            vectorstore = FAISS.from_documents(chunks, self.embedding_model)
            vectorstore.save_local(vectorstore_folder)
            print(f"✅ Vectorstore created at: {vectorstore_folder}")

        except Exception as e:
            print(f"❌ create_vectorstore failed: {e}")
            raise

    # -------------------- context retrieval --------------------

    def get_context_from_single_doc(
        self,
        question: str,
        chat_id: str,
        document_id: str,
        top_k: int = DEFAULT_K,
    ) -> Tuple[str, List[str], str]:
        """Return (context, used_docs, question_type) for a single document."""
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
        """Return (context, used_docs, question_type) from multiple (or all) docs in a chat."""
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

        # If list provided -> restrict to those docs; else use all folders
        target_folders = []
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

    # ✅ Backward‑compatible wrapper expected by QuestionAnswerPipeline
    def get_context(
        self,
        question: str,
        chat_id: str,
        document_id: Optional[str] = None,
        combine_docs: Optional[List[str]] = None,
        k: int = DEFAULT_K,
    ) -> Tuple[str, List[str]]:
        """
        Returns (context, used_docs). Kept minimal so callers that don't need
        question_type won't break.
        """
        if document_id:
            ctx, used, _ = self.get_context_from_single_doc(
                question, chat_id, document_id, top_k=k
            )
            return ctx, used
        ctx, used, _ = self.get_context_from_multiple_docs(
            question, chat_id, document_names=combine_docs, top_k=k
        )
        return ctx, used

    # -------------------- LLM call --------------------

    def answer_question(
        self,
        question: str,
        chat_id: str,
        document_id: Optional[str] = None,
        combine_docs: Optional[List[str]] = None,
    ) -> str:
        """
        Unified answer function that picks single vs multi based on args.
        """
        try:
            # get context
            context, used_docs = self.get_context(
                question=question,
                chat_id=chat_id,
                document_id=document_id,
                combine_docs=combine_docs,
                k=DEFAULT_K,
            )

            if not context:
                raise Exception("No relevant context retrieved for the selected documents.")

            qtype = self.detect_question_type(question)
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
