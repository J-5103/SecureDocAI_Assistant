# src/pipeline/question_answer_pipeline.py
import os
import requests
from typing import List, Optional

from langchain_community.vectorstores import FAISS

from src.components.rag_pipeline import RAGPipeline
from src.utils.synonym_expander import SynonymExpander

# Directories (match main.py usage)
UPLOAD_BASE = os.path.abspath(os.path.join(os.getcwd(), "uploaded_docs"))
VSTORE_BASE = os.path.abspath(os.path.join(os.getcwd(), "vectorstores"))


def _safe_filename(name: str) -> str:
    """Strip any client path/backslashes and return just the filename."""
    return os.path.basename((name or "").replace("\\", "/"))


def _resolve_pdf_path(chat_id: str, document_id: str) -> str:
    """
    Build a safe absolute path to the uploaded PDF. We try both:
      1) the exact filename provided
      2) the same stem forced to `.pdf` (guards against ext/case mismatches)
    """
    fname = _safe_filename(document_id)
    cand1 = os.path.join(UPLOAD_BASE, chat_id, fname)

    stem, ext = os.path.splitext(fname)
    cand2 = os.path.join(UPLOAD_BASE, chat_id, f"{stem}.pdf")

    if os.path.exists(cand1):
        return cand1
    if os.path.exists(cand2):
        return cand2
    # Return the "expected" path to help error messages upstream
    return cand2 if ext.lower() != ".pdf" else cand1


def _is_valid_pdf(path: str) -> bool:
    """
    Quick sanity check: header must be %PDF- and file should end with %%EOF (within last 2KB).
    Prevents trying to index Excel/CSV/images as PDFs.
    """
    try:
        with open(path, "rb") as f:
            header = f.read(5)
        if header != b"%PDF-":
            return False
        size = os.path.getsize(path)
        tail_len = min(2048, size)
        if tail_len <= 0:
            return False
        with open(path, "rb") as f:
            f.seek(-tail_len, os.SEEK_END)
            tail = f.read()
        return b"%%EOF" in tail
    except Exception:
        return False


class QuestionAnswerPipeline:
    def __init__(self):
        # Keep a single RAG pipeline instance so we share embeddings + Ollama config
        self.rag = RAGPipeline()

    def sanitize_text(self, text: str) -> str:
        return text.encode("utf-8", "replace").decode("utf-8")

    def run(
        self,
        question,
        chat_id,
        document_id: Optional[str] = None,
        combine_docs: Optional[List[str]] = None,
        return_sources: bool = False,
    ):
        try:
            # --------- validate & normalize question ----------
            if not question:
                raise ValueError("No question provided.")
            if isinstance(question, list):
                question = " ".join(str(q) for q in question)
            elif not isinstance(question, str):
                raise TypeError("Invalid input: 'question' must be a string.")
            question = question.strip()
            if not question:
                raise ValueError("Question is empty after stripping.")

            # --------- expand query for retrieval ----------
            expander = SynonymExpander()
            expanded_question = expander.find_similar_words(question)

            ql = question.lower()
            is_cost_question = any(
                kw in ql for kw in ["cost", "price", "charges", "amount", "budget"]
            )

            # =================== COMBINE MODE ===================
            if combine_docs and len(combine_docs) > 0:
                print("🔗 Combine Mode: Using selected documents only...")
                context, used_docs = self.rag.get_context(
                    question=expanded_question,
                    chat_id=chat_id,
                    document_id=None,
                    combine_docs=combine_docs,
                    k=8,
                )
                if not context.strip():
                    raise RuntimeError("No relevant content found in the selected documents.")

                combined_contexts = self.sanitize_text(context)
                prompt = (
                    self.build_cost_compare_prompt(question, combined_contexts)
                    if is_cost_question
                    else self.build_multi_doc_compare_prompt(question, combined_contexts)
                )
                print(f"📄 Context from {len(used_docs)} selected documents being sent to model.")
                answer = self.call_ollama(prompt)
                return (answer, used_docs) if return_sources else answer

            # =================== SINGLE DOCUMENT MODE ===================
            if document_id:
                safe_name = _safe_filename(str(document_id))
                stem = os.path.splitext(safe_name)[0]

                pdf_path = _resolve_pdf_path(chat_id, safe_name)
                if not os.path.exists(pdf_path):
                    raise FileNotFoundError(f"Document not found at {pdf_path}")

                # ❗ Guard: only real PDFs are allowed in the QA pipeline
                if not _is_valid_pdf(pdf_path):
                    return (
                        "❌ The selected file isn't a valid PDF (or it's corrupted). "
                        "If this is Excel/CSV data, use the Excel plot endpoints "
                        "(/api/excel/upload and /api/excel/plot) to generate charts."
                    )

                vs_folder = os.path.join(VSTORE_BASE, chat_id, stem)
                index_file = os.path.join(vs_folder, "index.faiss")

                # Check if vectorstore exists; if not, inform user to re-upload or wait
                if not os.path.exists(index_file):
                    return (
                        f"❌ Vectorstore for {stem} is missing or not yet processed. "
                        "Please re-upload the document or wait for background processing to complete."
                    )

                # fetch context for the selected single doc
                context, used_docs = self.rag.get_context(
                    question=expanded_question,
                    chat_id=chat_id,
                    document_id=stem,  # pass folder/doc key (stem)
                    combine_docs=None,
                    k=8,
                )
                if not context or not context.strip():
                    raise RuntimeError("No relevant content found in the document.")

                sanitized_context = self.sanitize_text(context)
                prompt = self.build_single_doc_prompt(question, sanitized_context)
                print(f"📄 Sending prompt to model from single document: {stem}")
                answer = self.call_ollama(prompt)

                if return_sources:
                    # langchain_community 0.2+: load_local(path, embeddings, allow_dangerous_deserialization=...)
                    vectorstore = FAISS.load_local(
                        vs_folder,
                        self.rag.embedding_model,
                        allow_dangerous_deserialization=True,
                    )
                    docs = vectorstore.similarity_search(expanded_question, k=5)
                    return answer, docs

                return answer

            # =================== NO SELECTION ===================
            print("ℹ️ No document selected; using all chat documents.")
            context, used_docs = self.rag.get_context(
                question=expanded_question,
                chat_id=chat_id,
                document_id=None,
                combine_docs=[],  # [] => "all docs"
                k=8,
            )
            if not context.strip():
                raise RuntimeError("No relevant content found across chat documents.")

            combined_contexts = self.sanitize_text(context)
            prompt = (
                self.build_cost_compare_prompt(question, combined_contexts)
                if is_cost_question
                else self.build_multi_doc_compare_prompt(question, combined_contexts)
            )
            answer = self.call_ollama(prompt)
            return (answer, used_docs) if return_sources else answer

        except Exception as e:
            print(f"❌ Pipeline Error: {str(e)}")
            return (f"❌ Backend Error: {str(e)}", []) if return_sources else f"❌ Backend Error: {str(e)}"

    # ---------------- prompts ----------------

    def build_single_doc_prompt(self, question, context):
        return f"""
You are a highly knowledgeable assistant. Answer ONLY from the provided document content.

Document Content:
-----------------
{context}

User Question:
--------------
{question}

Instructions:
-------------
- Do NOT include filenames or metadata.
- Focus strictly on the content for your answer.
- Use concise bullet points or a short structured paragraph.
- If the answer isn't present, say "Not found in the provided document."
"""

    def build_multi_doc_compare_prompt(self, question, combined_contexts):
        return f"""
You are comparing multiple PDFs. Produce a concise, decision-ready answer in this exact format:

# Executive Summary
• 2–4 bullet points with the key conclusions and critical differences.

# Comparison Table
Create a markdown table with columns:
| Criteria | Doc Name | Evidence (quote or paraphrase) | Page/Section |
Only include the most important 6–10 criteria.

# Key Differences / Conflicts
• Bullet points listing where documents disagree and why.

# Gaps / Missing Info
• What’s unclear or missing across the PDFs.

# Recommendation
• 1–2 sentences on the best option or next steps, considering trade-offs.

Rules:
- Use ONLY the provided content.
- Cite doc name and page/section where possible.
- Keep it under ~300–500 words.

Context from the PDFs:
----------------------
{combined_contexts}

User Question:
--------------
{question}
"""

    def build_cost_compare_prompt(self, question, combined_contexts):
        return f"""
You are a finance assistant extracting and comparing costs from multiple PDFs.

Output format:
# Executive Summary
• 2–4 bullets on total/relative costs and biggest drivers.

# Comparison Table
| Cost Item | Doc Name | Amount | Evidence (quote/paraphrase) | Page/Section |

# Notes & Assumptions
• Any normalization, ranges, or missing values.

# Recommendation
• Best option and why, including trade-offs.

Rules:
- Extract only explicit costs (e.g., "$20,000", "₹50,000", "Rs. 1L").
- If a document lacks costs, say so.
- Stay within ~300–500 words.

Context from the PDFs:
----------------------
{combined_contexts}

User Question:
--------------
{question}
"""

    # ---------------- llm call ----------------

    def call_ollama(self, prompt: str) -> str:
        """
        Calls Ollama with the same base URL/model configured in RAGPipeline so
        your environment variables / settings apply everywhere consistently.
        """
        url = self.rag.ollama_url or "http://localhost:11434/api/generate"
        model = getattr(self.rag, "model_name", None) or "llama3:8b"

        try:
            safe_prompt = self.sanitize_text(prompt)
            payload = {
                "model": model,
                "system": "You are a helpful assistant. Only use the provided content.",
                "prompt": safe_prompt,
                "stream": False,
                "keep_alive": "10m",
                "options": {
                    "temperature": 0.2,
                    "num_predict": 400,
                    "num_ctx": 3072,
                },
            }
            print(f"📡 Sending prompt to Ollama at {url} with model '{model}' ...")
            response = requests.post(url, json=payload, timeout=200)
            response.raise_for_status()

            result = (response.json().get("response") or "").strip()
            if not result:
                raise RuntimeError("Received empty response from Ollama.")
            return result

        except requests.exceptions.ConnectionError:
            return "❌ Ollama server is not reachable. Is it running?"
        except Exception as e:
            print(f"❌ Ollama Error: {str(e)}")
            return f"❌ LLM Error: {str(e)}"
