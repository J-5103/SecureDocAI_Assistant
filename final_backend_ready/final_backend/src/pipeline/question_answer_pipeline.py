# src/pipeline/question_answer_pipeline.py
import os
import requests
from typing import List, Optional, Tuple

from langchain_community.vectorstores import FAISS

from src.components.rag_pipeline import RAGPipeline
from src.utils.synonym_expander import SynonymExpander

# Directories (match main.py usage)
UPLOAD_BASE = os.path.abspath(os.path.join(os.getcwd(), "uploaded_docs"))
VSTORE_BASE = os.path.abspath(os.path.join(os.getcwd(), "vectorstores"))


def _safe_filename(name: str) -> str:
    """Strip any client path/backslashes and return just the filename."""
    return os.path.basename((name or "").replace("\\", "/"))


class QuestionAnswerPipeline:
    """
    High-level QA runner over your RAG pipeline.

    - Multi-file: pass selected doc_ids in `combine_docs` (these are the IDs you return from uploads).
    - Single-file: pass a single `document_id` (either a legacy stem or the new doc_id).
    - No selection: it will retrieve across every vectorstore under the chat.
    """

    def __init__(self):
        # Keep a single RAG pipeline instance so we share embeddings + Ollama config
        self.rag = RAGPipeline()

    def sanitize_text(self, text: str) -> str:
        return text.encode("utf-8", "replace").decode("utf-8")

    # ---------------- public entrypoint ----------------

    def run(
        self,
        question: str,
        chat_id: str,
        document_id: Optional[str] = None,         # single doc (legacy stem or new doc_id)
        combine_docs: Optional[List[str]] = None,  # multi-docs (list of doc_ids)
        return_sources: bool = False,
    ):
        """
        Returns one of:
          - answer (str)
          - (answer, sources) if return_sources=True (sources are either doc_ids used, or LangChain docs)
        """
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
            is_cost_question = any(kw in ql for kw in ["cost", "price", "charges", "amount", "budget"])

            # =================== MULTI-DOC MODE (selected) ===================
            if combine_docs and len(combine_docs) > 0:
                print("🔗 Combine Mode: Using selected documents only...")
                context, used_docs = self.rag.get_context(
                    question=expanded_question,
                    chat_id=chat_id,
                    document_id=None,
                    combine_docs=combine_docs,  # these are doc_ids/folder names under vectorstores/<chat_id>/
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
                # accepts legacy stems (e.g. "Jimi_patel_AIML") or new doc_ids ("jimi_patel_aiml-1a2b3c4d")
                safe_key = _safe_filename(str(document_id))
                stem = os.path.splitext(safe_key)[0]
                vs_folder = os.path.join(VSTORE_BASE, chat_id, stem)
                index_file = os.path.join(vs_folder, "index.faiss")

                # If vectorstore doesn't exist, user should (re)process or wait
                if not os.path.exists(index_file):
                    return (
                        f"❌ Vectorstore for '{stem}' is missing or not yet processed. "
                        f"Please upload the file under this chat again or wait for processing."
                    )

                context, used_docs = self.rag.get_context(
                    question=expanded_question,
                    chat_id=chat_id,
                    document_id=stem,   # folder/doc key (matches vectorstores/<chat_id>/<stem>)
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
                    try:
                        vectorstore = FAISS.load_local(
                            vs_folder,
                            self.rag.embedding_model,
                            allow_dangerous_deserialization=True,
                        )
                        docs = vectorstore.similarity_search(expanded_question, k=5)
                        return answer, docs
                    except Exception:
                        # If loading sources fails, still return the answer & the doc key
                        return answer, [stem]

                return answer

            # =================== NO SELECTION: ALL DOCS IN CHAT ===================
            print("ℹ️ No document selected; using all chat documents.")
            context, used_docs = self.rag.get_context(
                question=expanded_question,
                chat_id=chat_id,
                document_id=None,
                combine_docs=[],  # [] => "all docs" (RAGPipeline interprets this)
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

    def build_single_doc_prompt(self, question: str, context: str) -> str:
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

    def build_multi_doc_compare_prompt(self, question: str, combined_contexts: str) -> str:
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

    def build_cost_compare_prompt(self, question: str, combined_contexts: str) -> str:
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
