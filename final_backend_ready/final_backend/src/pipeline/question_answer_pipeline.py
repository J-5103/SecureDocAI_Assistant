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
            is_cost_question = any(
                kw in ql
                for kw in ["cost", "price", "charges", "amount", "budget", "fee", "bill", "expense"]
            )

            # =================== MULTI-DOC MODE (selected) ===================
            if combine_docs and len(combine_docs) > 0:
                print("🔗 Combine Mode: Using selected documents only...")
                context, used_docs = self.rag.get_context(
                    question=expanded_question,
                    chat_id=chat_id,
                    document_id=None,
                    combine_docs=combine_docs,  # these are doc_ids/folder names under vectorstores/<chat_id>/
                    k=12,  # a bit wider for multi-doc coverage
                )
                if not context.strip():
                    raise RuntimeError("No relevant content found in the selected documents.")

                combined_contexts = self.sanitize_text(context)

                # Pass the concrete doc IDs/names we actually retrieved to help the model cite correctly.
                used_doc_lines = "\n".join(f"• {d}" for d in (used_docs or []))
                doc_hint = f"\nKnown document identifiers for citation:\n{used_doc_lines}\n" if used_docs else ""

                prompt = (
                    self.build_cost_compare_prompt(question, combined_contexts, doc_hint)
                    if is_cost_question
                    else self.build_multi_doc_compare_prompt(question, combined_contexts, doc_hint)
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
                    k=10,
                )
                if not context or not context.strip():
                    raise RuntimeError("No relevant content found in the document.")

                sanitized_context = self.sanitize_text(context)
                prompt = self.build_single_doc_prompt(question, sanitized_context, stem)
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
                k=12,
            )
            if not context.strip():
                raise RuntimeError("No relevant content found across chat documents.")

            combined_contexts = self.sanitize_text(context)
            used_doc_lines = "\n".join(f"• {d}" for d in (used_docs or []))
            doc_hint = f"\nKnown document identifiers for citation:\n{used_doc_lines}\n" if used_docs else ""

            prompt = (
                self.build_cost_compare_prompt(question, combined_contexts, doc_hint)
                if is_cost_question
                else self.build_multi_doc_compare_prompt(question, combined_contexts, doc_hint)
            )
            answer = self.call_ollama(prompt)
            return (answer, used_docs) if return_sources else answer

        except Exception as e:
            print(f"❌ Pipeline Error: {str(e)}")
            return (f"❌ Backend Error: {str(e)}", []) if return_sources else f"❌ Backend Error: {str(e)}"

    # ---------------- prompt builders (table-aware & citation-guarded) ----------------

    def _table_skills(self) -> str:
        """
        Reusable instructions that teach the model to read/normalize tabular context.
        """
        return (
            "If the context includes tables (markdown tables, CSV-like lines, or PDF-extracted rows):\n"
            "• Treat each row as data; infer headers from surrounding text if missing.\n"
            "• Preserve units and numbers; normalize currencies but keep the original symbol (e.g., $/₹).\n"
            "• When quoting evidence, you may quote a cell or a short row snippet.\n"
            "• If page/section markers like 'p. 3' or 'Page: 3' appear near the text, include them in the table.\n"
        )

    def build_single_doc_prompt(self, question: str, context: str, doc_name: str) -> str:
        return f"""
You are a precise assistant answering ONLY from the provided document content.

Document Identifier:
- {doc_name}

Document Content (verbatim snippets & table fragments):
-------------------------------------------------------
{context}

User Question:
--------------
{question}

Instructions:
-------------
- Answer strictly from the document content above; do NOT invent facts.
- Prefer short, structured bullet points or a tight paragraph.
- {_safe_strip_margin(self._table_skills())}
- If the answer is not present, reply: "Not found in the provided document."
- Do not include file paths.

Output:
-------
Provide the best possible answer concisely. If helpful, include a tiny Markdown table.
""".strip()

    def build_multi_doc_compare_prompt(self, question: str, combined_contexts: str, doc_hint: str = "") -> str:
        return f"""
You are comparing multiple documents. Use ONLY the context below to produce a decision-ready comparison.

{doc_hint}
Context (verbatim snippets & table fragments from all selected docs):
--------------------------------------------------------------------
{combined_contexts}

User Question:
--------------
{question}

Output format (use ALL sections and keep headings exactly):
# Executive Summary
• 3–6 bullet points capturing the main conclusions and most important differences.

# Comparison Table
Create a compact markdown table with columns exactly:
| Criteria | Doc Name | Evidence (quote or paraphrase) | Page/Section |
- Include the most important 6–12 rows.
- Criteria can be Education, Skills, Experience, Achievements, Dates, Costs, etc.
- Doc Name must match the identifiers provided.
- Evidence should be short and specific (quote or paraphrase).
- Page/Section should use any page/section markers present in the context.

# Key Differences / Conflicts
• Bullet points where documents disagree or differ materially, with brief justification.

# Gaps / Missing Info
• Bullet points for unclear, missing, or contradictory parts.

# Recommendation
• 1–2 sentences suggesting the best option or next steps, with trade-offs.

Rules:
- Do NOT hallucinate; if something is not found, say "Not stated".
- {_safe_strip_margin(self._table_skills())}
- Keep the total under ~500 words.
""".strip()

    def build_cost_compare_prompt(self, question: str, combined_contexts: str, doc_hint: str = "") -> str:
        return f"""
You are a finance analyst extracting and comparing costs from multiple documents. Use ONLY the context below.

{doc_hint}
Context (verbatim snippets & table fragments):
----------------------------------------------
{combined_contexts}

User Question:
--------------
{question}

Output format (use ALL sections and keep headings exactly):
# Executive Summary
• 3–5 bullets summarizing totals, major drivers, and notable differences.

# Comparison Table
| Cost Item | Doc Name | Amount | Evidence (quote/paraphrase) | Page/Section |
- Amount must include the currency symbol exactly as shown (e.g., $, ₹, Rs).
- If a document lacks costs for an item, write "Not stated".

# Notes & Assumptions
• Any normalization, ranges, or missing values.

# Recommendation
• Brief guidance on lowest TCO / most reasonable option with trade-offs.

Rules:
- Extract only explicit numeric costs (e.g., "$20,000", "₹50,000").
- Do NOT guess numbers; if absent, say "Not stated".
- {_safe_strip_margin(self._table_skills())}
- Keep the total under ~500 words.
""".strip()

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
                "system": (
                    "You are a meticulous analyst. Use ONLY the provided context. "
                    "If information is missing, explicitly say so. Keep answers precise."
                ),
                "prompt": safe_prompt,
                "stream": False,
                "keep_alive": "10m",
                "options": {
                    "temperature": 0.15,
                    "top_p": 0.9,
                    "num_predict": 700,   # allow a full table + sections
                    "num_ctx": 6144,      # larger context for multi-doc
                },
            }
            print(f"📡 Sending prompt to Ollama at {url} with model '{model}' ...")
            response = requests.post(url, json=payload, timeout=240)
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


def _safe_strip_margin(s: str) -> str:
    """Utility: remove leading spaces from a multi-line snippet for cleaner prompts."""
    return "\n".join(line.strip() for line in s.splitlines())
