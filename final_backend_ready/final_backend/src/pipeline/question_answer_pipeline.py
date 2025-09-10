# src/pipeline/question_answer_pipeline.py
from src.utils.pydantic_compact import apply_pydantic_compat
apply_pydantic_compat()


import os
import requests
import traceback  # Import the traceback module
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
    """
    def __init__(self):
        self.rag = RAGPipeline()

    def sanitize_text(self, text: str) -> str:
        return text.encode("utf-8", "replace").decode("utf-8")

    def run(
        self,
        question: str,
        chat_id: str,
        document_id: Optional[str] = None,
        combine_docs: Optional[List[str]] = None,
        return_sources: bool = False,
    ):
        try:
            # (The rest of the run method is the same as before...)
            if not question:
                raise ValueError("No question provided.")
            if isinstance(question, list):
                question = " ".join(str(q) for q in question)
            elif not isinstance(question, str):
                raise TypeError("Invalid input: 'question' must be a string.")
            question = question.strip()
            if not question:
                raise ValueError("Question is empty after stripping.")

            expander = SynonymExpander()
            expanded_question = expander.find_similar_words(question)

            ql = question.lower()
            is_cost_question = any(
                kw in ql
                for kw in ["cost", "price", "charges", "amount", "budget", "fee", "bill", "expense"]
            )

            if combine_docs and len(combine_docs) > 0:
                print("🔗 Combine Mode: Using selected documents only...")
                context, used_docs = self.rag.get_context(
                    question=expanded_question, chat_id=chat_id, document_id=None, combine_docs=combine_docs, k=12
                )
                if not context.strip():
                    raise RuntimeError("No relevant content found in the selected documents.")
                combined_contexts = self.sanitize_text(context)
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

            if document_id:
                safe_key = _safe_filename(str(document_id))
                stem = os.path.splitext(safe_key)[0]
                vs_folder = os.path.join(VSTORE_BASE, chat_id, stem)
                index_file = os.path.join(vs_folder, "index.faiss")
                if not os.path.exists(index_file):
                    return (
                        f"❌ Vectorstore for '{stem}' is missing or not yet processed. "
                        f"Please upload the file under this chat again or wait for processing."
                    )
                context, used_docs = self.rag.get_context(
                    question=expanded_question, chat_id=chat_id, document_id=stem, combine_docs=None, k=10
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
                            vs_folder, self.rag.embedding_model, allow_dangerous_deserialization=True
                        )
                        docs = vectorstore.similarity_search(expanded_question, k=5)
                        return answer, docs
                    except Exception:
                        return answer, [stem]
                return answer

            print("ℹ️ No document selected; using all chat documents.")
            context, used_docs = self.rag.get_context(
                question=expanded_question, chat_id=chat_id, document_id=None, combine_docs=[], k=12
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
            # *** NEW IMPROVED ERROR LOGGING ***
            # This will print the full error details to your server console
            print("❌ Pipeline Error: An unexpected error occurred. Full traceback below:")
            traceback.print_exc()
            
            # Return a more informative error message to the frontend
            error_type = type(e).__name__
            error_msg = str(e)
            full_error = f"❌ Backend Error: {error_type}('{error_msg}')"
            return (full_error, []) if return_sources else full_error

    # (No changes needed for the prompt builders or _table_skills methods)
    # ...
    def _table_skills(self) -> str:
        return (
            "If the context includes tables (markdown tables, CSV-like lines, or PDF-extracted rows):\n"
            "• Treat each row as data; infer headers from surrounding text if missing.\n"
            "• Preserve units and numbers; normalize currencies but keep the original symbol (e.g., $/₹).\n"
            "• When quoting evidence, you may quote a cell or a short row snippet.\n"
            "• If page/section markers like 'p. 3' or 'Page: 3' appear near the text, include them in the table.\n"
        )
    def build_single_doc_prompt(self, question: str, context: str, doc_name: str) -> str:
        return f"""
    You are a precise assistant answering ONLY from the provided document content (this may be selectable text or OCR from scanned/handwritten images).
    Document Identifier: - {doc_name}
    Document Content (verbatim snippets & table fragments):
    -------------------------------------------------------
    {context}
    User Question:
    --------------
    {question}
    Instructions:
    -------------
    - Use only the content above; do NOT invent facts.
    - First infer the modality briefly in your mind: text OR image (scan/photo/handwritten). If image quality is poor, keep answers conservative.
    - If the document appears to be an ID or a filled form, extract key fields when present:
    • Aadhaar: name, dob_or_yob, gender, aadhaar_number, address
    • PAN: name, father_name, dob (YYYY-MM-DD), pan_number (pattern AAAAA9999A)
    • Ration Card: card_number, state, holder_name, family_members, address
    • Generic form: form_title; applicant (name, mobile, email, address); ids_mentioned (aadhaar, pan, ration); fields (key→value); signatures_present (true/false); photo_present (true/false); dates (YYYY-MM-DD)
    - Prefer short, structured bullet points or a tight paragraph.
    - {_safe_strip_margin(self._table_skills())}
    - Normalize dates to YYYY-MM-DD when possible; you may keep the raw date in parentheses if helpful.
    - If any specific field is unreadable from the scan/handwriting, write "Unreadable" for that field.
    - If the answer is not present, reply: "Not found in the provided document."
    - Do not include file paths.
    Output:
    -------
    Provide the best possible answer concisely. If helpful, include a tiny Markdown table.
    """.strip()
    def build_multi_doc_compare_prompt(self, question: str, combined_contexts: str, doc_hint: str = "") -> str:
        return f"""
        You are comparing multiple documents (mix of text PDFs and scanned/handwritten images). Use ONLY the provided context (text snippets, OCR text, and model-attached images).
        {doc_hint}
        Context (verbatim snippets & table fragments from all selected docs):
        --------------------------------------------------------------------
        {combined_contexts}
        User Question:
        --------------
        {question}
        Output format (use ALL sections and keep headings exactly):
        # Executive Summary
        • 3–6 bullets with the main conclusions and most important differences across documents.
        # Document Registry
        List each doc with its inferred modality/type:
        - Doc Name → modality: text|image, doc_type: aadhaar|pan|ration|form|other/unknown, notes (if any)
        # Comparison Table
        Create a compact markdown table with columns exactly:
        | Criteria | Doc Name | Evidence (quote or paraphrase) | Page/Section |
        - Include the most important 6–12 rows.
        - Criteria can be Dates, Costs, KYC Fields (name/ids/address), Requirements, Clauses, etc.
        - Doc Name must match the identifiers provided.
        - Evidence must be short and specific.
        - Page/Section uses whatever page/section markers appear in context.
        # KYC / Form Field Extracts (if applicable)
        Summarize extracted fields (Aadhaar/PAN/Ration/Form) briefly:
        - Aadhaar: name, dob/yob, gender, aadhaar_number, address
        - PAN   : name, father_name, dob, pan_number
        - Ration: card_number, state, holder_name, family_members, address
        - Form  : form_title, applicant (name/mobile/email/address), ids_mentioned, key fields, signatures_present, photo_present, dates
        # Pairwise Differences Matrix
        List notable doc-to-doc conflicts/differences (names, numbers, dates, totals, clauses). Keep each bullet concise.
        # Gaps / Missing Info
        • Bullets for unclear, missing, or contradictory parts (e.g., unreadable handwriting).
        # Recommendation
        • 1–2 sentences suggesting the best option or next steps, with trade-offs.
        Rules:
        - Do NOT hallucinate; if something is not found, say "Not stated".
        - Normalize dates to YYYY-MM-DD where possible; keep raw in Evidence if useful.
        - If a field is unreadable (scan/handwriting), set null/Not stated and mention in Gaps.
        - {_safe_strip_margin(self._table_skills())}
        - Keep total under ~500 words.
        """.strip()
    def build_cost_compare_prompt(self, question: str, combined_contexts: str, doc_hint: str = "") -> str:
        return f"""
    You are a finance analyst extracting and comparing costs from multiple documents. Use ONLY the context below.
    Documents may be text-based or OCR from scanned/handwritten images; if handwriting is unclear, be conservative.
    {doc_hint}
    Context (verbatim snippets & table fragments):
    ----------------------------------------------
    {combined_contexts}
    User Question:
    --------------
    {question}
    Output format (use ALL sections and keep headings exactly):
    # Executive Summary
    • 3–5 bullets summarizing totals, major cost drivers, taxes/fees, notable differences, and any mismatches between computed and reported totals.
    # Comparison Table
    | Cost Item | Doc Name | Amount | Evidence (quote/paraphrase) | Page/Section |
    - Amount must include the currency symbol exactly as shown (e.g., $, ₹, Rs, USD, INR) and any unit (per month, per hour).
    - Show taxes and fees when explicitly stated (GST, CGST, SGST, IGST, TDS, surcharge, discount).
    - If an item is presented as quantity × unit price, show the final line amount as written; if you compute a check, note it in "Notes & Assumptions".
    - If a document lacks costs for an item, write "Not stated".
    - If text is present but unreadable due to scan/handwriting, write "Unreadable".
    # Notes & Assumptions
    • Any normalization, ranges, or missing values.
    • Briefly show any check you performed (for example, subtotal plus taxes equals reported total). If there is a mismatch, mention it here.
    • Do not convert currencies using external rates; only normalize or convert if an explicit rate is provided in the context.
    • Mention OCR or handwriting uncertainties if they affect amounts.
    # Recommendation
    • Brief guidance on the lowest TCO or most reasonable option with trade-offs, based only on the provided context.
    Rules:
    - Extract only explicit numeric costs exactly as written (examples: "$20,000", "₹50,000", "Rs 1,200", "USD 300").
    - Do NOT guess numbers; if absent, say "Not stated". If illegible, say "Unreadable".
    - Include discounts, surcharges, one-time fees, and recurring charges when explicitly present.
    - Keep dates in YYYY-MM-DD if normalization is possible; keep raw text in Evidence if helpful.
    - Do not include file paths.
    - {_safe_strip_margin(self._table_skills())}
    - Keep the total under ~500 words.
    """.strip()

    def call_ollama(self, prompt: str) -> str:
        """
        Calls Ollama with the same base URL/model configured in RAGPipeline so
        your environment variables / settings apply everywhere consistently.
        """
        url = self.rag.ollama_url or "http://192.168.0.88:11434/api/generate"
        model = getattr(self.rag, "model_name", None) or "minicpm-v:latest"
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
                    "temperature": 0.15, "top_p": 0.9, "num_predict": 700, "num_ctx": 6144,
                },
            }
            print(f"📡 Sending prompt to Ollama at {url} with model '{model}' ...")
            response = requests.post(url, json=payload, timeout=2000)
            response.raise_for_status()
            response_data = response.json()
            if 'error' in response_data:
                error_message = response_data['error']
                print(f"❌ Ollama API returned an error: {error_message}")
                raise RuntimeError(f"Ollama API Error: {error_message}")
            result = (response_data.get("response") or "").strip()
            if not result:
                raise RuntimeError("Received empty response from Ollama.")
            return result
        except requests.exceptions.ConnectionError:
            raise RuntimeError("Ollama server is not reachable. Is it running?")
        except Exception as e:
            print(f"❌ Ollama Error: {str(e)}")
            raise e

def _safe_strip_margin(s: str) -> str:
    """Utility: remove leading spaces from a multi-line snippet for cleaner prompts."""
    return "\n".join(line.strip() for line in s.splitlines())