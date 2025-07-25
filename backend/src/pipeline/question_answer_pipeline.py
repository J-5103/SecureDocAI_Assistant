import os
import requests
from src.components.rag_pipeline import RAGPipeline

class QuestionAnswerPipeline:
    def __init__(self):
        self.rag = RAGPipeline()

    def run(self, question, document_id=None):
        try:
            # ✅ Validate & normalize input
            if isinstance(question, list):
                question = " ".join(str(q) for q in question)
            elif not isinstance(question, str):
                raise Exception("❌ Invalid input: 'question' must be a string.")

            question = question.strip()
            is_summary_all = "all" in question.lower() and "summary" in question.lower()

            # Multi-document case
            if is_summary_all or document_id is None:
                print("🧠 Multi-Document QA Mode activated...")

                context, context_map = self.rag.get_context_from_multiple_docs(question, return_doc_map=True)

                # ✅ Clean context_map to flatten and remove invalid entries
                cleaned_context_map = {}
                for doc, passages in context_map.items():
                    cleaned_passages = []
                    for p in passages:
                        if isinstance(p, list):  # Flatten nested list
                            cleaned_passages.extend(str(x) for x in p if isinstance(x, str))
                        elif isinstance(p, str):
                            cleaned_passages.append(p)
                    cleaned_context_map[doc] = cleaned_passages

                # ✅ Check if cleaned content has any meaningful text
                if not cleaned_context_map or all(
                    not any(p.strip() for p in v) for v in cleaned_context_map.values()
                ):
                    return "❗ No relevant content found in any document to answer the question."

                prompt = self.build_multi_doc_prompt(question, cleaned_context_map)

            else:
                # Single document flow remains unchanged
                filename = document_id
                pdf_path = os.path.join("uploaded_docs", filename)
                vectorstore_folder = os.path.join("vectorstores", filename.replace(".pdf", ""))

                if not os.path.exists(pdf_path):
                    raise Exception(f"PDF {filename} not found in uploaded_docs folder.")

                if not os.path.exists(vectorstore_folder):
                    print(f"⚙️ Vectorstore not found for {filename}, creating one...")
                    try:
                        self.rag.create_vectorstore(pdf_path)
                    except Exception as e:
                        if "no extractable text" in str(e).lower():
                            return f"⚠️ Cannot process {filename}: No extractable text found."
                        else:
                            raise e

                context = self.rag.get_context(question, vectorstore_folder)

                if not context.strip():
                    return "❗ The document does not contain sufficient information to answer this question."

                prompt = self.build_single_doc_prompt(question, context, filename)

            return self.call_ollama(prompt)

        except Exception as e:
            print(f"❌ Pipeline Error: {str(e)}")
            raise Exception(f"Question Answer Pipeline Error: {str(e)}")

    def build_single_doc_prompt(self, question, context, filename):
        return f"""
You are an intelligent assistant answering questions based strictly on the document content below.

---

## 📄 Summary for: {filename}

### 📘 Explanation
Provide a detailed answer using only the information from this document.

---

## 📂 Document Content:
{context}

---

## ❓ User Question:
{question}

---

## 🧠 Answer Style:
- Structure your answer into sections.
- Use headings and bullet points.
- Do not add any external information.

### 📝 Answer Format:

#### 🧩 Key Takeaways
- ...

#### 📘 Explanation
...

#### 🔍 Citation (if available)
Mention relevant page or section if identifiable.
"""

    def build_multi_doc_prompt(self, question, context_map):
        combined_contexts = "\n\n".join(
            f"## 📄 Summary for: {doc_name}\n\n{'\n'.join(content)}"
            for doc_name, content in context_map.items()
            if isinstance(content, list) and any(isinstance(p, str) and p.strip() for p in content)
        )

        return f"""
You are a helpful assistant answering the user's question based strictly on the provided document summaries.

Each section below corresponds to a different document.

---

{combined_contexts}

---

## ❓ User Question:
{question}

---

## 🧠 Answer Style:
- Format the answer like NotebookLM: structured, readable, with bullet points.
- Do NOT use any outside knowledge.
- If information is missing, clearly state that.

### 📝 Answer Format for Each Document

#### 🧩 Key Takeaways
- ...

#### 📘 Explanation
...

#### 🔍 Citation (Optional)
Mention filename, page, or section if possible.
"""

    def call_ollama(self, prompt):
        url = "http://192.168.0.88:11434/api/generate"

        payload = {
            "model": "llama3:8b",
            "system": "You are a helpful AI that gives structured answers using document content only. No external info allowed.",
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.2,
                "num_predict": 1500
            }
        }

        print("📡 Sending prompt to Ollama...")
        response = requests.post(url, json=payload)

        if response.status_code != 200:
            print(f"❌ Ollama API Error: {response.status_code}, {response.text}")
            raise Exception(f"Ollama API Error: {response.status_code}")

        result = response.json().get("response", "").strip()

        if not result:
            raise Exception("❌ Empty response from Ollama.")

        return result
