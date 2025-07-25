import os
import requests
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader

class RAGPipeline:
    def __init__(self, vector_store_path="vectorstores", ollama_url="192.168.0.88:11434/api/generate", model_name="llama3:8b"):
        self.embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.vector_store_path = vector_store_path
        self.ollama_url = ollama_url
        self.model_name = model_name

    def create_vectorstore(self, pdf_path):
        filename = os.path.basename(pdf_path).replace(".pdf", "")
        vectorstore_folder = os.path.join(self.vector_store_path, filename)

        if not os.path.exists(vectorstore_folder):
            os.makedirs(vectorstore_folder)

        loader = PyPDFLoader(pdf_path)
        pages = loader.load_and_split()

        if len(pages) == 0:
            raise Exception(f"PDF {pdf_path} has no extractable text pages. Possibly scanned or corrupted.")

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(pages)

        vectorstore = FAISS.from_documents(chunks, self.embedding_model)
        vectorstore.save_local(vectorstore_folder)

        print(f"✅ Vectorstore created at: {vectorstore_folder}")

    def get_context(self, question, vectorstore_folder):
        if not os.path.exists(vectorstore_folder):
            raise Exception(f"Vectorstore not found at {vectorstore_folder}")

        vectorstore = FAISS.load_local(
            vectorstore_folder,
            self.embedding_model,
            allow_dangerous_deserialization=True
        )

        docs = vectorstore.similarity_search(question, k=5)
        context = "\n\n".join([doc.page_content for doc in docs])
        return context

    def get_context_from_multiple_docs(self, question, top_k=8, return_doc_map=False):
        print("🔍 Loading all vectorstores for multi-doc QA...")

        base_folder = self.vector_store_path
        vectorstores = []
        doc_map = {}  # document mapping

        # Load all vectorstores
        for folder_name in os.listdir(base_folder):
            folder_path = os.path.join(base_folder, folder_name)
            index_file = os.path.join(folder_path, "index.faiss")

            if os.path.isdir(folder_path) and os.path.exists(index_file):
                vs = FAISS.load_local(
                    folder_path,
                    self.embedding_model,
                    allow_dangerous_deserialization=True
                )
                vectorstores.append((folder_name, vs))

        if not vectorstores:
            raise Exception("No vectorstores found. Please upload documents first.")

        # Merge all vectorstores into one
        merged_vectorstore = vectorstores[0][1]
        for _, vs in vectorstores[1:]:
            merged_vectorstore.merge_from(vs)

        print(f"✅ Merged {len(vectorstores)} vectorstores.")

        # Perform similarity search
        docs = merged_vectorstore.similarity_search(question, k=top_k)
        context = "\n\n".join([doc.page_content for doc in docs])

        # Build and return document map if needed
        if return_doc_map:
            for doc in docs:
                source = doc.metadata.get("source", "unknown")
                if source not in doc_map:
                    doc_map[source] = []
                doc_map[source].append(doc.page_content)
            return context, doc_map  # ✅ Returns a tuple when return_doc_map=True

        return context  # ✅ Otherwise, just return context string


    def build_notebooklm_prompt(self, question, context):
        return f"""
You are an expert AI assistant. Answer the user's question **strictly using the provided document context**.

Be clear, structured, and detailed. **Do not hallucinate or add any external knowledge.**

---

## 📄 **Document Context**:
{context}

---

## ❓ **User Question**:
{question}

---

## 📝 **Answer Format**

### 📘 **Introduction**
Give a relevant introduction **only from the context**.

### 🔑 **Key Concepts**
Explain 2-3 important points found in the context.
- **Concept 1:** ...
- **Concept 2:** ...
- **Concept 3:** ...

### 💡 **Applications**
List 1-2 practical uses from the context.
- **Use Case 1:** ...
- **Use Case 2:** ...

### 📊 **Summary**
Summarize the key points **without adding external knowledge**.

---

## ⚠️ **Important Instructions**
- Do **NOT** use any outside knowledge.
- If the document lacks details, respond with:  
  "**The provided document does not contain sufficient details to answer this question.**"
"""

    def generate_answer(self, question, context):
        if not context.strip():
            return "⚠️ The provided document does not contain sufficient details to answer this question."

        prompt = self.build_notebooklm_prompt(question, context)

        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False
        }

        try:
            response = requests.post(self.ollama_url, json=payload)
            response.raise_for_status()
            result = response.json()
            return result.get("response", "").strip()
        except Exception as e:
            print(f"❌ Error during generation: {str(e)}")
            return "⚠️ Failed to generate an answer due to a model or connection error."

    def ask_question_single_doc(self, question, vectorstore_folder):
        context = self.get_context(question, vectorstore_folder)
        return self.generate_answer(question, context)

    def ask_question_multi_doc(self, question):
        context = self.get_context_from_multiple_docs(question)
        return self.generate_answer(question, context)
