import os
import requests
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from langchain_community.document_loaders import TextLoader

class RAGPipeline:
    def __init__(self, vector_store_path="faiss_index"):
        self.vector_store_path = vector_store_path
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.vectorstore = None

        # Load existing vectorstore if available
        if os.path.exists(vector_store_path):
            self.vectorstore = FAISS.load_local(
                vector_store_path,
                self.embeddings,
                allow_dangerous_deserialization=True
            )

    def create_vector_store(self, documents):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        docs = text_splitter.create_documents([doc.page_content for doc in documents])

        self.vectorstore = FAISS.from_documents(docs, self.embeddings)
        self.vectorstore.save_local(self.vector_store_path)

    def call_ollama_api(self, prompt, model="llama3:8b"):
        url = "http://192.168.0.88:11434/api/generate"

        payload = {
            "model": model,
            "prompt": str(prompt),
            "stream": False,
            "options": {
                "temperature": 0.2,
                "num_predict": 512
            }
        }

        response = requests.post(url, json=payload)

        if response.status_code != 200:
            raise Exception(f"Ollama API Error: {response.json().get('error', 'Unknown error')}")

        return response.json()["response"]

    def ask_question(self, query):
        if self.vectorstore is None:
            raise ValueError("Vectorstore is not initialized!")

        docs = self.vectorstore.similarity_search(query, k=4)
        context = "\n".join([doc.page_content for doc in docs])

        prompt = f"""
        You are an expert AI assistant.

        Use the following context to answer the question in a **detailed and comprehensive manner**.

        Context:
        {context}

        Question: {query}

        Answer:
        """
        return self.call_ollama_api(prompt)

# ----------- Usage Example -----------

if __name__ == "__main__":
    print("🚀 Initializing RAG Pipeline with Mistral via Ollama...")

    pipeline = RAGPipeline()

    if pipeline.vectorstore is None:
        print("📄 No FAISS index found, creating new index...")
        loader = TextLoader("documents/sample.txt")  # ✅ Put your data here
        documents = loader.load()
        pipeline.create_vector_store(documents)
        print("✅ Vector DB Created.")

    query = "What is machine learning?"
    answer = pipeline.ask_question(query)
    print("🧠 Answer:", answer)
