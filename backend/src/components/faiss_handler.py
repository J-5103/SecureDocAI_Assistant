from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import os

class FAISSHandler:
    def __init__(self, index_path="faiss_index/"):
        self.index_path = index_path
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    def build_and_save(self, documents):
        vectorstore = FAISS.from_documents(documents, self.embeddings)
        vectorstore.save_local(self.index_path)

    def load(self):
        if not os.path.exists(self.index_path):
            raise FileNotFoundError("FAISS index not found. Run build_and_save first.")

        return FAISS.load_local(
            self.index_path,
            self.embeddings,
            allow_dangerous_deserialization=True  # ✅ Trust your own FAISS files
        )
