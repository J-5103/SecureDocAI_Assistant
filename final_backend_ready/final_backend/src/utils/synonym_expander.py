import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


class SynonymExpander:
    def __init__(self, vectorstore_dir="vectorstores", model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.vectorstore_dir = vectorstore_dir
        self.embedding_model = HuggingFaceEmbeddings(model_name=model_name)
        self.expanded_query = ""

    def find_similar_words(self, query: str, top_k: int = 10):
        """
        Expands a query by finding semantically similar words across all vectorstores.
        """
        query = query.strip().lower()
        similar_terms = set(query.split())

        for folder in os.listdir(self.vectorstore_dir):
            vs_path = os.path.join(self.vectorstore_dir, folder)
            index_file = os.path.join(vs_path, "index.faiss")
            if not os.path.isdir(vs_path) or not os.path.exists(index_file):
                continue

            try:
                vs = FAISS.load_local(
                    vs_path,
                    self.embedding_model,
                    allow_dangerous_deserialization=True
                )

                docs = vs.similarity_search(query, k=top_k)
                for doc in docs:
                    words = doc.page_content.strip().lower().split()
                    similar_terms.update([
                        w.strip(".,:;!?()[]") for w in words if len(w) > 3
                    ])
            except Exception as e:
                print(f"⚠️ Error loading vectorstore {folder}: {e}")

        self.expanded_query = " ".join(sorted(similar_terms))
        return self.expanded_query

    def replace(self, old: str, new: str, count: int = -1) -> str:
        """
        Replaces text inside the expanded query. Call `find_similar_words()` first.
        """
        if not self.expanded_query:
            raise ValueError("Run find_similar_words() before calling replace()")
        return self.expanded_query.replace(old, new, count)

    def __str__(self):
        return self.expanded_query

    def __call__(self):
        return self.expanded_query


# ✅ Example usage for testing
if __name__ == "__main__":
    expander = SynonymExpander()
    expanded = expander.find_similar_words("support")
    print("Expanded:", expanded)
    print("Replaced:", expander.replace("support", "assist"))
