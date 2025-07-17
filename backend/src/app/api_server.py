from fastapi import FastAPI
import asyncio
import requests
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

app = FastAPI()

# Load FAISS once
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.load_local("vectorstore/faiss_index/", embeddings, allow_dangerous_deserialization=True)

# Async call to LLaMA
async def call_llama(prompt):
    response = requests.post(
        "http://192.168.0.88:11434/api/generate",
        json={"model": "llama3:8b", "prompt": prompt, "stream": False}
    )
    return response.json()["response"]

@app.get("/ask")
async def ask(question: str):
    docs = vectorstore.similarity_search(question, k=1)  # Fast: top 1 only
    context = "\n".join([doc.page_content for doc in docs])

    prompt = f"Answer briefly based on the context.\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"

    answer = await asyncio.to_thread(call_llama, prompt)
    return {"answer": answer}
