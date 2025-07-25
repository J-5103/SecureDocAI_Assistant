import requests
import json
import os
import time

BASE_URL = "http://192.168.0.109:8000"
PDF_FOLDER = "test_files"


def upload_pdf(pdf_path):
    url = f"{BASE_URL}/upload_pdf"
    with open(pdf_path, "rb") as f:
        files = {"file": (os.path.basename(pdf_path), f, "application/pdf")}
        response = requests.post(url, files=files)

    if response.status_code == 200:
        print(f"✅ Uploaded {pdf_path}: {response.json()}")
    else:
        print(f"❌ Upload Failed for {pdf_path}: {response.text}")
    return response.json()

def list_documents():
    url = f"{BASE_URL}/list_documents"
    response = requests.get(url)

    if response.status_code == 200:
        docs = response.json()["documents"]
        print("📄 Available Documents:", docs)
        return docs
    else:
        print("❌ Document Listing Failed:", response.text)
        return []

def ask_question(chat_id, document_id, question):
    url = f"{BASE_URL}/document-chat"

    payload = {
        "chat_id": chat_id,
        "document_id": document_id,
        "question": question
    }

    headers = {"Content-Type": "application/json"}

    start_time = time.time()
    response = requests.post(url, data=json.dumps(payload), headers=headers)
    end_time = time.time()
    elapsed = end_time - start_time

    if response.status_code == 200:
        answer = response.json().get("answer", "")
        print(f"🧠 AI Response ({document_id}):\n{answer}")
        print(f"⏱️ Time taken: {elapsed:.2f} seconds\n")

        
    else:
        print(f"❌ Chat Failed for {document_id}: {response.text}")



if __name__ == "__main__":
    print("🚀 Starting Multi-Doc Test Pipeline...\n")

    # 1️⃣ Upload PDFs
    pdf_files = [f for f in os.listdir(PDF_FOLDER) if f.endswith(".pdf")]
    for pdf in pdf_files:
        pdf_path = os.path.join(PDF_FOLDER, pdf)
        upload_pdf(pdf_path)

    # 2️⃣ Wait briefly
    time.sleep(2)

    # 3️⃣ List available documents
    documents = list_documents()

    # 4️⃣ Ask per-document question
    question = "📌 Summarize this document with an emphasis on key insights, use-cases, and unique takeaways."
    for doc in documents:
        chat_id = f"chat_{doc.replace('.pdf', '')}"
        print(f"\n📘 Asking Question for: {doc}")
        ask_question(chat_id, doc, question)

    # 5️⃣ Ask multi-doc question
    multi_question = "📌 Provide a consolidated summary of all uploaded documents with key patterns and also draw a trend plot."
    print("\n📚 Asking Question for All Documents Together:")
    ask_question("multi_chat_001", "all", multi_question)
