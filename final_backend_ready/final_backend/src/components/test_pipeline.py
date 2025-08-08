import requests
import json
import os
import time

# ✅ Your actual API base
BASE_URL = "http://192.168.0.109:8000"
PDF_FOLDER = "test_files"  # 📁 Make sure this folder exists and contains your PDFs


def upload_file(pdf_path):
    url = f"{BASE_URL}/api/upload_file"
    try:
        with open(pdf_path, "rb") as f:
            filename = os.path.basename(pdf_path)
            files = {"file": (filename, f, "application/pdf")}
            response = requests.post(url, files=files)
    except Exception as e:
        print(f"❌ File open failed: {e}")
        return None

    if response.status_code == 200:
        print(f"✅ Uploaded {pdf_path}: {response.json()}")
        return response.json()
    else:
        print(f"❌ Upload Failed for {pdf_path}: {response.text}")
        return None


def list_documents():
    url = f"{BASE_URL}/list_documents"
    response = requests.get(url)

    if response.status_code == 200:
        docs = response.json().get("documents", [])
        print("📄 Available Documents:", docs)
        return docs
    else:
        print("❌ Document Listing Failed:", response.text)
        return []


def ask_question(chat_id, document_id, question):
    url = f"{BASE_URL}/api/ask"
    payload = {
        "chat_id": chat_id,
        "document_id": document_id,
        "question": question
    }

    headers = {"Content-Type": "application/json"}

    start_time = time.time()
    response = requests.post(url, headers=headers, data=json.dumps(payload))
    elapsed = time.time() - start_time

    if response.status_code == 200:
        answer = response.json().get("answer", "")
        print(f"🧠 AI Response ({document_id}):\n{answer}")
    else:
        print(f"❌ Chat Failed for {document_id}: {response.text}")

    print(f"⏱️ Time taken: {elapsed:.2f} seconds\n")


if __name__ == "__main__":
    print("🚀 Starting Multi-Doc Test Pipeline...\n")

    # 1️⃣ Upload PDFs
    pdf_files = [f for f in os.listdir(PDF_FOLDER) if f.endswith(".pdf")]
    for pdf in pdf_files:
        pdf_path = os.path.join(PDF_FOLDER, pdf)
        upload_file(pdf_path)

    # 2️⃣ Wait for all vectorstores to generate
    print("⏳ Waiting 3 more seconds for vectorstores to finalize...\n")
    time.sleep(3)

    # 3️⃣ List documents
    documents = list_documents()

    # 4️⃣ Ask Multi-Doc Question
    multi_question = "What is the cost breakdown in the Wave_Praposal?"
    print("\n📚 Asking Question for All Documents Together:")
    ask_question("multi_chat_001", "all", multi_question)
