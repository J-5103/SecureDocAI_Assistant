import requests
import json
import os
import time

BASE_URL = "http://192.168.0.109:8000"  # 🔁 Replace with your actual FastAPI server IP if needed
EXCEL_FOLDER = "excel_test"       # 🗂 Folder with Excel or CSV files

def upload_excel(file_path):
    url = f"{BASE_URL}/excel/upload/"
    with open(file_path, "rb") as f:
        files = {"file": (os.path.basename(file_path), f, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")}
        response = requests.post(url, files=files)

    if response.status_code == 200:
        print(f"✅ Uploaded {file_path}: {response.json()}")
        return response.json()["file_path"]
    else:
        print(f"❌ Upload Failed for {file_path}: {response.text}")
        return None

def ask_excel_question(chat_id, file_path, question):
    url = f"{BASE_URL}/excel/ask/"
    payload = {
        "file_path": file_path,
        "question": question,
        "chat_id": chat_id
    }

    # Only file_path and question are expected in your backend now
    response = requests.post(url, data={
        "file_path": file_path,
        "question": question
    })

    if response.status_code == 200:
        answer = response.json().get("answer", "")
        print(f"🧠 AI Response ({chat_id}):\n{answer}\n")
    else:
        print(f"❌ Q&A Failed for {chat_id}: {response.text}")

def generate_excel_plot(file_path, question="generate a bar chart"):
    url = f"{BASE_URL}/excel/plot/"
    response = requests.post(url, data={
        "file_path": file_path,
        "question": question
    })

    if response.status_code == 200:
        image_path = response.json().get("image_path", "")
        print(f"📊 Plot generated: {image_path}")
    else:
        print(f"❌ Plot Generation Failed: {response.text}")


if __name__ == "__main__":
    print("🚀 Starting Excel Multi-Doc Test Pipeline...\n")

    uploaded_files = {}

    # 1️⃣ Upload Excel/CSV files
    excel_files = [f for f in os.listdir(EXCEL_FOLDER) if f.endswith((".xlsx", ".csv"))]

    for excel_file in excel_files:
        full_path = os.path.join(EXCEL_FOLDER, excel_file)
        uploaded_path = upload_excel(full_path)
        if uploaded_path:
            uploaded_files[excel_file] = uploaded_path

    # 2️⃣ Wait briefly
    time.sleep(2)

    # 3️⃣ Ask per-file question
    question = "📌 What is the total sales or sum of the second column?"
    for file_name, file_path in uploaded_files.items():
        chat_id = f"excel_chat_{file_name.replace('.xlsx', '').replace('.csv', '')}"
        print(f"\n📘 Asking Question for: {file_name}")
        ask_excel_question(chat_id, file_path, question)

    # 4️⃣ Ask global multi-file plot (one example)
    print("\n📊 Generating Plot for Each Excel File")
    for file_name, file_path in uploaded_files.items():
        generate_excel_plot(file_path, "Draw a bar chart of sales and profit with first five columns")
