import os
import shutil
from typing import List, Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Body
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.pipeline.question_answer_pipeline import QuestionAnswerPipeline
from src.pipeline.document_pipeline import DocumentPipeline
from src.pipeline.image_question_pipeline import ImageQuestionPipeline
from src.components.plot_generator import PlotGenerator
from src.components.file_loader import FileLoader

# 🧠 Pipelines
qa_pipeline = QuestionAnswerPipeline()
doc_pipeline = DocumentPipeline()

# 🚀 FastAPI
app = FastAPI(
    title="SecureDocAI Backend",
    description="Multi-Document AI Chatbot with Vectorstore and Auto Plotting",
    version="1.0.0",
)

# 🌐 CORS
# Allow your Vite URL in prod; allow all in dev via env flag, if needed
# FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "http://192.168.0.109:3000")
DEV_OPEN_CORS = os.getenv("DEV_OPEN_CORS", "false").lower() == "true"

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://192.168.0.109:3000"],  # if DEV_OPEN_CORS else [FRONTEND_ORIGIN],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 📁 Paths
UPLOAD_BASE = "uploaded_docs"     # per-chat PDFs & others
UPLOAD_EXCEL = "uploaded_excels"
os.makedirs(UPLOAD_BASE, exist_ok=True)
os.makedirs(UPLOAD_EXCEL, exist_ok=True)

ALLOWED_UPLOAD_EXTS = {".pdf", ".doc", ".docx", ".xls", ".xlsx", ".png", ".jpg", ".jpeg"}
PDF_EXTS = {".pdf"}

# ----------------------- Uploads -----------------------

@app.post("/api/upload/upload_file")
async def upload_file(chat_id: str = Form(...), file: UploadFile = File(...)):
    try:
        ext = os.path.splitext(file.filename)[-1].lower()
        if ext not in ALLOWED_UPLOAD_EXTS:
            raise HTTPException(status_code=400, detail="Only PDF, Word, Excel, or image files allowed.")

        chat_folder = os.path.join(UPLOAD_BASE, chat_id)
        os.makedirs(chat_folder, exist_ok=True)

        document_id = file.filename  # store original name
        saved_path = os.path.join(chat_folder, document_id)

        with open(saved_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Build vectorstore only for PDFs
        if ext in PDF_EXTS:
            doc_pipeline.run(saved_path, document_id, chat_id=chat_id)
            status = "vectorstore_created"
        else:
            status = "saved"

        return {
            "document_path": saved_path,
            "filename": document_id,
            "chat_id": chat_id,
            "status": status,
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# ✅ List documents by chat
@app.get("/api/list_documents")
async def list_documents(chat_id: str):
    try:
        folder = os.path.join(UPLOAD_BASE, chat_id)
        if not os.path.exists(folder):
            return {"documents": []}

        files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
        pdfs = [f for f in files if f.lower().endswith(".pdf")]

        return {"documents": [{"name": f, "documentId": f} for f in pdfs]}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# ----------------------- Ask (PDF QA) -----------------------

class AskRequest(BaseModel):
    chat_id: str
    question: str
    document_id: Optional[str] = None      # single doc ("combine" means multi-select mode)
    combine_docs: Optional[List[str]] = None  # multi-doc list

@app.post("/api/ask")
async def ask_question(request: AskRequest = Body(...)):
    try:
        question = (request.question or "").strip()
        if not question or not request.chat_id:
            raise HTTPException(status_code=400, detail="Missing chat_id or question.")

        # Determine mode
        doc_id: Optional[str] = None
        combine_list: Optional[List[str]] = None

        if request.document_id and request.document_id != "combine":
            doc_id = request.document_id
        elif request.combine_docs is not None:
            # [] means "all docs in chat" (handled in pipeline)
            combine_list = [d for d in request.combine_docs if d]
        else:
            combine_list = []  # default: all docs

        print(
            f"🧠 Question: {question} | chat: {request.chat_id} "
            f"| single_doc: {doc_id} | combine_docs: {combine_list}"
        )

        # Run pipeline (returns plain string)
        answer = qa_pipeline.run(
            question=question,
            chat_id=request.chat_id,
            document_id=doc_id,
            combine_docs=combine_list,
        )

        # Normalize to string and wrap in JSON so frontend always reads .answer
        if isinstance(answer, tuple):
            answer = answer[0]
        answer = "" if answer is None else str(answer)

        return JSONResponse(
            status_code=200,
            content={
                "answer": answer,
                "question": question,
                "chat_id": request.chat_id,
                "document_id": request.document_id,
                "combine_docs": combine_list or [],
            },
        )

    except FileNotFoundError:
        return JSONResponse(status_code=404, content={"error": "Document not found."})
    except HTTPException as he:
        return JSONResponse(status_code=he.status_code, content={"error": he.detail})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# ----------------------- Image QA -----------------------

@app.post("/api/ask-image")
async def ask_image(front_image: UploadFile = File(...), back_image: UploadFile = File(None)):
    try:
        front_bytes = await front_image.read()
        back_bytes = await back_image.read() if back_image else None
        pipeline = ImageQuestionPipeline()
        result = pipeline.run(front_bytes, back_bytes)
        return {"status": "success", "data": result}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# ----------------------- Excel helpers -----------------------

@app.post("/api/excel/upload/")
def upload_excel(file: UploadFile = File(...)):
    try:
        ext = os.path.splitext(file.filename)[-1].lower()
        if ext not in [".xlsx", ".xls", ".csv"]:
            raise HTTPException(status_code=400, detail="Only Excel or CSV files allowed.")

        file_path = os.path.join(UPLOAD_EXCEL, file.filename)
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        return {"file_path": file_path, "message": "Upload successful."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/excel/ask/")
def ask_excel_question(file_path: str = Form(...), question: str = Form(...)):
    try:
        df = FileLoader.load_excel_data(file_path)
        question_l = question.lower()

        numeric_cols = df.select_dtypes(include="number").columns
        if not len(numeric_cols):
            raise HTTPException(status_code=400, detail="No numeric columns found.")

        if "total" in question_l or "sum" in question_l:
            answer = df[numeric_cols].sum().to_string()
        elif "average" in question_l or "mean" in question_l:
            answer = df[numeric_cols].mean().to_string()
        else:
            raise HTTPException(status_code=400, detail="Unrecognized question.")

        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/excel/plot/")
def plot_excel_data(file_path: str = Form(...), question: str = Form(...)):
    try:
        df = FileLoader.load_excel_data(file_path)
        plot_gen = PlotGenerator(df)
        base64_img = plot_gen.generate_plot(question)
        return {"image_base64": base64_img, "message": "Plot generated successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ----------------------- Health -----------------------

@app.get("/api/health")
async def health():
    return {"status": "ok"}

@app.get("/")
async def root():
    return {"message": "✅ SecureDocAI Backend is running"}
