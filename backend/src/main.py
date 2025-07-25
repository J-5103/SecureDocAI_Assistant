from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from src.components.plot_generator import PlotGenerator
from src.components.file_loader import FileLoader
from fastapi import Query
from pydantic import BaseModel
import shutil
from fastapi import UploadFile, File, Form
import os

# Pipelines
from src.pipeline.question_answer_pipeline import QuestionAnswerPipeline
from src.pipeline.document_pipeline import DocumentPipeline

# Initialize pipelines
qa_pipeline = QuestionAnswerPipeline()
doc_pipeline = DocumentPipeline()


# Initialize FastAPI
app = FastAPI(
    title="SecureDocAI Backend",
    description="Multi-Document AI Chatbot with Vectorstore and Auto Plotting",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development; restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 📁 Create upload directory if not exists
UPLOAD_FOLDER = "uploaded_docs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 📤 Upload PDF and Create Vectorstore
@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")

    save_path = os.path.join(UPLOAD_FOLDER, file.filename)

    try:
        with open(save_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Generate vectorstore for the uploaded PDF
        doc_pipeline.create_vectorstore_for_pdf(save_path)

        return {
            "document_path": save_path,
            "filename": file.filename,
            "status": "vectorstore_created"
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Upload/Vectorstore Error: {str(e)}"})

# 📄 List Uploaded PDFs
@app.get("/list_documents")
async def list_documents():
    try:
        files = os.listdir(UPLOAD_FOLDER)
        pdfs = [f for f in files if f.endswith(".pdf")]
        return {"documents": pdfs}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Failed to list documents: {str(e)}"})

# 📥 Document Question Request Model
class DocumentChatRequest(BaseModel):
    chat_id: str
    document_id: str  # Use "all" for multi-doc
    question: str

# 🤖 Document QA (Handles Single + Multi-Doc)
@app.post("/document-chat")
async def document_chat(req: DocumentChatRequest):
    try:
        print(f"📥 Received question: {req.question} for document: {req.document_id}")

        # Handle different types of document ID inputs
        if req.document_id.lower() == "all":
            document_ids = None  # Let QA pipeline handle 'all'
        elif "," in req.document_id:
            # Convert comma-separated string to list
            document_ids = [doc.strip() for doc in req.document_id.split(",")]
        else:
            # Single document
            document_ids = req.document_id

        answer = qa_pipeline.run(req.question, document_ids)

        print("✅ Answer generated successfully.")
        return {"answer": answer}

    except Exception as e:
        print(f"❌ QA Error: {str(e)}")
        return JSONResponse(status_code=500, content={"error": f"Question Answer Pipeline Error: {str(e)}"})
    
# 📘 Summary of All PDFs
@app.get("/summarize_all")
async def summarize_all_pdfs():
    try:
        summary = qa_pipeline.summarize_all_documents()
        return {"summary": summary}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Summary Generation Error: {str(e)}"})

UPLOAD_DIR = "uploaded_excels"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/excel/upload/")
def upload_excel(file: UploadFile = File(...)):
    try:
        file_ext = os.path.splitext(file.filename)[-1]
        if file_ext.lower() not in [".xlsx", ".xls", ".csv"]:
            raise HTTPException(status_code=400, detail="Only Excel or CSV files are allowed.")

        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        return {"file_path": file_path, "message": "Upload successful."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@app.post("/excel/ask/")
def ask_excel_question(file_path: str = Form(...), question: str = Form(...)):
    try:
        df = FileLoader.load_excel_data(file_path)

        # Simple built-in Q&A logic: total, average, etc.
        question = question.lower()
        if "total" in question or "sum" in question:
            numeric_cols = df.select_dtypes(include='number').columns
            if not numeric_cols.any():
                raise HTTPException(status_code=400, detail="No numeric columns found.")
            answer = df[numeric_cols].sum().to_string()
        elif "average" in question or "mean" in question:
            numeric_cols = df.select_dtypes(include='number').columns
            if not numeric_cols.any():
                raise HTTPException(status_code=400, detail="No numeric columns found.")
            answer = df[numeric_cols].mean().to_string()
        else:
            raise HTTPException(status_code=400, detail="Question not understood.")

        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/excel/plot/")
def plot_excel_data(file_path: str = Form(...), question: str = Form(...)):
    try:
        df = FileLoader.load_excel_data(file_path)
        plot_gen = PlotGenerator(df)
        base64_img = plot_gen.generate_plot(question)
        return {"image_base64": base64_img, "message": "Plot generated successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 🚦 Health Check Endpoint
@app.get("/")
async def root():
    return {"message": "✅ SecureDocAI Backend is running"}
