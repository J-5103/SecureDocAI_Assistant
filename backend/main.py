# main.py



# import psycopg2
# from dotenv import load_dotenv
# import os

# # Load environment variables from .env
# load_dotenv()

# # Fetch variables
# USER = os.getenv("user")
# PASSWORD = os.getenv("password")
# HOST = os.getenv("host")
# PORT = os.getenv("port")
# DBNAME = os.getenv("dbname")

# # Connect to the database
# try:
#     connection = psycopg2.connect(
#         user=USER,
#         password=PASSWORD,
#         host=HOST,
#         port=PORT,
#         dbname=DBNAME
#     )
#     print("Connection successful!")
    
#     # Create a cursor to execute SQL queries
#     cursor = connection.cursor()
    
#     # Example query
#     cursor.execute("SELECT NOW();")
#     result = cursor.fetchone()
#     print("Current Time:", result)

#     # Close the cursor and connection
#     cursor.close()
#     connection.close()
#     print("Connection closed.")

# except Exception as e:
#     print(f"Failed to connect: {e}")

    

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import shutil
import os

# Import pipelines
from src.pipeline.question_answer_pipeline import answer_question
from src.pipeline.plot_pipeline import generate_plot
from src.pipeline.question_answer_pipeline import QuestionAnswerPipeline

qa_pipeline = QuestionAnswerPipeline()

app = FastAPI(
    title="SecureDocAI Backend",
    description="Secure Offline AI-Powered Document Management and Conversational Assistant",
    version="1.0.0"
)

UPLOAD_FOLDER = "uploaded_docs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 🗂️ PDF Upload API
@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")

    file_location = os.path.join(UPLOAD_FOLDER, file.filename)
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {"message": "PDF uploaded successfully.", "filename": file.filename}


# 🧠 Ask Question API (GET)
@app.get("/ask")
def ask_question(question: str):
    answer = qa_pipeline.get_answer(question)
    return {"question": question, "answer": answer}


# 📊 Generate Plot API (POST with JSON Body)

class PlotRequest(BaseModel):
    question: str

@app.post("/generate_plot")
async def generate_dynamic_plot(request: PlotRequest):
    try:
        plot_path = generate_plot(request.question)
        return {"message": "Plot generated successfully.", "plot_path": plot_path}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


# 🗒️ List Uploaded PDFs API
@app.get("/list_documents")
async def list_documents():
    files = os.listdir(UPLOAD_FOLDER)
    pdfs = [f for f in files if f.endswith(".pdf")]
    return {"documents": pdfs}

@app.get("/")
async def root():
    return {"message": "SecureDocAI Backend is running"}
