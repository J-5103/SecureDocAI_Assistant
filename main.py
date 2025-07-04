from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.endpoints import upload, question, plot

app = FastAPI(
    title="SecureDocAI Backend",
    description="FastAPI backend for Secure Document Q&A and Plotting",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type"],
)

app.include_router(upload.router, prefix="/api", tags=["Upload"])
app.include_router(question.router, prefix="/api", tags=["Question Answering"])
app.include_router(plot.router, prefix="/api", tags=["Plotting"])

@app.get("/")
def read_root():
    return {"message": "SecureDocAI backend is running."}