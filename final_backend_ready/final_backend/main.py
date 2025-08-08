
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.endpoints import upload, plot, question

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload.router, prefix="/api")
app.include_router(plot.router, prefix="/api")
app.include_router(question.router, prefix="/api")
