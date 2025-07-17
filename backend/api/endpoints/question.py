from fastapi import APIRouter, HTTPException, Query
from src.pipeline.question_answer_pipeline import QuestionAnswerPipeline

router = APIRouter()
qa_pipeline = QuestionAnswerPipeline()

@router.get("/ask/")
def ask_question(question: str = Query(...)):
    try:
        answer = qa_pipeline.answer_question(question)
        return {"question": question, "answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
