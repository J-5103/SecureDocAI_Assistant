from fastapi import APIRouter, HTTPException, Query
from src.pipeline.question_answer_pipeline import QuestionAnswerPipeline
from src.utils.synonym_expander import SynonymExpander
import os

router = APIRouter()
qa_pipeline = QuestionAnswerPipeline()
expander = SynonymExpander()

@router.get("/ask/")
def ask_question(
    question: str = Query(..., description="User question to be answered"),
    chat_id: str = Query(..., description="Chat ID to scope the search"),
    document_id: str = Query(None, description="Optional specific document filename within chat")
):
    try:
        if not chat_id:
            raise HTTPException(status_code=400, detail="chat_id is required.")

        doc_id = document_id.strip().replace(".pdf", "") if document_id else None

        # ✅ Expand query using synonym expander
        expanded_query = expander.find_similar_words(question)

        # ✅ Run question-answer pipeline with chat_id context
        answer, source_docs = qa_pipeline.run(
            question=expanded_query,
            chat_id=chat_id,
            document_id=doc_id,
            return_sources=True
        )

        # ✅ Format source citations
        citations = []
        for doc in source_docs:
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page_number", "?")
            citations.append(f"{source} (Page {page})")

        return {
            "question": question,
            "expanded_question": expanded_query,
            "chat_id": chat_id,
            "document_id": document_id or "All documents in this chat",
            "answer": answer,
            "citations": citations
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"QA error: {str(e)}")
