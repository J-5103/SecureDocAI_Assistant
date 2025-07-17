from src.components.rag_pipeline import RAGPipeline

class QuestionAnswerPipeline:
    def __init__(self):
        print("🔧 Initializing Q&A Pipeline using RAGPipeline")
        self.rag_pipeline = RAGPipeline()  # ✅ Use your actual pipeline

    def run(self, question):
        try:
            answer = self.rag_pipeline.ask_question(question)
            return answer
        except Exception as e:
            return f"Error: {str(e)}"


def answer_question(question: str):
    pipeline = QuestionAnswerPipeline()
    return pipeline.run(question)
