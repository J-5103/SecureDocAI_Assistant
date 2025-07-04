from src.components.rag_pipeline import RAGPipeline
from src.exception import CustomException
import sys

class QuestionAnswerPipeline:
    def __init__(self):
        try:
            self.qa = RAGPipeline()
        except Exception as e:
            raise CustomException(e,sys)

    def run(self,question:str) -> str:
        try:
            return self.qa.ask_question(question)
        except Exception as e:
            raise CustomException(e,sys)        