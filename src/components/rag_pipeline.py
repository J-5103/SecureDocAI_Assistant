import sys
from  langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import huggingface_pipeline
from langchain.chains import retrieval_qa
from transformers import pipeline 
import pandas as pd

from src.logger import logging
from src.exception import CustomException
from src.components.db_handler import DBHandler

class RAGPipeline :
    def __init__(self , db_url =  "postgresql://postgres:5103@localhost:5432/SecureDocAI"):
        try:
            self.db = DBHandler(db_url=db_url)
            logging.info("RAGPipeline : DBHandler initialized.")
        except Exception as e:
            logging.error("RAGPipeline : Failed to initialized DBHandler.")
            raise CustomException(e,sys) 

    def get_all_documents(self):
        try:
            df = self.db.fetch_all("documents")
            if df.empty:
                raise ValueError("No documents found in the database")
            return df
        except Exception as e:
            raise CustomException(e,sys)
        
    def prepare_vector_store(self,texts):
        try:
            embedding_model = HuggingFaceEmbeddings(model_name= "sentence-transformers/all-MiniLM-L6-v2")    
            vectorstore = FAISS.from_texts(texts,embedding_model)
            logging.info("vector stores preparing using FAISS")
            return vectorstore
        except Exception as e:
            raise CustomException(e,sys)
        
    def load_llm(self):
        try:
            pipe = pipeline(
                "text-generation",
                model = "tiiuae/falcon-7b-instruct",
                max_new_tokens=256,
                temtemperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1
            )    
            llm = huggingface_pipeline(pipeline=pipe)
            return llm
        except Exception as e:
            raise CustomException(e,sys)
        
    def build_qa_chain(self,vectorstore):
        try:
            llm = self.load_llm()
            retriever = vectorstore.as_retriever(search_type = "similarity",search_kwargs ={"k":3}) 
            chain = retrieval_qa.from_chain_type(llm=llm , retriever = retriever)
            logging.info("QA chain created successfully.")
            return chain
        except Exception as e:
            raise CustomException(e,sys)

    def ask_question(self,question:str) -> str:
        try:
            df = self.get_all_documents()
            documents = df['clean_text'].dropna().tolist()

            vectorstore = self.prepare_vector_store(documents)
            chain = self.build_qa_chain(vectorstore)

            response = chain.run(question)
            return response
        except Exception as e:
            raise CustomException(e,sys)

if __name__ == "__main__":
    rag = RAGPipeline()
    question = "What is the summary of the document?"
    print("\n🧠 Answer:", rag.ask_question(question))
        


    
