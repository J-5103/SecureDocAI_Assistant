import sys
import torch
from  langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import huggingface_pipeline
from langchain.chains import retrieval_qa
# from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA

from transformers import pipeline , AutoTokenizer, AutoModelForCausalLM
from langchain_huggingface import HuggingFacePipeline
import pandas as pd

from src.logger import logging
from src.exception import CustomException
from src.components.db_handler import DBHandler

class RAGPipeline :
    def __init__(self , db_url =  "postgresql://postgres:5103@localhost:5432/SecureDocAI"):
        try:
            self.db = DBHandler(db_url=db_url)
            self.device = torch.device("cpu")
            self.pipe = self.load_llm()
            # self.device = 0 if torch.cuda.is_available() else -1
            logging.info(f"RAGPipeline : DBHandler initialized.")
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
            print("Device set to use cpu")
            model_id = "google/flan-t5-base" # ✅ Use a light model (adjust as needed)

            # tokenizer = AutoTokenizer.from_pretrained(model_id)
            # model = AutoModelForCausalLM.from_pretrained(model_id)

            self.pipe = pipeline(
                "text2text-generation",
                model="google/flan-t5-base",
                tokenizer="google/flan-t5-base",
                max_new_tokens=256,
                device=-1  # CPU
            )
            return HuggingFacePipeline(pipeline=self.pipe)
        except Exception as e:
            raise CustomException(e, sys)
        
    def build_qa_chain(self, vectorstore):
        try:
            retriever = vectorstore.as_retriever()
            llm = self.load_llm()

            chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
            return chain

        except Exception as e:
            raise CustomException(e, sys)


    def ask_question(self, question: str) -> str:
        try:
            print("🧠 Device set to CPU")

            df = self.get_all_documents()
            documents = df['clean_text'].dropna().tolist()
            if not documents:
                return "No documents found."

            # Step 2: Vector store and retriever
            vectorstore = self.prepare_vector_store(documents)
            retriever = vectorstore.as_retriever()
            docs = retriever.invoke(question) 

            # Step 3: Format context and prompt
            context = "\n\n".join([doc.page_content for doc in docs])
            prompt = f"Use the following context to answer the question.\n\nContext:\n{context}\n\nQuestion: {question}"

            # Step 4: Generate answer
            output = self.pipe.invoke(prompt) 
            return output.strip()

        except Exception as e:
            raise CustomException(e, sys)


# if __name__ == "__main__":
#     try:
#         question = "What is the summary of the document?"
#         rag = RAGPipeline()
#         print(f"\n❓ Question: {question}")
#         answer = rag.ask_question(question)
#         print("\n🧠 Answer:", answer)
#     except Exception as e:
#         print("❌ An error occurred while answering the question:", e)
