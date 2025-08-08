import os
import sys
import pandas as pd
import pdfplumber
from langchain_core.documents import Document
from src.components.file_loader import FileLoader
from src.components.data_extractor import DataExtractor
from src.components.data_cleaner import Datacleaner
from src.components.db_handler import DBHandler
from src.components.rag_pipeline import RAGPipeline
from src.logger import logging
from src.exception import CustomException


class DocumentPipeline:
    def __init__(self):
        try:
            self.extractor = DataExtractor()
            self.cleaner = Datacleaner()
            self.db = DBHandler()
            self.rag_pipeline = RAGPipeline()
            logging.info("✅ Document Pipeline initialized.")
        except Exception as e:
            raise CustomException(e, sys)

    def create_vectorstore_for_pdf(self, pdf_path: str, combined_text: str, chat_id: str, document_id: str):
        try:
            # Use exact document_id (already includes .pdf)
            file_name = document_id.replace(".pdf", "")

            vectorstore_path = os.path.join("vectorstores", chat_id, file_name)
            os.makedirs(vectorstore_path, exist_ok=True)

            logging.info("📖 Extracting individual page text with metadata using pdfplumber...")
            documents = []

            with pdfplumber.open(pdf_path) as pdf:
                for i, page in enumerate(pdf.pages):
                    text = page.extract_text()
                    if text and text.strip():
                        doc = Document(
                            page_content=text.strip(),
                            metadata={
                                "source": file_name,
                                "page_number": i + 1,
                                "chat_id": chat_id
                            }
                        )
                        documents.append(doc)

            if not documents:
                raise Exception("No extractable text found in any PDF pages.")

            logging.info("📦 Passing document list to RAGPipeline for vectorstore creation.")
            self.rag_pipeline.create_vectorstore_from_documents(
                documents=documents,
                vector_store_path=vectorstore_path
            )

            logging.info(f"✅ Vectorstore successfully created at {os.path.abspath(vectorstore_path)}")

        except Exception as e:
            logging.error(f"❌ Error in create_vectorstore_for_pdf: {str(e)}")
            raise CustomException(e, sys)

    def run(self, file_path: str, document_id: str, chat_id: str):
        try:
            logging.info(f"📄 Loading file: {os.path.abspath(file_path)}")
            raw_content = FileLoader.load_file(file_path)

            raw_text = raw_content if isinstance(raw_content, str) else raw_content.to_string()
            if not raw_text.strip():
                raise Exception(f"No text found in {file_path}. File might be empty or unsupported format.")

            logging.info("🔍 Extracting structured data from raw content...")
            extracted_input = {
                "text": raw_text,
                "table": pd.DataFrame()
            }
            extracted_data = self.extractor.extract(extracted_input)

            if not extracted_data["text"].strip():
                raise Exception(f"No text could be extracted from {file_path}. Check content quality.")

            logging.info("🧹 Cleaning extracted data...")
            cleaned_text = extracted_data["text"]
            cleaned_table = extracted_data["table"]
            table_text = self.extractor.dataframe_to_text(cleaned_table) if not cleaned_table.empty else ""

            combined_text = cleaned_text
            if table_text:
                combined_text += "\n\n📊 Table Data:\n" + table_text

            logging.info("💾 Storing cleaned data into database...")
            self.db.save_to_database({
                "text": cleaned_text,
                "table": cleaned_table
            })

            logging.info("🔗 Creating FAISS vectorstore from cleaned text...")
            self.create_vectorstore_for_pdf(file_path, combined_text=combined_text, chat_id=chat_id, document_id=document_id)

            logging.info("✅ Document processing complete for file: %s", document_id)
            return {"message": "Document processed and vectorstore created successfully."}

        except Exception as e:
            logging.error(f"❌ Error in DocumentPipeline.run: {str(e)}")
            raise CustomException(e, sys)
