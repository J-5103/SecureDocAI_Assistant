import os
import sys
from src.components.file_loader import FileLoader
from src.components.data_extractor import DataExtractor
from src.components.data_cleaner import Datacleaner
from src.components.db_handler import DBHandler
from src.components.rag_pipeline import RAGPipeline
from src.logger import logging
from src.exception import CustomException
from langchain_community.document_loaders import PyPDFLoader

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

    def create_vectorstore_for_pdf(self, pdf_path):
        """
        Safely create a vectorstore for a single PDF.
        Checks for empty or scanned PDFs before processing.
        """
        try:
            file_name = os.path.basename(pdf_path).replace(".pdf", "")
            vectorstore_path = os.path.join("vectorstores", file_name)

            # Load PDF pages
            loader = PyPDFLoader(pdf_path)
            pages = loader.load_and_split()

            if not pages:
                raise Exception(f"PDF {pdf_path} has no extractable text pages. Possibly scanned or corrupted.")

            # Create vectorstore
            rag = RAGPipeline()
            rag.create_vectorstore(pdf_path)

            logging.info(f"✅ Vectorstore created for {pdf_path}")
        
        except Exception as e:
            logging.error(f"❌ Error in create_vectorstore_for_pdf: {str(e)}")
            raise CustomException(e, sys)

    def run(self, file_path: str, document_id: str):
        """
        Full pipeline: Load -> Extract -> Clean -> Save -> Vectorstore
        """
        try:
            logging.info(f"📄 Loading file: {file_path}")
            raw_content = FileLoader.load_file(file_path)

            if isinstance(raw_content, str):
                raw_text = raw_content
            else:
                # For Excel/CSV dataframes, convert to string (e.g., for RAG or text processing)
                raw_text = raw_content.to_string()

            if not raw_text.strip():
                raise Exception(f"No text found in {file_path}. File might be empty or unsupported format.")

            logging.info("🔍 Extracting text from file.")
            extracted_text = self.extractor.extract(raw_text)

            if not extracted_text.strip():
                raise Exception(f"No text could be extracted from {file_path}. Check content quality.")

            logging.info("🧹 Cleaning extracted text.")
            cleaned_text = self.cleaner.clean(extracted_text)

            logging.info("💾 Saving cleaned text to DB.")
            self.db.save_to_database(cleaned_text)

            logging.info("🔗 Creating or loading FAISS vectorstore for RAG.")
            self.rag_pipeline.create_vectorstore(file_path)

            logging.info("✅ Document processing complete.")
            return {"message": "Document processed and vectorstore created successfully."}

        except Exception as e:
            logging.error(f"❌ Error in DocumentPipeline.run: {str(e)}")
            raise CustomException(e, sys)
