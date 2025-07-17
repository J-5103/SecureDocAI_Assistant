from src.components.file_loader import FileLoader
from src.components.data_extractor import DataExtractor
from src.components.data_cleaner import Datacleaner
from src.components.db_handler import DBHandler
from src.logger import logging
from src.exception import CustomException
import sys

class DocumentPipeline:
    def __init__(self):
        try:
            self.loader = FileLoader()
            self.extractor = DataExtractor()
            self.cleaner = Datacleaner()
            self.db = DBHandler()
            logging.info("Document Pipeline initialized")
        except Exception as e:
            raise CustomException(e,sys)

    def run(self,file_path:str):
        try:
            logging.info(f"Loading file:{file_path}")       
            raw_text = self.loader.load(file_path)

            logging.info("Extracting text from file.") 
            extracted_text = self.extractor.extract(raw_text)

            logging.info("Cleaning extracted text.")
            cleaned_text = self.cleaner.clean(extracted_text)

            logging.info("Saving cleaned text to DB.")
            self.db.save_to_database(cleaned_text)

            logging.info("Document processing complete.")
            return cleaned_text
        
        except Exception as e:
            raise CustomException(e,sys)
