import os
import pandas as pd
import pdfplumber
import docx  # python-docx

class FileLoader:

    @staticmethod
    def load_pdf_text(file_path):  #  Load PDF
        text = ""
        try:
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    text += page.extract_text() or ""
            return text.strip()
        except Exception as e:
            raise Exception(f"Failed to load PDF file: {e}")

    @staticmethod
    def load_docx_text(file_path):  #  Load DOCX
        try:
            doc = docx.Document(file_path)
            return "\n".join([para.text for para in doc.paragraphs])
        except Exception as e:
            raise Exception(f"Failed to load DOCX file: {e}")

    @staticmethod
    def load_excel_data(file_path:str)-> pd.DataFrame:  #  Load Excel/CSV
        try:
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file_path.endswith(".xlsx"):
                df = pd.read_excel(file_path, engine='openpyxl')    
            else:
                raise ValueError("unsupported file formate.")
            return df
        except Exception as e:
            raise Exception(f"Failed to load Excel file: {e}")

    @staticmethod
    def load_file(file_path):  #  Unified loader if needed
        ext = os.path.splitext(file_path)[-1].lower()
        if ext == '.pdf':
            return FileLoader.load_pdf_text(file_path)
        elif ext == '.docx':
            return FileLoader.load_docx_text(file_path)
        elif ext in ['.xlsx', '.xls', '.csv']:
            return FileLoader.load_excel_data(file_path)
        else:
            raise Exception(f"Unsupported file type: {ext}")
        

# from src.components.data_extractor import DataExtractor
# import pandas as pd

# extractor = DataExtractor()

# raw_text = "  Hello World! \n\n This is a test.   \n\n\n"
# print(extractor.clean_text_data(raw_text))

# df = pd.read_csv("src/components/Customers.csv")
# cleaned_df = extractor.clean_table_data(df)
# print(extractor.dataframe_to_text(cleaned_df))        