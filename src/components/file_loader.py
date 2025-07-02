import docx.document
import pandas as pd
import pdfplumber
import os
import docx

def load_pdf_text(file_path):   ###  Load PDF text
    text =""
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text or ""
        return text.strip()
    except Exception as e:
        raise Exception(f"Failed to load PDF file : {e}")
    
    
def load_docx_text(file_path): #### Load DOCX text
    try:
        doc=docx.document(file_path)
        return "\n".join([para.text for para in doc.paragraphs])
    except Exception as e:    
        raise Exception(f"Failed to load DOCX file : {e}")
    

def load_excel_data(file_path): #### load Excel Data 
    try:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            df =pd.read_excel(file_path , engine="openpyxl")
        return df
    except Exception as e:
        raise Exception(f"Failed To load Excel : {e}")        


