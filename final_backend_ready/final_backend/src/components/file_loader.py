# src/components/file_loader.py
import os
from typing import Optional

import pandas as pd
import pdfplumber
import docx  # python-docx


class FileLoader:
    # -------- PDF --------
    @staticmethod
    def load_pdf_text(file_path: str) -> str:
        """Extract text from a PDF using pdfplumber."""
        text = ""
        try:
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    text += page.extract_text() or ""
            return text.strip()
        except Exception as e:
            raise Exception(f"Failed to load PDF file: {e}")

    # -------- DOCX --------
    @staticmethod
    def load_docx_text(file_path: str) -> str:
        """Extract text from a DOCX using python-docx."""
        try:
            doc = docx.Document(file_path)
            return "\n".join([para.text for para in doc.paragraphs])
        except Exception as e:
            raise Exception(f"Failed to load DOCX file: {e}")

    # -------- Excel/CSV --------
    @staticmethod
    def load_excel_data(file_path: str) -> pd.DataFrame:
        """
        Load .csv, .xlsx or .xls into a DataFrame.
        - CSV: tries utf-8, utf-8-sig, latin-1, then engine='python'
        - XLSX/XLS: uses pandas.read_excel with engine auto/fallback.
        """
        try:
            ext = os.path.splitext(file_path)[-1].lower()

            if ext == ".csv":
                # Try common encodings first
                for enc in ("utf-8", "utf-8-sig", "latin-1"):
                    try:
                        return pd.read_csv(file_path, encoding=enc)
                    except UnicodeDecodeError:
                        continue
                # Fallback: python engine (more permissive)
                try:
                    return pd.read_csv(file_path, engine="python")
                except Exception as e:
                    raise Exception(f"Failed to read CSV (encoding/format issue): {e}")

            elif ext in (".xlsx", ".xls"):
                # Try default engine first (pandas chooses)
                try:
                    return pd.read_excel(file_path)
                except ImportError as e:
                    # Missing engine packages
                    need = "openpyxl for .xlsx, xlrd (<2.0) for .xls"
                    raise Exception(f"Excel engine not installed. Please install {need}. Error: {e}")
                except ValueError:
                    # Some environments demand explicit engine
                    engine = "openpyxl" if ext == ".xlsx" else "xlrd"
                    try:
                        return pd.read_excel(file_path, engine=engine)
                    except Exception as e2:
                        raise Exception(f"Failed to read Excel with engine '{engine}': {e2}")

            else:
                raise ValueError("Unsupported file format. Allowed: .csv, .xlsx, .xls")

        except Exception as e:
            raise Exception(f"Failed to load Excel/CSV file: {e}")

    # -------- Unified helper --------
    @staticmethod
    def load_file(file_path: str):
        """Dispatch to the correct loader based on file extension."""
        ext = os.path.splitext(file_path)[-1].lower()
        if ext == ".pdf":
            return FileLoader.load_pdf_text(file_path)
        elif ext == ".docx":
            return FileLoader.load_docx_text(file_path)
        elif ext in [".xlsx", ".xls", ".csv"]:
            return FileLoader.load_excel_data(file_path)
        else:
            raise Exception(f"Unsupported file type: {ext}")
