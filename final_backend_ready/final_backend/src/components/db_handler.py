import sys
import psycopg2
import pandas as pd
from src.logger import logging
from src.exception import CustomException
from src.utils.db_utils import get_database_connection

class DBHandler:
    def __init__(self):
        try:
            self.conn = get_database_connection()
            self.cursor = self.conn.cursor()
        except Exception as e:
            raise CustomException(e, sys)

    def insert_metadata(self, filename: str, filepath: str) -> str:
        try:
            document_id = filename.replace(" ", "_").replace(".pdf", "")
            query = "INSERT INTO documents (id, file_name, file_path) VALUES (%s, %s, %s)"
            self.cursor.execute(query, (document_id, filename, filepath))
            self.conn.commit()
            return document_id
        except Exception as e:
            self.conn.rollback()
            raise CustomException(e, sys)

    def get_file_path(self, document_id: str) -> str:
        try:
            query = "SELECT file_path FROM documents WHERE id = %s"
            self.cursor.execute(query, (document_id,))
            result = self.cursor.fetchone()
            if result:
                return result[0]
            else:
                raise Exception("Document not found")
        except Exception as e:
            raise CustomException(e, sys)

    def save_to_database(self, data: dict, document_id: str = "unknown"):
        """
        Save cleaned data (text + table) to DB.
        Args:
            data: dict with 'text' and 'table'
            document_id: document reference
        """
        try:
            text = data.get("text", "")
            table = data.get("table", pd.DataFrame())
            table_text = ""

            if isinstance(table, pd.DataFrame) and not table.empty:
                table_text = table.to_string(index=False)

            query = """
                INSERT INTO processed_documents (document_id, cleaned_text, table_text)
                VALUES (%s, %s, %s)
                ON CONFLICT (document_id) DO UPDATE
                SET cleaned_text = EXCLUDED.cleaned_text,
                    table_text = EXCLUDED.table_text
            """
            self.cursor.execute(query, (document_id, text, table_text))
            self.conn.commit()

            logging.info(f"💾 Saved cleaned data to DB for document: {document_id} (text length={len(text)})")

        except Exception as e:
            self.conn.rollback()
            raise CustomException(e, sys)
