import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
from src.logger import logging
from src.exception import CustomException
import sys

class DBHandler:
    def __init__(self,db_url =  "postgresql://postgres:5103@localhost:5432/SecureDocAI"):
        """
        Initialize the database engine.
        """

        try:
            self.engine = create_engine(db_url)
            logging.info("Database Engine Created Successfully.")
        except Exception as e:
            logging.error("Error Creating Database Engine.")
            raise CustomException(e,sys)

    def insert_dataframe(self, df : pd.DataFrame, table_name :str):
        """
        Insert a DataFrame into a specified SQL table.
        """

        try:
            df.to_sql(table_name,con=self.engine , if_exists='append',index=False)
            logging.info("inserted data into documents")
        except SQLAlchemyError as e:
            logging.error("faild to insert data into documents")
            raise CustomException(e,sys)

    def fetch_all(self,table_name : str) -> pd.DataFrame:
        """
        Fetch all rows from the specified SQL table.
        """
        try:
            result = pd.read_sql_table(table_name,con = self.engine)
            logging.info("Fetch all data from documents")
            return result
        except SQLAlchemyError as e:
            logging.error("Failed to fetch data from documents")
            raise CustomException(e,sys)

    def fetch_by_condition(self,table_name:str,condition :str) -> pd.DataFrame:
        """
        Fetch rows with a specific SQL condition (e.g. WHERE clause).
        """

        try:
            query = f"SELECT * FROM documents WHERE {condition};"
            result = pd.read_sql_query(query,con=self.engine)
            logging.info("fetched all data from documents with condition") 
            return result
        except SQLAlchemyError as e:
            logging.error("Failed to fetch data from documents with condition")
            raise CustomException(e,sys)             

                
    
# db_url = "postgresql://postgres:5103@localhost:5432/SecureDocAI"

# db = DBHandler(db_url=db_url)

# data = {
#     "file_name": ["chatgpt_report.pdf", "summary_notes.pdf"],
#     "file_type": ["pdf", "pdf"],
#     "content": [
#         "This is the extracted content of the report.",
#         "These are notes extracted from the PDF."
#     ],
#     "clean_text": [
#         "Cleaned version of report content.",
#         "Cleaned summary notes from PDF."
#     ]
# }

# df = pd.DataFrame(data)

# db.insert_dataframe(df, "documents")

# documents_df = db.fetch_all("documents")

# print("\n Documents Table:")
# print(documents_df)