import pandas as pd
from typing import Union

class DataExtractor:
    def __init__(self):
        pass

    def clean_text_data(self, raw_text: str) -> str:
        """
        Cleans and normalizes plain text by:
        - Removing leading/trailing whitespace
        - Removing empty lines

        Args:
            raw_text (str): The raw text input

        Returns:
            str: Cleaned text
        """
        if not isinstance(raw_text, str):
            return ""

        lines = raw_text.splitlines()
        cleaned = [line.strip() for line in lines if line.strip()]
        return "\n".join(cleaned)

    def clean_table_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans tabular data by:
        - Removing fully empty rows/columns
        - Filling NaN with blank
        - Stripping column names

        Args:
            df (pd.DataFrame): Raw DataFrame

        Returns:
            pd.DataFrame: Cleaned DataFrame
        """
        if not isinstance(df, pd.DataFrame):
            return pd.DataFrame()

        df.dropna(axis=0, how='all', inplace=True)
        df.dropna(axis=1, how='all', inplace=True)
        df.fillna("", inplace=True)
        df.columns = df.columns.str.strip()

        return df

    def dataframe_to_text(self, df: pd.DataFrame, max_row: int = 10) -> str:
        """
        Converts a DataFrame to a plain text string (max N rows) for embedding or LLM input.

        Args:
            df (pd.DataFrame): DataFrame to convert
            max_row (int): Max number of rows to include

        Returns:
            str: String representation of the DataFrame
        """
        try:
            if df.empty:
                return "No table data found."

            limited_df = df.head(max_row)
            return limited_df.to_string(index=False)
        except Exception as e:
            return f"Error formatting table: {e}"