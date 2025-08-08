import pandas as pd
from typing import Union, Dict

__all__ = ["DataExtractor"]

class DataExtractor:
    def __init__(self):
        pass

    def extract(self, extracted_data: Dict[str, Union[str, pd.DataFrame]]) -> Dict[str, Union[str, pd.DataFrame]]:
        """
        Main method called by pipeline to clean and return extracted text and table data.

        Args:
            extracted_data (dict): Contains raw 'text' and 'table' data

        Returns:
            dict: Cleaned text and cleaned table data
        """
        cleaned = {}

        # Clean text data
        raw_text = extracted_data.get("text", "")
        cleaned["text"] = self.clean_text_data(raw_text)

        # Clean table data
        raw_table = extracted_data.get("table", pd.DataFrame())
        cleaned["table"] = self.clean_table_data(raw_table)

        return cleaned

    def clean_text_data(self, raw_text: str) -> str:
        """
        Cleans and normalizes plain text by:
        - Removing leading/trailing whitespace
        - Removing empty lines
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
        """
        if not isinstance(df, pd.DataFrame):
            return pd.DataFrame()

        df = df.dropna(axis=0, how='all')
        df = df.dropna(axis=1, how='all')
        df = df.fillna("")

        df.columns = [str(col).strip() for col in df.columns]
        return df

    def dataframe_to_text(self, df: pd.DataFrame, max_row: int = 10) -> str:
        """
        Converts a DataFrame to a plain text string (max N rows) for embedding or LLM input.
        """
        try:
            if df.empty:
                return "No table data found."

            limited_df = df.head(max_row)
            return limited_df.to_string(index=False)
        except Exception as e:
            return f"Error formatting table: {e}"
