import pandas as pd
from typing import Union

def clean_text_data(raw_text:str) -> str:    ## clean Text Data

    """
    Cleans and normalizes plain text.
    Removes extra whitespace, empty lines, etc.
    """

    if not isinstance(raw_text,str):
        return ""
    
    lines = raw_text.splitlines()
    cleaned =[line.strip() for line in lines if line.strip()]
    return "\n".join(cleaned)

def clean_table_data(df :pd.DataFrame) -> pd.DataFrame:    ## Clean Table data

    """
    Cleans and normalizes tabular data:
    - Removes completely empty rows/columns
    - Fills NaNs if needed
    """

    if not isinstance(df , pd.DataFrame):
        return pd.DataFrame()
    
    df.dropna(axis = 0 , how = 'all' , inplace = True)
    df.dropna(axis = 1 , how = 'all' , inplace = True)
    
    df.fillna("",inplace = True)

    df.columns = df.columns.str.strip()

    return df
    
def dataframe_to_text(df:pd.DataFrame, max_row=10) -> str:
    """
    Converts DataFrame to string format for embedding or QA model input.
    Truncates to max_rows for performance.
    """

    try:
        if df.empty:
            return " No table data found."
        limited_df = df.head(max_row)
        return limited_df.to_string(index=False)
    except Exception as e:
        return f"Error formatting table : {e}"