import re 
import pandas as pd
from typing import Union , List

class Datacleaner:
    def __init__(self):
        pass
    def clean_text(self,text:str) -> str:
        """
        Clean raw text by removing unwanted characters, extra spaces, etc.
        """

        if not isinstance(text,str):
            return ""
        
        text =re.sub(r'\s+', ' ',text)
        text = re.sub(r'[^\x00-\x7F]+', ' ', text)
        text = text.strip()
        return text
    
    def clean_dataframe(self,df:pd.DataFrame) -> pd.DataFrame:
        """
        Clean a pandas DataFrame by stripping spaces, removing empty rows, etc.
        """

        df.dropna(how="all",inplace=True)
        for col in df.columns:
            df[col]=df[col].apply(lambda x : str(x).strip() if pd.notnull(x) else "")
        return df
    
    def process_text_list(self,text_list: List[str]) -> str:
        """
        Combine and clean a list of texts into a single cleaned paragraph.
        """

        cleaned = [self.clean_text(txt) for txt in text_list if txt.strip()]
        return " ".join(cleaned)
    
    def structure_document(self,raw_data:Union[str,pd.DataFrame , List[str]]) -> dict:
        """
        Convert cleaned text or table into a dictionary structure for storage or RAG.
        """

        if isinstance(raw_data,pd.DataFrame):
            raw_data = self.clean_dataframe(raw_data)
            return {"type" : "table", "data" : raw_data.to_dict(orient = 'records')}
        elif isinstance(raw_data , list):
            return {"type" : "text" , "data" : self.process_text_list(raw_data)}
        elif isinstance(raw_data,str):
            return {"type" : "text" , "data" : self.clean_text(raw_data) }
        else:
            return {"type" : "unknown" , "data" : ""}
        
        

# if __name__ == "__main__":
#     cleaner = Datacleaner()

#     raw_text = "  This is  a    raw   text! \nNew line here.   "
#     cleaned = cleaner.clean_text(raw_text)
#     print("Cleaned Text:", cleaned)

#     df = pd.DataFrame({
#         "Column1": [" value1 ", " value2 ", None],
#         "Column2": [" 123 ", None, "   456"]
#     })
#     cleaned_df = cleaner.clean_dataframe(df)
#     print("Cleaned DataFrame:\n", cleaned_df)

    