import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import base64
import io
import re

class PlotGenerator:
    def __init__(self,dataframe :pd.DataFrame):
        self.df = dataframe
        sns.set(style="whitegrid")

    def generate_plot(self,question: str) -> str:
        question = question.lower()
        if "bar" in question:
            return self._generate_bar_plot(question)
        elif "line" in question:
            return self._generate_line_plot(question)
        elif "scatter" in question:
            return self._generate_scatter_plot(question)
        elif "histogram" in question:
            return self._generate_histogram(question)
        elif "box" in question:
            return self._generate_box_plot(question)
        else:
            raise ValueError("Unsupported plot type in question")
        
    def _extract_columns(self, question: str) -> tuple:

        # Step 1: Extract columns from question text
        tokens = re.findall(r'\b[a-zA-Z_]+\b', question)
        cols = [col for col in tokens if col in self.df.columns]

        # Step 2: If at least 2 valid columns found in question, return them
        if len(cols) >= 2:
            return cols[0], cols[1]

        # Step 3: Fallback — auto-select first 2 numeric columns
        numeric_cols = self.df.select_dtypes(include='number').columns.tolist()
        if len(numeric_cols) >= 2:
            return numeric_cols[0], numeric_cols[1]

        # Step 4: Raise error if not enough valid columns
        raise ValueError("Could not extract two valid columns from question or data.")

    
    def _extract_single_column(self, question: str) -> str:
        tokens = re.findall(r'\b[a-zA-Z_]+\b', question)
        for col in tokens:
            if col in self.df.columns:
                return col
        raise ValueError("Could not extract one valid column.")
        
    def _save_plot_to_base64(self) -> str:
        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf,format='png')
        buf.seek(0)
        image_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        plt.clf()
        return image_base64
    
    def _generate_bar_plot(self,question :str) -> str:
        x,y = self._extract_columns(question)
        plt.figure(figsize=(10,6))
        sns.barplot(data=self.df,x=x,y=y)
        plt.title(f"Bar plot of {y} by {x}")
        # plt.show() 
        return self._save_plot_to_base64()
    
    def _generate_line_plot(self,question : str) -> str:
        x,y = self._extract_columns(question)
        plt.figure(figsize=(10,6))
        sns.lineplot(data=self.df,x=x,y=y)
        plt.title(f"Line plot of {y} by {x}")
        return self._save_plot_to_base64()
    
    def _generate_scatter_plot(self,question:str) -> str:
        x,y = self._extract_columns(question)
        plt.figure(figsize=(10,6))
        sns.scatterplot(data=self.df,x=x,y=y)
        plt.title(f"Scatter plot of {y} by {x}")
        return self._save_plot_to_base64()
    
    def _generate_histogram(self,question : str) -> str:
        col = self._extract_single_column(question)
        plt.figure(figsize=(10,6))
        sns.histplot(data=self.df,x=col)
        plt.title(f"Histogram of {col}")
        return self._save_plot_to_base64()
    
    def _generate_box_plot(self,question : str) -> str:
        col = self._extract_single_column(question)
        plt.figure(figsize=(10,6))
        sns.boxplot(data=self.df,x=col)
        plt.title(f"Box plot of {col}")
        return self._save_plot_to_base64()
    

# df_sample = pd.DataFrame({
#     "year": [2020, 2021, 2022, 2023],
#     "revenue": [100, 150, 200, 250]
# })

# plotter = PlotGenerator(df_sample)
# question = "Generate a bar chart of year and revenue"
# image_base64 = plotter.generate_plot(question)    



            