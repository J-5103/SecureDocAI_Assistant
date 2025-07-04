from src.components.plot_generator import PlotGenerator
from src.exception import CustomException
import sys

class PlotPipeline:
    def __init__(self):
        try:
            self.generator = PlotGenerator()
        except Exception as e:
            raise CustomException(e,sys)

    def run(self,question:str) -> str:
        try:
            return self.generator.generate_plot(question)
        except Exception as e:
            raise CustomException(e,sys)
        
        
                