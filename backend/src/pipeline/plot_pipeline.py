from src.components.plot_generator import PlotGenerator

class PlotGenerationPipeline:
    def __init__(self):
        print("📊 Initializing Plot Pipeline using real PlotPipeline")
        self.plot_pipeline = PlotGenerator()  # ✅ Use your actual plot pipeline

    def run(self, question):
        try:
            plot_output = self.plot_pipeline.generate_plot(question)
            return plot_output
        except Exception as e:
            return f"Error: {str(e)}"


def generate_plot(question: str):
    pipeline = PlotGenerationPipeline()
    return pipeline.run(question)
