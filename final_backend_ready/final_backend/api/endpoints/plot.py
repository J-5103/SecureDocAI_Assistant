from fastapi import APIRouter, HTTPException, Query
from src.pipeline.plot_pipeline import PlotGenerationPipeline

router = APIRouter()
plot_pipeline = PlotGenerationPipeline()

@router.get("/")
def generate_plot(question: str = Query(...)):
    try:
        image_base64 = plot_pipeline.generate_plot(question)
        return {"image_base64": image_base64}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
