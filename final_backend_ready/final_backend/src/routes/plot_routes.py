from fastapi import APIRouter

router = APIRouter()

@router.get("/visualizations")
async def get_plots():
    return {"plots": []}
