from fastapi import FastAPI, APIRouter, Depends, HTTPException
import uvicorn
import sys
import os
from fastapi.templating import Jinja2Templates
from starlette.responses import RedirectResponse
from fastapi.responses import Response
from TextSummarizer.pipeline.inferance_pipeline import PredictionPipeline


text:str = "What is Text Summarization?"

app = FastAPI()

summarizer_router = APIRouter(
    prefix="/api/v1/summarizer",
    tags=["api_v1", "summarizer"],
)

@summarizer_router.get("/", tags=["authentication"])
async def index():
    return RedirectResponse(url="/docs")



@summarizer_router.get("/train")
async def training():
    try:
        os.system("python main.py")
        return Response("Training successful !!")

    except Exception as e:
        return Response(f"Error Occurred! {e}")
    



@summarizer_router.post("/predict")
async def predict_route(text: str):  
    try:
        obj = PredictionPipeline()
        result = obj.predict(text)
        return {"summary": result}  
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def create_app():
    app = FastAPI()
    app.include_router(summarizer_router)
    return app
    
if __name__ == "__main__":
    app = create_app()
    uvicorn.run(app, host="127.0.0.1", port=8000)