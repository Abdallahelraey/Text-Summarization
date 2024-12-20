from fastapi import FastAPI
import uvicorn
from TextSummarizer.routes.inferance import summarizer_router  

app = FastAPI()


app.include_router(summarizer_router)

# To run use the following commands
# uvicorn app:app --reload        # Uses default host (127.0.0.1) and port (8000)
# uvicorn app:app --reload --host 0.0.0.0 --port 8080  # Explicit host and port