# main.py
from fastapi import FastAPI
from routes.main_routes import router as api_router
from utils.file_utils import setup_directories
from utils.model_config import get_disamb_model

app = FastAPI(
    title="TABASCO FastAPI", 
    description="A modular FastAPI version of TABASCO for detecting intra-domain ambiguities", 
    version="1.0.0"
)

@app.on_event("startup")
async def startup_event():
    setup_directories()
    app.state.disamb_model = get_disamb_model()

app.include_router(api_router, prefix="/api")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)