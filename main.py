from fastapi import FastAPI
from routes.ambiguous_routes import router as ambiguous_router

app = FastAPI(
    title="TABASCO FastAPI",
    description="A modular FastAPI version of TABASCO for detecting intra-domain ambiguities",
    version="1.0.0"
)

# Include our modular routes under the '/api' prefix.
app.include_router(ambiguous_router, prefix="/api")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
