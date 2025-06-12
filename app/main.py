import logging
from typing import cast
from fastapi import FastAPI
from contextlib import asynccontextmanager
import uvicorn
import nltk

from app.logger import configure_logging
from app.settings import settings
from app.types import AppState
from app.utils.api_response import ApiResponse
from app.routes import router as api_router


def ensure_nltk_data():
    for resource in ['punkt', 'stopwords', 'averaged_perceptron_tagger']:
        try:
            nltk.data.find(f'tokenizers/{resource}')
        except LookupError:
            nltk.download(resource)

@asynccontextmanager
async def lifespan(fastapi_app: FastAPI):
    # Configure custom logger
    configure_logging(log_level=settings.LOG_LEVEL, log_file=settings.LOG_FILE_PATH if settings.LOG_TO_FILE else None)

    # Sync Uvicorn loggers
    for logger_name in ("uvicorn", "uvicorn.access"):
        logging.getLogger(logger_name).handlers = []
        logging.getLogger(logger_name).propagate = True

    # Log startup
    logging.getLogger(__name__).info(
        f"Starting TABASCO FastAPI application (env: {settings.RUN_ENV}, port: {settings.PORT})")

    # Download NLTK Resources once
    ensure_nltk_data()

    # Tell the type-checker that .state is an AppState
    state = cast(AppState, fastapi_app.state)
    state.settings = settings
    yield


app = FastAPI(
    title="TABASCO FastAPI",
    description="A FastAPI REST API for detecting intra-domain ambiguities",
    version="1.0.0",
    lifespan=lifespan
)


@app.get('/')
def read_root():
    return ApiResponse.success('Welcome to TABASCO FastAPI')


app.include_router(api_router, prefix="/api/v1")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=settings.PORT, log_level="info")
