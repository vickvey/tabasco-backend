# app/main.py

from fastapi import FastAPI
from contextlib import asynccontextmanager
from typing import cast

from app.routes import router as api_router
from app.utils.model_config import get_disamb_model
from app.settings import Settings
from app.types import AppState

@asynccontextmanager
async def lifespan(fastapi_app: FastAPI):
    # Tell the type‐checker that .state is an AppState
    state = cast(AppState, fastapi_app.state)

    # Now “state.settings” and “state.disamb_model” are recognized by Pyright/MyPy/IntelliSense
    state.settings = Settings()
    state.disamb_model = get_disamb_model(request=None)
    yield

app = FastAPI(
    title="TABASCO FastAPI",
    description="A FastAPI REST API for detecting intra-domain ambiguities",
    version="1.0.0",
    lifespan=lifespan
)

app.include_router(api_router, prefix="/api/v1")
