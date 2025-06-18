from fastapi import FastAPI
from contextlib import asynccontextmanager
from nltk.corpus import stopwords, wordnet as wn
from .utils import (
    ApiResponse, 
    ensure_nltk_data
)
from .v1_routes import router as api_v1_router
from .config import settings

@asynccontextmanager
async def lifespan(app_: FastAPI):
    # TODO: Add Logger

    # Download NLTK Resources once
    ensure_nltk_data()

    # Load stopwords once
    app.state.stop_words = set(stopwords.words("english"))

    # Build ALL_NOUNS set once
    all_nouns = set()
    for synset in wn.all_synsets('n'):
        for lemma in synset.lemmas():
            all_nouns.add(lemma.name().lower())
    app.state.all_nouns = all_nouns

    # TODO: Add settings


    yield


def init_routers(app_: FastAPI):
    app_.include_router(api_v1_router, prefix='/api/v1')


def create_app() -> FastAPI:
    app_ = FastAPI(
        # Basic
        title=settings.PROJECT_NAME,
        summary="A FastAPI REST API for detecting intra-domain ambiguities",
        description="A FastAPI REST API for detecting intra-domain ambiguities",
        version=settings.RELEASE_VERSION,

        # Docs config depending upon ENVIRONMENT
        docs_url=None if settings.ENVIRONMENT == "production" else "/docs", 
        redoc_url=None if settings.ENVIRONMENT == "production" else "/redoc",
        
        # Configuring Lifespan
        lifespan=lifespan
    )

    @app_.get('/')
    def read_root():
        return ApiResponse.success('Welcome to TABASCO FastAPI')

    init_routers(app_=app_)
    return app_

# App singleton instance
app = create_app()
