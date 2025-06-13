import nltk
from fastapi import FastAPI
from contextlib import asynccontextmanager

from app.utils import ApiResponse
from app.v1_routes import router as api_v1_router
from app.settings import settings


def ensure_nltk_data():
    for resource in ['punkt', 'stopwords', 'averaged_perceptron_tagger']:
        try:
            nltk.data.find(f'tokenizers/{resource}')
        except LookupError:
            nltk.download(resource)


@asynccontextmanager
async def lifespan(app_: FastAPI):
    # TODO: Add Logger
    """
    # Configure custom logger
    # configure_logging(log_level=settings.LOG_LEVEL, log_file=settings.LOG_FILE_PATH if settings.LOG_TO_FILE else None)

    # Sync Uvicorn loggers
    # for logger_name in ("uvicorn", "uvicorn.access"):
    #     logging.getLogger(logger_name).handlers = []
    #     logging.getLogger(logger_name).propagate = True

    # Log startup
    # logging.getLogger(__name__).info(
    #     f"Starting TABASCO FastAPI application (env: {settings.RUN_ENV}, port: {settings.PORT})")
    """

    # Download NLTK Resources once
    ensure_nltk_data()

    # TODO: Add settings
    """
    # Tell the type-checker that .state is an AppState
    # state = cast(AppState, app_.state)
    # state.settings = settings
    """
    yield


def init_routers(app_: FastAPI):
    app_.include_router(api_v1_router, prefix='/api/v1')


def create_app() -> FastAPI:
    app_ = FastAPI(
        title=settings.PROJECT_NAME,
        summary="A FastAPI REST API for detecting intra-domain ambiguities",
        description="A FastAPI REST API for detecting intra-domain ambiguities",
        docs_url=None if settings.ENVIRONMENT == "production" else "/docs",
        redoc_url=None if settings.ENVIRONMENT == "production" else "/redoc",
        version=settings.RELEASE_VERSION,
        lifespan=lifespan
    )

    @app_.get('/')
    def read_root():
        return ApiResponse.success('Welcome to TABASCO FastAPI')

    init_routers(app_=app_)
    return app_


# App singleton instance
app = create_app()
