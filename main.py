import uvicorn

# from app.config.settings import settings # TODO: add me
from app.server import app

if __name__ == "__main__":
    uvicorn.run(
        app=app,
        port=8000,
        host="0.0.0.0",
        # log_level=settings.LOG_LEVEL, # TODO: add me
        # reload=True if settings.ENVIRONMENT != "production" else False, # TODO: change me
        workers=1
    )
