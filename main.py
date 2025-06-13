import uvicorn

from app.settings import settings
from app.server import app

if __name__ == "__main__":
    uvicorn.run(
        app=app,
        port=settings.PORT,
        host=settings.HOST,
        log_level=settings.LOG_LEVEL,
        reload=True if settings.ENVIRONMENT != "production" else False,
        workers=1
    )
