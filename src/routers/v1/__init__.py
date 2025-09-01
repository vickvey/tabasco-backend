from fastapi import APIRouter
from .general import router as general_router
from .ambiguities import router as ambiguities_router
from .reports import router as reports_router
# from .download import router as download_router

router = APIRouter(tags=['v1'])
router.include_router(general_router, prefix="/general")
router.include_router(ambiguities_router, prefix="/ambiguities")
router.include_router(reports_router, prefix="/reports")
# router.include_router(download_router, prefix="/download")
