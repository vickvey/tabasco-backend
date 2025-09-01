from fastapi import APIRouter
from .upload import router as upload_router

router = APIRouter(tags=['v2'])

router.include_router(upload_router)
# router.include_router(general_router, prefix="/general")
# router.include_router(ambiguities_router, prefix="/ambiguities")
# router.include_router(reports_router, prefix="/reports")
# router.include_router(download_router, prefix="/download")
