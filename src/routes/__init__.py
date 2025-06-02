from fastapi import APIRouter
from .ambiguous_routes import router as ambiguous_router

router = APIRouter()
router.include_router(ambiguous_router, prefix="/ambiguities", tags=["ambiguities"])