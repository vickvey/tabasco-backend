from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from src.config import settings

# Mount path configs from settings
PROJECT_ROOT = settings.PROJECT_ROOT
UPLOAD_FOLDER = settings.UPLOAD_FOLDER
SUMMARY_FOLDER = settings.SUMMARY_FOLDER
DETAILED_FOLDER = settings.DETAILED_FOLDER
# LOG_DIR = settings.LOG_DIR # TODO: Complete this

router = APIRouter()

# === Download Endpoints ===
@router.get("/summary-report/{cluster_number}")
async def download_summary_report_for_a_cluster(cluster_number: int) -> FileResponse:
    """
    Download the summary text file for a given cluster index.
    """
    if cluster_number < 0:
        raise HTTPException(status_code=400, detail="cluster_number must be non-negative")
    file_path = SUMMARY_FOLDER / f"summary_text_{cluster_number}.txt"
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Summary file not found.")
    return FileResponse(
        file_path,
        media_type="text/plain",
        filename=f"summary_cluster_{cluster_number}.txt",
    )

@router.get("/detailed-report/{cluster_number}")
async def download_detailed_report_for_a_cluster(cluster_number: int) -> FileResponse:
    """
    Download the detailed text file for a given cluster index.
    """
    if cluster_number < 0:
        raise HTTPException(status_code=400, detail="cluster_number must be non-negative")
    file_path = DETAILED_FOLDER / f"text_{cluster_number}.txt"
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Detailed file not found.")
    return FileResponse(
        file_path,
        media_type="text/plain",
        filename=f"detailed_cluster_{cluster_number}.txt",
    )
