import shutil
from fastapi import (
    APIRouter, HTTPException, UploadFile, File
)
from fastapi.responses import JSONResponse

from src.config import settings
from src.utils import (
    ApiResponse,
)
from pathlib import Path
from src.utils.file_processing import read_file_text, allowed_file
from slugify import slugify

UPLOAD_DIR = settings.UPLOAD_FOLDER
router = APIRouter()


@router.post('/upload')
async def upload_file(file: UploadFile = File(...)) -> JSONResponse:
    if not file.filename or not allowed_file(file.filename):
        raise HTTPException(status_code=400, detail="Only PDF or TXT files are accepted.")

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    original_name = file.filename
    safe_stem = slugify(Path(original_name).stem)
    safe_filename = safe_stem + Path(original_name).suffix.lower()
    uploaded_path = UPLOAD_DIR / safe_filename

    # Save file
    with open(uploaded_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # Extract and store text
        text = read_file_text(uploaded_path)  # pass full path
        text_path = uploaded_path.with_suffix(".txt")
        with open(text_path, "w", encoding="utf-8") as f:
            f.write(text)
    finally:
        # Clean up original
        if uploaded_path.exists():
            uploaded_path.unlink()

    return ApiResponse.success(
        message="File uploaded and processed successfully.",
        data={
            "filename": text_path.name,
            "text_length": len(text),
            "path": str(text_path)
        }
    )



# ========== Dev-Only Utilities ==========

@router.get("/list-uploads")
async def list_uploads() -> JSONResponse:
    """
    Dev-only: List all uploaded (processed) .txt files.
    """
    try:
        files = [f.name for f in UPLOAD_DIR.iterdir() if f.is_file()]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list uploads: {str(e)}")

    return JSONResponse(content={"uploads": files})
