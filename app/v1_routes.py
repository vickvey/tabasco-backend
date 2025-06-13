import shutil

from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse

from app.settings import UPLOAD_FOLDER
from app.utils import (
    allowed_file,
    read_file_text,
    ApiResponse
)

router = APIRouter()

@router.post('/upload')
async def upload_file(file: UploadFile = File(...)) -> JSONResponse:
    """
    Upload a PDF or TXT file to UPLOAD_FOLDER and return metadata.
    """
    if not file.filename or not allowed_file(file.filename):
        raise HTTPException(status_code=400, detail="File type not allowed; only PDF or TXT are accepted.")

    file_location = UPLOAD_FOLDER / file.filename
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    text = read_file_text(file_location)
    return ApiResponse.success(
        message="File uploaded successfully",
        data={
            "filename": file.filename,
            "text_length": len(text),
            "upload_path": str(file_location),
        },
    )
