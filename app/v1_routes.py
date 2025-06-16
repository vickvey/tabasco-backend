import shutil

from fastapi import APIRouter, Form, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse

from .models import DisambModel
from .services import TextPreprocessor
from .config.settings import UPLOAD_FOLDER
from .utils import (
    allowed_file,
    read_file_text,
    ApiResponse,
    ensure_uploaded_file_exists
)

router = APIRouter()

@router.post('/upload')
async def upload_file(file: UploadFile = File(...)) -> JSONResponse:
    """
    If the file is PDF, then extract the text from it and save it as TXT file and delete the original file.
    If the file is TXT, then save it directly.
    """
    if not file.filename or not allowed_file(file.filename):
        raise HTTPException(status_code=400, detail="File type not allowed; only PDF or TXT are accepted.")

    # Save uploaded file temporarily
    file_location = UPLOAD_FOLDER / file.filename
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # Extract text from the temporary file
        text = read_file_text(file_location)

        # Save extracted text as a .txt file
        text_filename = file_location.with_suffix('.txt')
        with open(text_filename, "w", encoding="utf-8") as f:
            f.write(text)

    finally:
        # Clean up: delete the original uploaded file
        if file_location.exists():
            file_location.unlink()

    return ApiResponse.success(
        message="File uploaded and processed successfully",
        data={
            "filename": text_filename.name,
            "text_length": len(text),
            "upload_path": str(text_filename),
        },
    )

@router.post("/top-nouns")
async def get_n_top_nouns_freq(filename: str = Form(...), top_n: int = Form(50)) -> JSONResponse:
    """
    Return the top-N most frequent nouns with their frequency in the uploaded file.
    """
    if 0 >= top_n > 200:
        raise HTTPException(status_code=400, detail="top_n must be in range [0, 200]")

    file_path = ensure_uploaded_file_exists(filename)
    if not file_path.is_file():
        raise HTTPException(status_code=404, detail=f"File '{filename}' not found.")

    text_content = read_file_text(file_path)
    nouns = TextPreprocessor.extract_top_n_nouns_with_frequency(text_content, top_n)
    return ApiResponse.success(message="Noun list retrieved with frequency", data={"nouns": nouns})

@router.post("/target-matrix")
async def target_matrix(
        filename: str= Form(...),
        target_word: str = Form(...),
        frequency_limit: int = Form(100),
        disamb_model: DisambModel = Depends(get_disamb_model)
) -> JSONResponse:
    pass




# TODO: [START HERE]