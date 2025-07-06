import asyncio
import shutil
from fastapi import APIRouter, Form, HTTPException,Request, UploadFile, File
from fastapi.responses import JSONResponse
from .services import (
    extract_top_n_nouns_with_frequency,
    build_target_word_similarity_matrix,
    suggest_num_clusters_with_data
)
from .config import settings
from .utils import (
    allowed_file,
    read_file_text,
    ApiResponse,
    ensure_uploaded_file_exists
)

# Mount path configs from settings
PROJECT_ROOT = settings.PROJECT_ROOT
UPLOAD_FOLDER = settings.UPLOAD_FOLDER
SUMMARY_FOLDER = settings.SUMMARY_FOLDER
DETAILED_FOLDER = settings.DETAILED_FOLDER
# LOG_DIR = settings.LOG_DIR # TODO: Complete this

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

# INFO: This is dev only route, Remove in production
@router.get('/view-txt-files')
async def get_uploaded_text_filenames() -> JSONResponse:
    filenames = []
    for file in UPLOAD_FOLDER.iterdir():
        if file.is_file() and file.suffix == '.txt':
            filenames.append(file.name)
    return JSONResponse(content={"filenames": filenames})

@router.post("/top-nouns")
async def get_n_top_nouns_freq(
    request: Request,
    filename: str = Form(...),
    top_n: int = Form(50)
) -> JSONResponse:
    """
    Return the top-N most frequent nouns with their frequency in the uploaded file.
    """
    if not (0 < top_n <= 200):
        raise HTTPException(status_code=400, detail="top_n must be in range (0, 200]")

    file_path = ensure_uploaded_file_exists(filename)
    if not file_path.is_file():
        raise HTTPException(status_code=404, detail=f"File '{filename}' not found.")

    text_content = read_file_text(file_path)

    stop_words = request.app.state.stop_words
    all_nouns = request.app.state.all_nouns

    nouns = await asyncio.to_thread(
        extract_top_n_nouns_with_frequency,
        text_content,
        top_n,
        stop_words,
        all_nouns
    )
    return ApiResponse.success(message="Noun list retrieved with frequency", data={"nouns": nouns})


@router.post("/target-matrix")
async def get_target_matrix_api(
    request: Request,
    filename: str = Form(...),
    target_word: str = Form(...),
    frequency: int = Form(...)
) -> JSONResponse:
    """
    Build a similarity matrix for sentences containing the target word in the given file,
    and return the optimal number of clusters and elbow plot data.
    """
    # Validate file
    file_path = ensure_uploaded_file_exists(filename)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"File '{filename}' not found.")

    # Read text
    text_content = read_file_text(file_path)
    if not text_content:
        raise HTTPException(status_code=400, detail="The file contains no readable text.")

    try:
        matrix, sentences = build_target_word_similarity_matrix(
            text_content=text_content,
            target_word=target_word,
            model=request.app.state.disamb_model,  # Assuming disamb_model is stored in app state
            frequency_limit=frequency
        )

        opt_k, k_range, wcss = suggest_num_clusters_with_data(matrix)

        return ApiResponse.success(
            message="Target matrix generated successfully.",
            data={
                "target_word": target_word,
                "sentence_count": len(sentences),
                "matrix_shape": list(matrix.shape),
                "optimal_k": int(opt_k),
                "k_range": [int(k) for k in k_range],
                "wcss": [float(w) for w in wcss],
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Matrix generation failed: {str(e)}")

# Club similar sentences

# TODO: [START HERE]