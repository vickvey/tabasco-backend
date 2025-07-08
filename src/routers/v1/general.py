import uuid
import asyncio
import shutil
from fastapi import (
    APIRouter, Form, HTTPException, Request, UploadFile, File, Query
)
from fastapi.responses import JSONResponse

from src.services import extract_top_n_nouns_with_frequency
from src.config import settings
from src.utils import (
    allowed_file,
    read_file_text,
    ApiResponse,
    ensure_uploaded_file_exists,
)

SESSION_FOLDER = settings.SESSION_FOLDER
ENVIRONMENT = settings.ENVIRONMENT

router = APIRouter()


@router.post('/upload')
async def upload_file(file: UploadFile = File(...)) -> JSONResponse:
    """
    Accepts a PDF or TXT file, extracts text, stores it in a session folder, and returns session_id + metadata.
    """
    if not file.filename or not allowed_file(file.filename):
        raise HTTPException(status_code=400, detail="Only PDF or TXT files are accepted.")

    # 🔐 Create unique session
    session_id = str(uuid.uuid4())
    session_dir = SESSION_FOLDER / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    uploaded_path = session_dir / file.filename
    with open(uploaded_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # 🧠 Extract and store text
        text = read_file_text(session_id, file.filename)
        text_path = uploaded_path.with_suffix(".txt")

        with open(text_path, "w", encoding="utf-8") as f:
            f.write(text)

    finally:
        if uploaded_path.exists():
            uploaded_path.unlink()

    return ApiResponse.success(
        message="File uploaded and processed successfully.",
        data={
            "session_id": session_id,
            "filename": text_path.name,
            "text_length": len(text),
            "session_dir": str(session_dir)
        }
    )


# ========== Dev-Only Utilities ==========

if ENVIRONMENT != "production":

    @router.get('/view-txt-files')
    async def get_txt_files_for_session(session_id: str = Query(...)) -> JSONResponse:
        """
        🧪 Dev-only: Return all .txt files for a given session_id.
        """
        session_dir = SESSION_FOLDER / session_id
        if not session_dir.exists() or not session_dir.is_dir():
            raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")

        txt_files = [
            file.name
            for file in session_dir.iterdir()
            if file.is_file() and file.suffix.lower() == '.txt'
        ]

        return JSONResponse(content={"filenames": txt_files})


    @router.get("/list-sessions")
    async def list_active_sessions() -> JSONResponse:
        """
        🧪 Dev-only: List all active session IDs (folder names) in the sessions directory.
        """
        try:
            sessions = [
                folder.name
                for folder in SESSION_FOLDER.iterdir()
                if folder.is_dir()
            ]
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to list sessions: {str(e)}")

        return JSONResponse(content={"sessions": sessions})


# ========== Core NLP Route ==========

@router.post("/top-nouns")
async def get_n_top_nouns_freq(
    request: Request,
    session_id: str = Form(...),
    filename: str = Form(...),
    top_n: int = Form(50)
) -> JSONResponse:
    """
    Return the top-N most frequent nouns from a given .txt file in a session.
    """
    if not (0 < top_n <= 200):
        raise HTTPException(status_code=400, detail="top_n must be in range (0, 200]")

    file_path = ensure_uploaded_file_exists(session_id, filename)

    text_content = read_file_text(session_id, filename)
    if not text_content.strip():
        raise HTTPException(status_code=400, detail="The file contains no readable text.")

    stop_words = request.app.state.stop_words
    all_nouns = request.app.state.all_nouns

    nouns = await asyncio.to_thread(
        extract_top_n_nouns_with_frequency,
        text_content,
        top_n,
        stop_words,
        all_nouns
    )

    return ApiResponse.success(
        message="Noun list retrieved with frequency.",
        data={"nouns": nouns}
    )
