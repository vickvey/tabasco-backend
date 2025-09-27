import json
from fastapi.responses import JSONResponse
from fastapi import Request, Form, HTTPException, APIRouter
from config import settings
from utils.file_processing import ensure_uploaded_file_exists, read_file_text
from utils.api_response import ApiResponse
from config import settings


UPLOAD_FOLDER = settings.UPLOAD_FOLDER

router = APIRouter()

@router.post("/top-nouns")
async def get_n_top_nouns_freq(
    request: Request,
    filename: str = Form(...),
    top_n: int = Form(50)
) -> JSONResponse:
    """
    Return (and cache) top-N most frequent nouns from a .txt file inside session folder.
    If top-nouns.json exists and has enough data, reuse it.
    """
    if not (0 < top_n <= 200):
        raise HTTPException(status_code=400, detail="top_n must be in range (0, 200]")

    # Validate file existence and read contents
    file_path = UPLOAD_FOLDER / filename
    ensure_uploaded_file_exists(file_path) # TODO: Merge this later into the read_file_text
    text_content = read_file_text(file_path)
    if not text_content.strip():
        raise HTTPException(status_code=400, detail="The file contains no readable text.")

    session_dir = settings.SESSION_FOLDER / session_id
    cache_path = session_dir / f"{filename}_top-nouns.json"

    try:
        if cache_path.exists():
            with open(cache_path, "r", encoding="utf-8") as f:
                cached_nouns = json.load(f)

            if isinstance(cached_nouns, dict) and len(cached_nouns) >= top_n:
                sorted_nouns = sorted(
                    cached_nouns.items(),
                    key=lambda item: item[1],
                    reverse=True
                )
                top_nouns = dict(sorted_nouns[:top_n])
                return ApiResponse.success(
                    message="Top nouns loaded from cache.",
                    data={"nouns": top_nouns}
                )

        # ⏳ Compute new nouns
        stop_words = request.app.state.stop_words
        all_nouns = request.app.state.all_nouns

        # This already returns a dict
        new_nouns_dict = await asyncio.to_thread(
            extract_top_n_nouns_with_frequency,
            text_content,
            top_n,
            stop_words,
            all_nouns
        )

        # 💾 Save full dict to cache
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(new_nouns_dict, f, indent=2)

        return ApiResponse.success(
            message="Top nouns computed and cached.",
            data={"nouns": new_nouns_dict}
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process top nouns: {str(e)}")
    