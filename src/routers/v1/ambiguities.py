import json
from fastapi import APIRouter, HTTPException, Request, Form
from fastapi.responses import JSONResponse
from src.config import settings
from src.services import (
    build_target_word_similarity_matrix,
    suggest_num_clusters_with_data,
)
from src.utils import (
    ensure_uploaded_file_exists,
    read_file_text,
    ApiResponse
)

SESSION_FOLDER = settings.SESSION_FOLDER

router = APIRouter()


@router.post("/target-matrix")
async def get_optimal_clusters_data_from_target_matrix(
    request: Request,
    session_id: str = Form(...),
    filename: str = Form(...),
    target_word: str = Form(...),
    frequency: int = Form(...)
) -> JSONResponse:
    """
    Build or reuse a similarity matrix for sentences containing the target word,
    then return optimal clusters and elbow plot data.
    """
    # Ensure session file exists
    file_path = ensure_uploaded_file_exists(session_id, filename)
    text_content = read_file_text(session_id, filename)
    if not text_content.strip():
        raise HTTPException(status_code=400, detail="The file contains no readable text.")

    session_dir = SESSION_FOLDER / session_id
    cache_filename = f"target-mat_{filename}_{target_word}_{frequency}.json"
    cache_path = session_dir / cache_filename

    # ⚡ Check if cached result exists
    if cache_path.exists():
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cached_data = json.load(f)
            return ApiResponse.success(
                message="Target matrix loaded from cache.",
                data=cached_data
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load cached matrix: {str(e)}")

    # 🧠 Compute matrix & elbow data
    try:
        matrix, sentences = build_target_word_similarity_matrix(
            text_content=text_content,
            target_word=target_word,
            model=request.app.state.disamb_model,
            frequency_limit=frequency
        )

        opt_k, k_range, wcss = suggest_num_clusters_with_data(matrix)

        result_data = {
            "target_word": target_word,
            "sentence_count": len(sentences),
            "matrix_shape": list(matrix.shape),
            "optimal_k": int(opt_k),
            "k_range": [int(k) for k in k_range],
            "wcss": [float(w) for w in wcss],
        }

        # 💾 Save to cache
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(result_data, f, indent=2)

        return ApiResponse.success(
            message="Target matrix computed and cached.",
            data=result_data
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Matrix generation failed: {str(e)}")

