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
    Build a similarity matrix for sentences containing the target word in the given file (within a session),
    and return the optimal number of clusters and elbow plot data.
    """
    # Ensure file exists within the session
    file_path = ensure_uploaded_file_exists(session_id, filename)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"File '{filename}' not found in session '{session_id}'.")

    # Read the content
    text_content = read_file_text(session_id, filename)
    if not text_content:
        raise HTTPException(status_code=400, detail="The file contains no readable text.")

    try:
        matrix, sentences = build_target_word_similarity_matrix(
            text_content=text_content,
            target_word=target_word,
            model=request.app.state.disamb_model,
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
