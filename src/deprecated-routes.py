# app/deprecated-l.py
from pathlib import Path
from fastapi import APIRouter, Form, HTTPException
from fastapi.responses import JSONResponse
from src.utils import pdf2text
from src.services.sentence_processing import TextProcessingService
from src.config import UPLOAD_FOLDER
from src.utils.api_response import ApiResponse

router = APIRouter()

def _ensure_uploaded_file_exists(filename: str) -> Path:
    """
    Ensure the file exists in UPLOAD_FOLDER. Returns the full Path or raises HTTPException 404.
    """
    file_path = UPLOAD_FOLDER / filename
    if not file_path.is_file():
        raise HTTPException(status_code=404, detail=f"File '{filename}' not found.")
    return file_path

def _read_file_text(file_path: Path) -> str:
    """
    Read text from a Path. Convert PDF via pdf2text or read as UTF-8 text.
    """
    try:
        return pdf2text(file_path) if file_path.suffix.lower() == ".pdf" else file_path.read_text(encoding="utf-8")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read file: {str(e)}")

async def _process_file_and_sentences(filename: str, target_word: str = None, frequency_limit: int = 100) -> tuple[Path, str, list[str]]:
    """
    Common logic to validate file and process sentences.
    Returns (file_path, text, sentences).
    """
    if frequency_limit <= 0:
        raise HTTPException(status_code=400, detail="frequency_limit must be positive")
    file_path = _ensure_uploaded_file_exists(filename)
    text = _read_file_text(file_path)
    sentences = TextProcessingService.process_sentences(text, target_word, frequency_limit) if target_word else []
    return file_path, text, sentences

# === File Upload Route ===
# @router.post("/files/upload")
# async def upload_file(file: UploadFile = File(...)) -> JSONResponse:
#     """
#     Upload a PDF or TXT file to UPLOAD_FOLDER and return metadata.
#     """
#     if not file.filename or not allowed_file(file.filename):
#         raise HTTPException(status_code=400, detail="File type not allowed; only PDF or TXT are accepted.")
#
#     file_location = UPLOAD_FOLDER / file.filename
#     with open(file_location, "wb") as buffer:
#         shutil.copyfileobj(file.file, buffer)
#
#     text = _read_file_text(file_location)
#     return ApiResponse.success(
#         message="File uploaded successfully",
#         data={
#             "filename": file.filename,
#             "text_length": len(text),
#             "upload_path": str(file_location),
#         },
#     )


# === Utility Endpoints ===
# @router.post("/top-n-nouns-with-frequency")
# async def get_n_top_nouns_freq(filename: str = Form(...), top_n: int = Form(50)) -> JSONResponse:
#     """
#     Return the top-N most frequent nouns in the uploaded file.
#     """
#     if 0 >= top_n > 200:
#         raise HTTPException(status_code=400, detail="top_n must be in range [0, 200]")
#
#     file_path = _ensure_uploaded_file_exists(filename)
#     if not file_path.is_file():
#         raise HTTPException(status_code=404, detail=f"File '{filename}' not found.")
#
#     nouns = TextProcessingService.extract_top_n_nouns_with_frequency(filename, top_n)
#     return ApiResponse.success(message="Noun list retrieved", data={"nouns": nouns})

# @router.post("/{prefix}/target-matrix")
# async def target_matrix(
#     prefix: str,
#     filename: str = Form(...),
#     target_word: str = Form(...),
#     frequency_limit: int = Form(100),
#     disamb_model: DisambModel = Depends(get_disamb_model),
# ) -> JSONResponse:
#     """
#     Generate a target-word matrix for clustering and suggest the number of clusters.
#     """
#     if prefix not in ["main", "ambiguities"]:
#         raise HTTPException(status_code=400, detail="Invalid prefix; must be 'main' or 'ambiguities'")
#     if not target_word.strip():
#         raise HTTPException(status_code=400, detail="target_word cannot be empty")
#     _, text, sentences = await _process_file_and_sentences(filename, target_word, frequency_limit)
#     if not sentences:
#         raise HTTPException(status_code=404, detail=f"No sentences found containing the word '{target_word}'.")
#
#     matrix = get_target_matrix(sentences, disamb_model, target_word)
#     suggested_k = int(suggest_num_clusters(matrix))  # Convert to Python int
#     message = "Target-matrix generated" if prefix == "main" else "Ambiguity target-matrix generated"
#
#     return ApiResponse.success(
#         message=message,
#         data={
#             "matrix_shape": [int(dim) for dim in matrix.shape],  # Ensure Python int
#             "num_sentences": len(sentences),
#             "target_word": target_word,
#             "suggested_k": suggested_k,
#         },
#     )

# # === Ambiguity-focused Endpoints ===
# @router.post("/ambiguities/cluster-sentences")
# async def cluster_sentences(
#     filename: str = Form(...),
#     target_word: str = Form(...),
#     frequency_limit: int = Form(100),
#     num_clusters: int = Form(3),
#     disamb_model: DisambModel = Depends(get_disamb_model),
# ) -> JSONResponse:
#     """
#     Cluster sentences containing the target word into num_clusters.
#     """
#     if num_clusters <= 0:
#         raise HTTPException(status_code=400, detail="num_clusters must be positive")
#     if not target_word.strip():
#         raise HTTPException(status_code=400, detail="target_word cannot be empty")
#     _, text, sentences = await _process_file_and_sentences(filename, target_word, frequency_limit)
#     if not sentences:
#         raise HTTPException(status_code=404, detail=f"No sentences found containing the word '{target_word}'.")
#
#     matrix = get_target_matrix(sentences, disamb_model, target_word)
#     clustered = label_sentences_by_cluster(sentences, matrix, num_clusters)
#     result = {f"Cluster {idx + 1}": bucket for idx, bucket in clustered.items()}
#
#     return ApiResponse.success(message="Sentences clustered", data={"clusters": result})

# @router.post("/ambiguities/context-words")
# async def ambiguities_context_words(
#     filename: str = Form(...),
#     target_word: str = Form(...),
#     frequency_limit: int = Form(100),
#     top_k: int = Form(10),
#     disamb_model: DisambModel = Depends(get_disamb_model),
# ) -> JSONResponse:
#     """
#     Return top-k most similar context words for the first sentence containing target_word.
#     """
#     if top_k <= 0:
#         raise HTTPException(status_code=400, detail="top_k must be positive")
#     if not target_word.strip():
#         raise HTTPException(status_code=400, detail="target_word cannot be empty")
#     _, text, sentences = await _process_file_and_sentences(filename, target_word, frequency_limit)
#     if not sentences:
#         raise HTTPException(status_code=404, detail=f"No sentences found containing the word '{target_word}'.")
#
#     first_sentence = sentences[0]
#     context_words = disamb_model.get_context_words(first_sentence, target_word, top_k=top_k)
#
#     return ApiResponse.success(
#         message="Context words retrieved",
#         data={
#             "sentence": first_sentence,
#             "target_word": target_word,
#             "context_words": context_words,
#         },
#     )

# @router.post("/ambiguities/generate-summary")
# async def generate_summary(
#     filename: str = Form(...),
#     target_word: str = Form(...),
#     frequency_limit: int = Form(100),
#     num_clusters: int = Form(3),
#     disamb_model: DisambModel = Depends(get_disamb_model),
# ) -> JSONResponse:
#     """
#     Cluster sentences by target_word and generate summary/detailed TXT files for each cluster.
#     """
#     if num_clusters <= 0:
#         raise HTTPException(status_code=400, detail="num_clusters must be positive")
#     if not target_word.strip():
#         raise HTTPException(status_code=400, detail="target_word cannot be empty")
#     _, text, sentences = await _process_file_and_sentences(filename, target_word, frequency_limit)
#     if not sentences:
#         raise HTTPException(status_code=404, detail=f"No sentences found containing the word '{target_word}'.")
#
#     matrix = get_target_matrix(sentences, disamb_model, target_word)
#     clusters = label_sentences_by_cluster(sentences, matrix, num_clusters)
#
#     generate_summary_files(target_word, clusters, SUMMARY_FOLDER, disamb_model)
#     generate_detailed_files(clusters, disamb_model, target_word, DETAILED_FOLDER)
#
#     summary_files = [f"summary_text_{i}.txt" for i in range(len(clusters))]
#     detailed_files = [f"text_{i}.txt" for i in range(len(clusters))]
#
#     return ApiResponse.success(
#         message=f"Generated files for {len(clusters)} clusters.",
#         data={"summary_files": summary_files, "detailed_files": detailed_files},
#     )

# === Download Endpoints ===
# @router.get("/ambiguities/download-summary/{cluster_number}")
# async def download_summary(cluster_number: int) -> FileResponse:
#     """
#     Download the summary text file for a given cluster index.
#     """
#     if cluster_number < 0:
#         raise HTTPException(status_code=400, detail="cluster_number must be non-negative")
#     file_path = SUMMARY_FOLDER / f"summary_text_{cluster_number}.txt"
#     if not file_path.exists():
#         raise HTTPException(status_code=404, detail="Summary file not found.")
#     return FileResponse(
#         file_path,
#         media_type="text/plain",
#         filename=f"summary_cluster_{cluster_number}.txt",
#     )

# @router.get("/ambiguities/download-detailed/{cluster_number}")
# async def download_detailed(cluster_number: int) -> FileResponse:
#     """
#     Download the detailed text file for a given cluster index.
#     """
#     if cluster_number < 0:
#         raise HTTPException(status_code=400, detail="cluster_number must be non-negative")
#     file_path = DETAILED_FOLDER / f"text_{cluster_number}.txt"
#     if not file_path.exists():
#         raise HTTPException(status_code=404, detail="Detailed file not found.")
#     return FileResponse(
#         file_path,
#         media_type="text/plain",
#         filename=f"detailed_cluster_{cluster_number}.txt",
#     )