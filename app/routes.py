# app/routes.py
import shutil
from pathlib import Path

from fastapi import APIRouter, Form, HTTPException, Depends, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse

from app.utils.helpers import pdf2text, allowed_file
from app.services.text_processing import get_noun_list, process_sentences
from app.services.clustering import (
    get_target_matrix,
    label_sentences_by_cluster,
    suggest_num_clusters,
)
from app.services.file_generation import (
    generate_summary_files,
    generate_detailed_files,
)
from app.models.disamb_model import DisambModel
from app.utils.model_config import get_disamb_model
from app.settings import DETAILED_FOLDER, SUMMARY_FOLDER, UPLOAD_FOLDER
from app.utils.api_response import ApiResponse

router = APIRouter()


def _ensure_uploaded_file_exists(filename: str) -> Path:
    """
    Given a filename (string), ensure the file exists under UPLOAD_FOLDER.
    Returns the full Path if it exists; otherwise raises HTTPException 404.
    """
    file_path = UPLOAD_FOLDER / filename
    if not file_path.is_file():
        raise HTTPException(status_code=404, detail=f"File '{filename}' not found.")
    return file_path


def _read_file_text(file_path: Path) -> str:
    """
    Read text from a Path. If it’s a PDF, convert via pdf2text; otherwise read as UTF-8 text.
    """
    if file_path.suffix.lower() == ".pdf":
        return pdf2text(file_path)
    return file_path.read_text(encoding="utf-8")


# === File Upload Route ===
@router.post("/files/upload")
async def upload_file(file: UploadFile = File(...)) -> JSONResponse:
    """
    Upload a PDF or TXT. Saves the file in UPLOAD_FOLDER and returns its metadata.
    """
    if not allowed_file(file.filename):
        return ApiResponse.error(
            message="File type not allowed; only PDF or TXT are accepted.",
            status_code=400,
        )

    # Save to disk
    file_location = UPLOAD_FOLDER / file.filename
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Extract text length for feedback
    try:
        text = _read_file_text(file_location)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read file: {e}")

    return ApiResponse.success(
        message="File uploaded successfully",
        data={
            "filename": file.filename,
            "text_length": len(text),
            "upload_path": str(file_location),
        },
    )


# === Utility Endpoints for “Main” ===
@router.post("/main/list")
async def list_nouns(
    filename: str = Form(...),
    top_n: int = Form(50),
) -> JSONResponse:
    """
    Returns the top‐N most frequent nouns in the uploaded file.
    """
    file_path = _ensure_uploaded_file_exists(filename)
    nouns = get_noun_list(filename, top_n)
    return ApiResponse.success(message="Noun list retrieved", data={"nouns": nouns})


@router.post("/main/target-matrix")
async def target_matrix(
    filename: str = Form(...),
    target_word: str = Form(...),
    frequency_limit: int = Form(100),
    disamb_model: DisambModel = Depends(get_disamb_model),
) -> JSONResponse:
    """
    Build a target‐word matrix for clustering:
      1. Read file → extract sentences containing target_word (limited by frequency_limit).
      2. Compute BERT embeddings via disamb_model.
      3. Suggest ideal cluster count.
    Returns matrix shape, sentence count, and suggested k.
    """
    file_path = _ensure_uploaded_file_exists(filename)
    text = _read_file_text(file_path)

    sentences = process_sentences(text, target_word, frequency_limit)
    if not sentences:
        raise HTTPException(
            status_code=404,
            detail=f"No sentences found containing the word '{target_word}'.",
        )

    matrix = get_target_matrix(sentences, disamb_model, target_word)
    suggested_k = suggest_num_clusters(matrix)

    return ApiResponse.success(
        message="Target‐matrix generated",
        data={
            "matrix_shape": list(matrix.shape),
            "num_sentences": len(sentences),
            "target_word": target_word,
            "suggested_k": suggested_k,
        },
    )


# === Ambiguity‐focused Endpoints ===
@router.post("/ambiguities/list")
async def ambiguities_list(
    filename: str = Form(...),
    top_n: int = Form(50),
) -> JSONResponse:
    """
    Extract top N frequent nouns from the uploaded file for ambiguity listing.
    """
    file_path = _ensure_uploaded_file_exists(filename)
    noun_list = get_noun_list(filename, top_n=top_n)
    return ApiResponse.success(message="Ambiguity noun list", data={"nouns": noun_list})


@router.post(
    "/ambiguities/target-matrix",
    dependencies=[Depends(get_disamb_model)],
)
async def ambiguities_target_matrix(
    filename: str = Form(...),
    target_word: str = Form(...),
    frequency_limit: int = Form(100),
    disamb_model: DisambModel = Depends(get_disamb_model),
) -> JSONResponse:
    """
    Same as /main/target-matrix, but under the /ambiguities path.
    """
    file_path = _ensure_uploaded_file_exists(filename)
    text = _read_file_text(file_path)

    sentences = process_sentences(text, target_word, frequency_limit)
    if not sentences:
        raise HTTPException(
            status_code=404,
            detail=f"No sentences found containing the word '{target_word}'.",
        )

    matrix = get_target_matrix(sentences, disamb_model, target_word)
    suggested_k = suggest_num_clusters(matrix)

    return ApiResponse.success(
        message="Ambiguity target‐matrix generated",
        data={
            "matrix_shape": list(matrix.shape),
            "num_sentences": len(sentences),
            "target_word": target_word,
            "suggested_k": suggested_k,
        },
    )


@router.post("/ambiguities/cluster-sentences")
async def cluster_sentences(
    filename: str = Form(...),
    target_word: str = Form(...),
    frequency_limit: int = Form(100),
    num_clusters: int = Form(3),
    disamb_model: DisambModel = Depends(get_disamb_model),
) -> JSONResponse:
    """
    Cluster sentences that contain the target word into `num_clusters`, returning each cluster’s sentences.
    """
    file_path = _ensure_uploaded_file_exists(filename)
    text = _read_file_text(file_path)

    sentences = process_sentences(text, target_word, frequency_limit)
    if not sentences:
        raise HTTPException(
            status_code=404,
            detail=f"No sentences found containing the word '{target_word}'.",
        )

    matrix = get_target_matrix(sentences, disamb_model, target_word)
    clustered = label_sentences_by_cluster(sentences, matrix, num_clusters)

    # Convert to JSON‐friendly format: { "Cluster 1": [...], "Cluster 2": [...] }
    result = {f"Cluster {idx + 1}": bucket for idx, bucket in clustered.items()}

    return ApiResponse.success(message="Sentences clustered", data={"clusters": result})


@router.post("/ambiguities/context-words")
async def ambiguities_context_words(
    filename: str = Form(...),
    target_word: str = Form(...),
    frequency_limit: int = Form(100),
    top_k: int = Form(10),
    disamb_model: DisambModel = Depends(get_disamb_model),
) -> JSONResponse:
    """
    For the first sentence containing `target_word`, return its top‐k most similar context words.
    """
    file_path = _ensure_uploaded_file_exists(filename)
    text = _read_file_text(file_path)

    sentences = process_sentences(text, target_word, frequency_limit)
    if not sentences:
        raise HTTPException(
            status_code=404,
            detail=f"No sentences found containing the word '{target_word}'.",
        )

    first_sentence = sentences[0]
    context_words = disamb_model.get_context_words(first_sentence, target_word, top_k=top_k)

    return ApiResponse.success(
        message="Context words retrieved",
        data={
            "sentence": first_sentence,
            "target_word": target_word,
            "context_words": context_words,
        },
    )


@router.post("/ambiguities/generate-summary")
async def generate_summary(
    filename: str = Form(...),
    target_word: str = Form(...),
    frequency_limit: int = Form(100),
    num_clusters: int = Form(3),
    disamb_model: DisambModel = Depends(get_disamb_model),
) -> JSONResponse:
    """
    1. Cluster sentences by `target_word`,
    2. Generate summary & detailed TXT files for each cluster,
    3. Return the generated filenames.
    """
    file_path = _ensure_uploaded_file_exists(filename)
    text = _read_file_text(file_path)

    sentences = process_sentences(text, target_word, frequency_limit)
    if not sentences:
        raise HTTPException(
            status_code=404,
            detail=f"No sentences found containing the word '{target_word}'.",
        )

    matrix = get_target_matrix(sentences, disamb_model, target_word)
    clusters = label_sentences_by_cluster(sentences, matrix, num_clusters)

    # Generate files on disk
    generate_summary_files(target_word, clusters, SUMMARY_FOLDER, disamb_model)
    generate_detailed_files(clusters, disamb_model, target_word, DETAILED_FOLDER)

    summary_files = [f"summary_text_{i}.txt" for i in range(len(clusters))]
    detailed_files = [f"text_{i}.txt" for i in range(len(clusters))]

    return ApiResponse.success(
        message=f"Generated files for {len(clusters)} clusters.",
        data={"summary_files": summary_files, "detailed_files": detailed_files},
    )


# === Download Endpoints ===
@router.get("/ambiguities/download-summary/{cluster_number}")
async def download_summary(cluster_number: int) -> FileResponse:
    """
    Download the summary text file for a given cluster index.
    """
    file_path = SUMMARY_FOLDER / f"summary_text_{cluster_number}.txt"
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Summary file not found.")
    return FileResponse(
        file_path,
        media_type="text/plain",
        filename=f"summary_cluster_{cluster_number}.txt",
    )


@router.get("/ambiguities/download-detailed/{cluster_number}")
async def download_detailed(cluster_number: int) -> FileResponse:
    """
    Download the detailed text file for a given cluster index.
    """
    file_path = DETAILED_FOLDER / f"text_{cluster_number}.txt"
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Detailed file not found.")
    return FileResponse(
        file_path,
        media_type="text/plain",
        filename=f"detailed_cluster_{cluster_number}.txt",
    )
