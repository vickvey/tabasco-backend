import shutil
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from utils.helpers import allowed_file, pdf2text, standardize_text
from core.business_logic import get_noun_list, process_sentences, get_target_matrix, DisambModel, label_sentences_by_cluster, generate_summary_files
from transformers import BertTokenizer, BertModel
import torch
from pathlib import Path

router = APIRouter()

# === File System Setup ===
PROJECT_ROOT = Path(__file__).resolve().parent.parent
UPLOAD_FOLDER = PROJECT_ROOT / "uploads"
SUMMARY_FOLDER = PROJECT_ROOT / "static" / "summary"

# Create directories if they don't exist
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
SUMMARY_FOLDER.mkdir(parents=True, exist_ok=True)

# Debug prints for directory confirmation
print("📁 PROJECT_ROOT:", str(PROJECT_ROOT))
print("📁 UPLOAD_FOLDER:", str(UPLOAD_FOLDER))
print("📁 SUMMARY_FOLDER:", str(SUMMARY_FOLDER))

# === Model Initialization ===
bert_model = BertModel.from_pretrained("bert-base-uncased", output_hidden_states=True)
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
device = torch.device("cpu")
bert_model.to(device)
disamb_model = DisambModel(bert_model, bert_tokenizer, device)

# === API Endpoints ===

@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Endpoint to upload a file (PDF or TXT). Saves the file to disk and returns metadata.
    """
    if not allowed_file(file.filename):
        raise HTTPException(status_code=400, detail="File type not allowed; only PDF or TXT are accepted.")
    
    file_location = UPLOAD_FOLDER / file.filename
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    if file.filename.lower().endswith(".pdf"):
        text = pdf2text(file_location)
    else:
        with open(file_location, "r", encoding="utf-8") as f:
            text = f.read()

    return JSONResponse(content={
        "filename": file.filename,
        "text_length": len(text),
        "upload_path": str(file_location)
    })

@router.post("/list")
async def get_list(filename: str = Form(...), top_n: int = Form(50)):
    """
    Extract top N frequent nouns from the uploaded file.
    """
    file_path = UPLOAD_FOLDER / filename
    if not file_path.is_file():
        raise HTTPException(status_code=404, detail=f"File '{filename}' not found at {file_path}")
    
    if filename.lower().endswith(".pdf"):
        text = pdf2text(file_path)
    else:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
    
    noun_list = get_noun_list(text, top_n=top_n)
    return JSONResponse(content={"nouns": noun_list})

@router.post("/target-matrix")
async def target_matrix(
    filename: str = Form(...),
    target_word: str = Form(...),
    frequency_limit: int = Form(100)
):
    """
    Compute similarity matrix for all sentences containing a target word in the uploaded file.
    """
    file_path = UPLOAD_FOLDER / filename
    if not file_path.is_file():
        raise HTTPException(status_code=404, detail=f"File '{filename}' not found at {file_path}")
    
    if filename.lower().endswith(".pdf"):
        text = pdf2text(file_path)
    else:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
    
    sentences = process_sentences(text, target_word, frequency_limit)
    if not sentences:
        raise HTTPException(status_code=404, detail=f"No sentences found containing the word '{target_word}'.")

    matrix = get_target_matrix(sentences, disamb_model)
    
    return JSONResponse(content={
        "matrix_shape": list(matrix.shape),
        "num_sentences": len(sentences),
        "target_word": target_word
    })

@router.post("/cluster-sentences")
async def cluster_sentences(
    filename: str = Form(...),
    target_word: str = Form(...),
    frequency_limit: int = Form(100),
    num_clusters: int = Form(3)
):
    """
    Cluster sentences containing the target word using cosine similarity.
    """
    file_path = UPLOAD_FOLDER / filename
    if not file_path.is_file():
        raise HTTPException(status_code=404, detail=f"File '{filename}' not found at {file_path}")
    
    if filename.lower().endswith(".pdf"):
        text = pdf2text(file_path)
    else:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
    
    sentences = process_sentences(text, target_word, frequency_limit)
    if not sentences:
        raise HTTPException(status_code=404, detail=f"No sentences found containing the word '{target_word}'.")

    matrix = get_target_matrix(sentences, disamb_model)
    clustered = label_sentences_by_cluster(sentences, matrix, num_clusters)

    result = {f"Cluster {k + 1}": v for k, v in clustered.items()}
    return JSONResponse(content={"clusters": result})

@router.post("/context-words")
async def get_context_words(
    filename: str = Form(...),
    target_word: str = Form(...),
    frequency_limit: int = Form(100),
    top_k: int = Form(10)
):
    """
    Return top-k most similar context words to the target word in selected sentences.
    """
    file_path = UPLOAD_FOLDER / filename
    if not file_path.is_file():
        raise HTTPException(status_code=404, detail=f"File '{filename}' not found.")

    if filename.lower().endswith(".pdf"):
        text = pdf2text(file_path)
    else:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

    sentences = process_sentences(text, target_word, frequency_limit)
    if not sentences:
        raise HTTPException(status_code=404, detail="No matching sentences found.")

    first_sentence = sentences[0]
    context_words = disamb_model.get_context_words(first_sentence, target_word, top_k=top_k)

    return JSONResponse(content={
        "sentence": first_sentence,
        "target_word": target_word,
        "context_words": context_words
    })

@router.post("/generate-summary")
async def generate_summary(
    filename: str = Form(...),
    target_word: str = Form(...),
    frequency_limit: int = Form(100),
    num_clusters: int = Form(3)
):
    """
    Generate summary .txt files for each cluster of sentences containing the target word.
    """
    file_path = UPLOAD_FOLDER / filename
    if not file_path.is_file():
        raise HTTPException(status_code=404, detail="File not found.")

    if filename.lower().endswith(".pdf"):
        text = pdf2text(file_path)
    else:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

    sentences = process_sentences(text, target_word, frequency_limit)
    if not sentences:
        raise HTTPException(status_code=404, detail="No matching sentences found.")

    matrix = get_target_matrix(sentences, disamb_model)
    clusters = label_sentences_by_cluster(sentences, matrix, num_clusters)

    generate_summary_files(target_word, clusters, SUMMARY_FOLDER)

    return {"message": f"Summary files created for {len(clusters)} clusters.", "clusters": list(clusters.keys())}

@router.get("/download-summary/{cluster_number}")
async def download_summary(cluster_number: int):
    """
    Download the summary file for a given cluster.
    """
    file_path = SUMMARY_FOLDER / f"summary_text_{cluster_number}.txt"
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Summary file not found.")
    return FileResponse(file_path, media_type="text/plain", filename=f"summary_cluster_{cluster_number}.txt")