# routes/api_routes.py
from fastapi import APIRouter, Form, Depends, HTTPException, FileResponse, BackgroundTasks
from services.text_processing import get_noun_list, process_sentences
from services.clustering import get_target_matrix, label_sentences_by_cluster, suggest_num_clusters
from services.file_generation import generate_summary_files, generate_detailed_files
from utils.model_config import get_disamb_model
from config.settings import UPLOAD_FOLDER, SUMMARY_FOLDER, DETAILED_FOLDER
from utils.helpers import allowed_file, pdf2text

router = APIRouter()

@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    if not allowed_file(file.filename):
        raise HTTPException(status_code=400, detail="File type not allowed. Please upload a PDF or TXT file.")
    file_path = UPLOAD_FOLDER / file.filename
    with open(file_path, "wb") as f:
        f.write(await file.read())
    text = pdf2text(file_path) if file.filename.lower().endswith(".pdf") else file_path.read_text(encoding="utf-8")
    return {"filename": file.filename, "text_length": len(text), "upload_path": str(file_path)}

@router.post("/list")
async def list_nouns(filename: str = Form(...), top_n: int = Form(50)):
    file_path = UPLOAD_FOLDER / filename
    if not file_path.is_file():
        raise HTTPException(status_code=404, detail="File not found.")
    nouns = get_noun_list(filename, top_n)
    return {"nouns": nouns}

@router.post("/target-matrix")
async def target_matrix(
    filename: str = Form(...),
    target_word: str = Form(...),
    frequency_limit: int = Form(100),
    disamb_model=Depends(get_disamb_model)
):
    file_path = UPLOAD_FOLDER / filename
    if not file_path.is_file():
        raise HTTPException(status_code=404, detail="File not found.")
    text = pdf2text(file_path) if filename.lower().endswith(".pdf") else file_path.read_text(encoding="utf-8")
    sentences = process_sentences(text, target_word, frequency_limit)
    if not sentences:
        raise HTTPException(status_code=404, detail="No matching sentences found.")
    matrix = get_target_matrix(sentences, disamb_model, target_word)
    suggested_k = suggest_num_clusters(matrix)
    return {"matrix_shape": list(matrix.shape), "num_sentences": len(sentences), "target_word": target_word, "suggested_k": suggested_k}