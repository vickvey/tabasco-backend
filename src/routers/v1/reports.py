# from fastapi import APIRouter, Form, HTTPException
# from fastapi.responses import JSONResponse
# from src.config import settings
# from src.utils import (
#     ApiResponse,
# )

# # Mount path configs from settings
# PROJECT_ROOT = settings.PROJECT_ROOT
# UPLOAD_FOLDER = settings.UPLOAD_FOLDER
# SUMMARY_FOLDER = settings.SUMMARY_FOLDER
# DETAILED_FOLDER = settings.DETAILED_FOLDER
# # LOG_DIR = settings.LOG_DIR # TODO: Complete this

# router = APIRouter()


# @router.post("/generate-summary")
# async def generate_summary_report_for_all_clusters(
#     filename: str = Form(...),
#     target_word: str = Form(...),
#     frequency_limit: int = Form(100),
#     num_clusters: int = Form(3),
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

#     matrix = get_target_matrix(sentences, disamb_model, target_word)
#     clusters = label_sentences_by_cluster(sentences, matrix, num_clusters)

#     generate_summary_files(target_word, clusters, SUMMARY_FOLDER, disamb_model)
#     generate_detailed_files(clusters, disamb_model, target_word, DETAILED_FOLDER)

#     summary_files = [f"summary_text_{i}.txt" for i in range(len(clusters))]
#     detailed_files = [f"text_{i}.txt" for i in range(len(clusters))]

#     return ApiResponse.success(
#         message=f"Generated files for {len(clusters)} clusters.",
#         data={"summary_files": summary_files, "detailed_files": detailed_files},
#     )

# @router.post("/generate-detailed")
# async def get_detailed_report_for_all_clusters() -> JSONResponse:
#     return JSONResponse({"message": "here is the detailed report"})
