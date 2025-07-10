import pandas as pd
import json
from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import JSONResponse
from src.config import settings
from src.utils import ApiResponse, ensure_uploaded_file_exists, read_file_text
from src.services import (
    build_target_word_similarity_matrix,
    label_sentences_by_cluster,
)
from src.services.report_generation import (
    generate_summary_json,
    generate_detailed_json,
)

router = APIRouter()


@router.post("/generate-summary")
async def generate_summary_report_for_all_clusters(
    request: Request,
    session_id: str = Form(...),
    filename: str = Form(...),
    target_word: str = Form(...),
    frequency_limit: int = Form(100),
    num_clusters: int = Form(3),
    threshold: float = Form(0.5)
) -> JSONResponse:
    if num_clusters <= 0:
        raise HTTPException(status_code=400, detail="num_clusters must be positive")
    if not target_word.strip():
        raise HTTPException(status_code=400, detail="target_word cannot be empty")

    file_path = ensure_uploaded_file_exists(session_id, filename)
    text_content = read_file_text(session_id, filename)
    if not text_content.strip():
        raise HTTPException(status_code=400, detail="The file contains no readable text.")

    matrix, sentences = build_target_word_similarity_matrix(
        text_content=text_content,
        target_word=target_word,
        model=request.app.state.disamb_model,
        frequency_limit=frequency_limit
    )
    if not sentences:
        raise HTTPException(status_code=404, detail=f"No sentences found containing the word '{target_word}'.")

    clusters = label_sentences_by_cluster(sentences, matrix, num_clusters)

    summary_folder = settings.SESSION_FOLDER / session_id / "summaries"
    generate_summary_json(target_word, clusters, summary_folder, request.app.state.disamb_model)

    summary_data = []
    for path in sorted(summary_folder.glob("summary_text_*.json")):
        summary_data.append(json.loads(path.read_text(encoding="utf-8")))

    return ApiResponse.success(
        message=f"Generated summary reports for {len(clusters)} clusters.",
        data={"summaries": summary_data}
    )


@router.post("/generate-detailed")
async def get_detailed_report_for_all_clusters(
    request: Request,
    session_id: str = Form(...),
    filename: str = Form(...),
    target_word: str = Form(...),
    frequency_limit: int = Form(100),
    num_clusters: int = Form(3),
    threshold: float = Form(0.5)
) -> JSONResponse:
    if num_clusters <= 0:
        raise HTTPException(status_code=400, detail="num_clusters must be positive")
    if not target_word.strip():
        raise HTTPException(status_code=400, detail="target_word cannot be empty")

    file_path = ensure_uploaded_file_exists(session_id, filename)
    text_content = read_file_text(session_id, filename)
    if not text_content.strip():
        raise HTTPException(status_code=400, detail="The file contains no readable text.")

    matrix, sentences = build_target_word_similarity_matrix(
        text_content=text_content,
        target_word=target_word,
        model=request.app.state.disamb_model,
        frequency_limit=frequency_limit
    )
    if not sentences:
        raise HTTPException(status_code=404, detail=f"No sentences found containing the word '{target_word}'.")

    clusters = label_sentences_by_cluster(sentences, matrix, num_clusters)

    detailed_folder = settings.SESSION_FOLDER / session_id / "detailed"
    generate_detailed_json(clusters, request.app.state.disamb_model, target_word, detailed_folder, threshold=threshold)

    detailed_data = []
    for path in sorted(detailed_folder.glob("text_*.json")):
        detailed_data.append(json.loads(path.read_text(encoding="utf-8")))

    return ApiResponse.success(
        message=f"Generated detailed reports for {len(clusters)} clusters.",
        data={"detailed": detailed_data}
    )


@router.post("/generate-threshold-plots")
async def generate_threshold_plots(
    request: Request,
    session_id: str = Form(...),
    filename: str = Form(...),
    target_word: str = Form(...),
    frequency_limit: int = Form(100),
    num_clusters: int = Form(3)
) -> JSONResponse:
    file_path = ensure_uploaded_file_exists(session_id, filename)
    text_content = read_file_text(session_id, filename)
    if not text_content.strip():
        raise HTTPException(status_code=400, detail="The file contains no readable text.")
    
    matrix, sentences = build_target_word_similarity_matrix(
        text_content=text_content,
        target_word=target_word,
        model=request.app.state.disamb_model,
        frequency_limit=frequency_limit
    )
    if not sentences:
        raise HTTPException(status_code=404, detail=f"No sentences found containing the word '{target_word}'.")

    clusters = label_sentences_by_cluster(sentences, matrix, num_clusters)
    disamb_model = request.app.state.disamb_model
    plot_data = []

    for cluster_num, sentences in clusters.items():
        all_context_words = []
        for sentence in sentences:
            context_words = disamb_model.get_context_words(sentence, target_word, top_k=10)
            all_context_words.extend(context_words)

        words, similarities = zip(*all_context_words) if all_context_words else ([], [])
        df = pd.DataFrame({"word": words, "distance": similarities})
        df = df.sort_values(by="distance", ascending=False)

        cluster_json = {
            "cluster": cluster_num,
            "words": df["word"].tolist(),
            "distances": df["distance"].tolist(),
            "stats": {
                "mean": float(df["distance"].mean()) if not df.empty else 0.0,
                "median": float(df["distance"].median()) if not df.empty else 0.0,
                "max": float(df["distance"].max()) if not df.empty else 0.0,
            },
        }
        plot_data.append(cluster_json)

    return ApiResponse.success(
        message=f"Generated threshold data for {len(clusters)} clusters.",
        data={"plot_data": plot_data}
    )


# @router.post("/visualize-embeddings")
# async def visualize_embeddings(
#     request: Request,
#     session_id: str = Form(...),
#     filename: str = Form(...),
#     target_word: str = Form(...),
#     frequency_limit: int = Form(100),
#     method: str = Form("pca")
# ) -> JSONResponse:
#     file_path = ensure_uploaded_file_exists(session_id, filename)
#     text_content = read_file_text(session_id, filename)
#     if not text_content.strip():
#         raise HTTPException(status_code=400, detail="The file contains no readable text.")
    
#     sentence_pairs = get_sentences_with_target_word(text_content, target_word, frequency_limit=frequency_limit)
#     sentences = [pair[0] for pair in sentence_pairs]
#     if not sentences:
#         raise HTTPException(status_code=404, detail=f"No sentences found containing the word '{target_word}'.")
    
#     disamb_model = request.app.state.disamb_model
#     plot_folder = settings.SESSION_FOLDER / session_id / "plots"
#     plot_folder.mkdir(parents=True, exist_ok=True)
#     plot_paths = []
    
#     for idx, sentence in enumerate(sentences[:5]):  # Limit to 5 sentences for visualization
#         token_vecs = disamb_model.get_all_token_vectors(sentence)
#         tokens = [t for t, _ in token_vecs]
#         vectors = torch.stack([v for _, v in token_vecs]).cpu().numpy()
        
#         if method == "pca":
#             reducer = PCA(n_components=2)
#         elif method == "tsne":
#             reducer = TSNE(n_components=2, random_state=42)
#         else:
#             raise HTTPException(status_code=400, detail="method must be 'pca' or 'tsne'")
        
#         reduced = reducer.fit_transform(vectors)
        
#         plt.figure(figsize=(10, 7))
#         for i, token in enumerate(tokens):
#             plt.scatter(reduced[i, 0], reduced[i, 1])
#             plt.text(reduced[i, 0] + 0.01, reduced[i, 1] + 0.01, token, fontsize=9)
#         plt.title(f"{method.upper()} of Token Embeddings for Sentence {idx + 1}")
#         plt.grid(True)
#         plot_path = plot_folder / f"embedding_plot_{idx}_{method}.png"
#         plt.savefig(plot_path, dpi=300)
#         plt.close()
#         plot_paths.append(f"plots/embedding_plot_{idx}_{method}.png")
    
#     return ApiResponse.success(
#         message="Generated embedding visualizations.",
#         data={"plot_paths": plot_paths}
#     )
