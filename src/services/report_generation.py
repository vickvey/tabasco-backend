from nltk.tokenize import word_tokenize
from pathlib import Path
from src.models import DisambModel
from typing import List, Tuple
import json
import re

def merge_subwords_with_scores(tokens: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
    """
    Merges WordPiece tokens like 'em', '##bed', '##ding' into 'embedding'
    while averaging their similarity scores.
    """
    merged = []
    temp_token = ""
    temp_scores = []

    for token, score in tokens:
        if token.startswith("##"):
            temp_token += token[2:]
            temp_scores.append(score)
        else:
            if temp_token:
                merged.append((temp_token, sum(temp_scores) / len(temp_scores)))
                temp_token = ""
                temp_scores = []
            temp_token = token
            temp_scores = [score]

    if temp_token:
        merged.append((temp_token, sum(temp_scores) / len(temp_scores)))

    # Remove any special characters accidentally introduced
    clean_merged = [
        (re.sub(r"[^a-zA-Z0-9]", "", word), sim)
        for word, sim in merged if re.sub(r"[^a-zA-Z0-9]", "", word)
    ]
    return clean_merged


def generate_summary_json(
    target_word: str,
    clusters_dict: dict[int, list[str]],
    summary_folder_path: Path,
    disamb_model: DisambModel | None = None
):
    summary_folder_path.mkdir(parents=True, exist_ok=True)

    for cluster_num, sentences in clusters_dict.items():
        if disamb_model:
            all_context_words = []
            for sentence in sentences:
                context_words = disamb_model.get_context_words(sentence, target_word, top_k=10)
                all_context_words.extend(context_words)

            word_sim_dict = {}
            for word, sim in all_context_words:
                word_sim_dict[word] = max(word_sim_dict.get(word, sim), sim)

            top_words = sorted(word_sim_dict.items(), key=lambda x: x[1], reverse=True)[:50]
        else:
            word_freq = {}
            for sentence in sentences:
                tokens = word_tokenize(sentence)
                for word in tokens:
                    if word.lower() != target_word.lower() and word.isalpha():
                        word_freq[word] = word_freq.get(word, 0) + 1
            top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:50]

        summary_data = {
            "cluster_id": int(cluster_num),
            "top_words": top_words,
            "sentences": sentences[:50]
        }

        file_path = summary_folder_path / f"summary_text_{cluster_num}.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=2)


def generate_detailed_json(
    clusters: dict[int, list[str]],
    disamb_model: DisambModel,
    target_word: str,
    detailed_folder: Path,
    threshold: float = 0.5
):
    detailed_folder.mkdir(parents=True, exist_ok=True)
    cache_folder = detailed_folder / "cache"
    cache_folder.mkdir(exist_ok=True)

    detailed_paths = []

    for cluster_num, sentences in clusters.items():
        context_data = []
        for idx, sentence in enumerate(sentences):
            context_words = disamb_model.get_context_words(
                sentence, target_word, top_k=10, threshold=threshold
            )
            context_data.append({
                "cluster_id": int(cluster_num),
                "sentence_id": idx,
                "sentence": sentence,
                "context_words": context_words
            })

        file_path = detailed_folder / f"text_{cluster_num}.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(context_data, f, indent=2)
        detailed_paths.append(str(file_path))

    return detailed_paths
