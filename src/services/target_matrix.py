# src/services/generate_similarity_matrix.py

import torch
import os
import pickle
from typing import List, Tuple
from src.models.disamb_model import DisambModel
from src.services.sentence_processing import get_sentences_with_target_word


def compute_cosine_similarity_matrix(
    sentences: List[str],
    target_word: str,
    model: DisambModel
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """
    Given a list of sentences, compute target word embeddings and return cosine similarity matrix.
    """
    cosine_sim = torch.nn.CosineSimilarity(dim=0)
    embedding_list = []

    for sent in sentences:
        try:
            emb = model.forward(sent, target_word)
            embedding_list.append(emb)
        except ValueError:
            continue

    n = len(embedding_list)
    similarity_matrix = torch.zeros((n, n))
    for i in range(n):
        for j in range(n):
            similarity_matrix[i][j] = cosine_sim(embedding_list[i], embedding_list[j])

    return similarity_matrix, embedding_list


def build_target_word_similarity_matrix(
    text_content: str,
    target_word: str,
    model: DisambModel,
    frequency_limit: int = 100,
    save_path: str | None = None
) -> Tuple[torch.Tensor, List[str]]:
    """
    Full pipeline: extract sentences, compute similarity matrix, optionally save.
    """
    sentences = get_sentences_with_target_word(text_content, target_word, frequency_limit=frequency_limit)
    similarity_matrix, _ = compute_cosine_similarity_matrix(sentences, target_word, model)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(similarity_matrix, save_path)
        print(f"[✓] Saved similarity matrix of shape {similarity_matrix.shape} to {save_path}")
    return similarity_matrix, sentences
