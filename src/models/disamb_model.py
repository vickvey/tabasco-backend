import torch
import json
import numpy as np
import torch.nn.functional as F
from torch import device as torch_device
from transformers import BertTokenizer, BertModel
from typing import List, Tuple
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import re

def _standardize_text(text: str) -> str:
    """
    Cleans and standardizes raw text for downstream NLP tasks.

    This function performs the following steps:
    - Returns an empty string if the input is not a string.
    - Removes URLs (e.g., http://..., www...).
    - Removes email mentions (e.g., @username).
    - Removes special characters, symbols, and numeric digits.
    - Converts text to lowercase.
    - Removes isolated single-character tokens (e.g., 'a', 'b').
    - Collapses multiple consecutive whitespace characters into a single space.

    Args:
        text (str): The input string to clean.

    Returns:
        str: The standardized and cleaned version of the input text.
    """
    if not isinstance(text, str):
        return ""

    text = re.sub(r"http\S+|www\S+", " ", text)      # Remove URLs
    text = re.sub(r"@\w+", " ", text)                # Remove mentions
    text = re.sub(r"[^a-zA-Z\s]", " ", text)         # Remove special chars and digits
    text = re.sub(r"\d+", " ", text)                 # Remove numbers
    text = text.lower()                              # Lowercase
    text = re.sub(r"\b[a-zA-Z]\b", " ", text)        # Remove single characters
    text = re.sub(r"\s+", " ", text)                 # Normalize spaces

    return text.strip()

class DisambModel:
    """Wrapper around BERT to extract contextual embeddings for word sense/context disambiguation in a specific domain.

    Provides methods to extract:
    - Target word embeddings
    - Sentence embeddings via mean-pooling
    - Token embeddings for all tokens
    - Context tokens most similar to a target word
    - Visualization of token embeddings
    - Saving embeddings to files
    """

    def __init__(
        self,
        model: BertModel,
        tokenizer: BertTokenizer,
        device: torch_device
    ) -> None:
        """Initialize the DisambModel with a pre-trained BERT model and tokenizer.

        Args:
            model (BertModel): Pre-trained BERT model from HuggingFace Transformers.
            tokenizer (BertTokenizer): Tokenizer associated with the BERT model.
            device (torch.device): Device (CPU or CUDA) for model inference.

        Notes:
            - Configures model to output hidden states.
            - Sets model to evaluation mode.
            - Moves model to the specified device.
        """
        model.config.output_hidden_states = True
        model.eval()
        self.model = model.to(device)  # type: ignore
        self.tokenizer = tokenizer
        self.device = device

    @staticmethod
    def _find_subword_span(tokens: List[str], target_tokens: List[str]) -> Tuple[int, int]:
        """Find the start and end indices of target subword tokens in a token list.

        Args:
            tokens (List[str]): List of tokenized subwords from a sentence.
            target_tokens (List[str]): Tokenized subwords of the target word/phrase.

        Returns:
            Tuple[int, int]: Start and end indices of the target tokens.

        Raises:
            ValueError: If target_tokens are not found in the token list.

        Example:
            >>> tokens = ["[CLS]", "I", "love", "em", "##bed", "##ding", "[SEP]"]
            >>> target_tokens = ["em", "##bed", "##ding"]
            >>> DisambModel._find_subword_span(tokens, target_tokens)
            (3, 5)
        """
        span_len = len(target_tokens)
        for idx in range(len(tokens) - span_len + 1):
            if tokens[idx : idx + span_len] == target_tokens:
                return idx, idx + span_len - 1
        raise ValueError(f"Target tokens {target_tokens} not found in token list.")

    def forward(self, input_sentence: str, target_word: str) -> torch.Tensor:
        """Extract contextual embedding for a target word in a sentence.

        Averages the embeddings of the target word's subword tokens from the last four BERT layers.

        Args:
            input_sentence (str): Sentence containing the target word.
            target_word (str): Word/phrase to extract embedding for.

        Returns:
            torch.Tensor: Averaged embedding vector for the target word (shape: [768]).

        Raises:
            ValueError: If the target word is not found in the sentence.
        """
        marked_text = f"{self.tokenizer.cls_token} {input_sentence} {self.tokenizer.sep_token}"
        tokens = self.tokenizer.tokenize(marked_text)
        target_tokens = self.tokenizer.tokenize(target_word.lower())

        try:
            tgt_start, tgt_end = self._find_subword_span(tokens, target_tokens)
        except ValueError:
            raise ValueError(f"Target word '{target_word}' not found in sentence.")

        encodings = self.tokenizer.encode_plus(
            marked_text,
            add_special_tokens=False,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        input_ids = encodings["input_ids"].to(self.device)  # type: ignore
        token_type_ids = torch.ones_like(input_ids).to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids, token_type_ids=token_type_ids)
            hidden_states = outputs.hidden_states

        subword_embs = []
        for idx in range(tgt_start, tgt_end + 1):
            last_four = [hidden_states[-i][0][idx] for i in range(1, 5)]
            subword_embs.append(torch.stack(last_four).sum(dim=0))

        target_emb = torch.stack(subword_embs).mean(dim=0)
        return target_emb

    def get_context_words(
        self,
        sentence: str,
        target_word: str,
        top_k: int = 10,
        threshold: float = 0.5
    ) -> List[Tuple[str, float]]:
        cleaned_sentence = _standardize_text(sentence)
        marked_text = f"{self.tokenizer.cls_token} {cleaned_sentence} {self.tokenizer.sep_token}"

        tokens = self.tokenizer.tokenize(marked_text)
        target_tokens = self.tokenizer.tokenize(target_word.lower())

        try:
            first_start, first_end = self._find_subword_span(tokens, target_tokens)
        except ValueError:
            return []

        encodings = self.tokenizer.encode_plus(
            marked_text,
            add_special_tokens=False,
            return_tensors="pt"
        )

        input_ids = encodings["input_ids"].to(self.device)
        token_type_ids = torch.ones_like(input_ids).to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids, token_type_ids=token_type_ids)
            hidden_states = outputs.hidden_states

        token_embeddings = []
        for idx in range(input_ids.size(1)):
            layers = [hidden_states[-i][0][idx] for i in range(1, 5)]
            token_embeddings.append(torch.stack(layers).sum(dim=0))

        target_vector = token_embeddings[first_start]
        similarities = []

        merged_tokens = []
        current_token = ""
        current_score = 0.0

        for idx, token_str in enumerate(tokens):
            if idx in range(first_start, first_end + 1):
                continue
            if token_str in {str(self.tokenizer.cls_token), str(self.tokenizer.sep_token)}:
                continue

            score = F.cosine_similarity(
                target_vector.unsqueeze(0),
                token_embeddings[idx].unsqueeze(0),
                dim=1
            ).item()

            if score >= threshold:
                if token_str.startswith("##"):
                    current_token += token_str[2:]
                    current_score = max(current_score, score)
                else:
                    if current_token:
                        merged_tokens.append((current_token, current_score))
                    current_token = token_str
                    current_score = score

        if current_token:
            merged_tokens.append((current_token, current_score))

        filtered = [(word, sim) for word, sim in merged_tokens if word.isalpha()]
        filtered.sort(key=lambda x: x[1], reverse=True)

        return filtered[:top_k]

    def get_sentence_embedding(self, sentence: str) -> torch.Tensor:
        """Compute sentence embedding by mean-pooling token embeddings.

        Uses the last four BERT layers for each token.

        Args:
            sentence (str): Input sentence.

        Returns:
            torch.Tensor: Sentence embedding vector (shape: [768]).
        """
        marked_text = f"{self.tokenizer.cls_token} {sentence} {self.tokenizer.sep_token}"
        encodings = self.tokenizer.encode_plus(
            marked_text,
            add_special_tokens=False,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        input_ids = encodings["input_ids"].to(self.device)  # type: ignore
        token_type_ids = torch.ones_like(input_ids).to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids, token_type_ids=token_type_ids)
            hidden_states = outputs.hidden_states

        seq_len = input_ids.size(1)
        token_embs = []
        for idx in range(seq_len):
            layers = [hidden_states[-i][0][idx] for i in range(1, 5)]
            token_embs.append(torch.stack(layers).sum(dim=0))

        sentence_emb = torch.stack(token_embs).mean(dim=0)
        return sentence_emb

    def get_all_token_vectors(self, sentence: str) -> List[Tuple[str, torch.Tensor]]:
        """Retrieve embeddings for all tokens in a sentence.

        Args:
            sentence (str): Input sentence.

        Returns:
            List[Tuple[str, torch.Tensor]]: List of (token, embedding) pairs, including special tokens.
        """
        marked_text = f"{self.tokenizer.cls_token} {sentence} {self.tokenizer.sep_token}"
        tokens = self.tokenizer.tokenize(marked_text)

        encodings = self.tokenizer.encode_plus(
            marked_text,
            add_special_tokens=False,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        input_ids = encodings["input_ids"].to(self.device)  # type: ignore
        token_type_ids = torch.ones_like(input_ids).to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids, token_type_ids=token_type_ids)
            hidden_states = outputs.hidden_states

        token_vectors = []
        for idx, token in enumerate(tokens):
            layers = [hidden_states[-i][0][idx] for i in range(1, 5)]
            vector = torch.stack(layers).sum(dim=0)
            token_vectors.append((token, vector))

        return token_vectors

    def visualize(self, sentence: str, method: str = "pca") -> None:
        """Visualize token embeddings in 2D using PCA or t-SNE.

        Args:
            sentence (str): Input sentence.
            method (str, optional): Dimensionality reduction method ("pca" or "tsne"). Defaults to "pca".

        Raises:
            ValueError: If method is not "pca" or "tsne".

        Notes:
            - Displays a scatter plot with token labels.
            - Requires a graphical environment for matplotlib.
        """
        token_vecs = self.get_all_token_vectors(sentence)
        tokens = [t for t, _ in token_vecs]
        vectors = torch.stack([v for _, v in token_vecs]).cpu().numpy()

        if method == "pca":
            reducer = PCA(n_components=2)
        elif method == "tsne":
            reducer = TSNE(n_components=2, random_state=42)
        else:
            raise ValueError("method must be 'pca' or 'tsne'")

        reduced = reducer.fit_transform(vectors)

        plt.figure(figsize=(10, 7))
        for i, token in enumerate(tokens):
            plt.scatter(reduced[i, 0], reduced[i, 1])
            plt.text(reduced[i, 0] + 0.01, reduced[i, 1] + 0.01, token, fontsize=9)
        plt.title(f"{method.upper()} of Token Embeddings")
        plt.grid(True)
        plt.show()

    def save_to_file(self, sentence: str, path: str, format: str = "pt") -> None:
        """Save token embeddings to a file.

        Args:
            sentence (str): Input sentence.
            path (str): File path to save embeddings.
            format (str, optional): File format ("pt", "npy", or "json"). Defaults to "pt".

        Raises:
            ValueError: If format is not "pt", "npy", or "json".
        """
        token_vecs = self.get_all_token_vectors(sentence)
        if format == "pt":
            torch.save({t: v for t, v in token_vecs}, path)
        elif format == "npy":
            np.save(path, {t: v.cpu().numpy() for t, v in token_vecs})  # type: ignore
        elif format == "json":
            data = {t: v.cpu().tolist() for t, v in token_vecs}
            with open(path, "w") as f:
                json.dump(data, f)
        else:
            raise ValueError("Format must be one of: pt, npy, json")
        
