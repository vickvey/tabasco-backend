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


class DisambModel:
    """
    Wrapper around BERT to extract contextual embeddings:
    - Target word embedding
    - Sentence embedding
    - Token embeddings for all tokens
    - Context tokens most similar to a target word
    """

    def __init__(
        self,
        model: BertModel,
        tokenizer: BertTokenizer,
        device: torch_device
    ) -> None:
        """
        Initializes the DisambModel with a pre-trained BERT model and tokenizer.

        Args:
            model (BertModel): A pre-trained BERT model from HuggingFace Transformers.
            tokenizer (BertTokenizer): The tokenizer associated with the BERT model.
            device (torch.device): The device (CPU or CUDA) to move the model to.

        Sets up the model for inference by:
            - Enabling output of hidden states.
            - Switching to evaluation mode.
            - Moving the model to the specified device.
        """

        model.config.output_hidden_states = True
        model.eval()

        self.model = model.to(device) # type: ignore
        self.tokenizer = tokenizer
        self.device = device

    @staticmethod
    def _find_subword_span(tokens: List[str], target_tokens: List[str]) -> Tuple[int, int]:
        """
        Finds the start and end indices of a sequence of target subword tokens 
        within a larger list of tokens.

        Args:
            tokens (List[str]): The full list of tokenized subwords (e.g., from a sentence).
            target_tokens (List[str]): The tokenized subwords representing the target word or phrase.

        Returns:
            Tuple[int, int]: A tuple (start_index, end_index) indicating the span of the target tokens.

        Raises:
            ValueError: If the target_tokens sequence is not found in the tokens list.

        Example:
            tokens = ["[CLS]", "I", "love", "em", "##bed", "##ding", "models", "[SEP]"]
            target_tokens = ["em", "##bed", "##ding"]
            Output: (3, 5)
        """
        span_len = len(target_tokens)
        for idx in range(len(tokens) - span_len + 1):
            if tokens[idx : idx + span_len] == target_tokens:
                return idx, idx + span_len - 1
        raise ValueError(f"Target tokens {target_tokens} not found in token list.")

    def forward(self, input_sentence: str, target_word: str) -> torch.Tensor:
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
        input_ids = encodings["input_ids"].to(self.device) # type: ignore
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

    def get_context_words(self, sentence: str, target_word: str, top_k: int = 10) -> List[Tuple[str, float]]:
        marked_text = f"{self.tokenizer.cls_token} {sentence} {self.tokenizer.sep_token}"
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
        input_ids = encodings["input_ids"].to(self.device) # type: ignore
        token_type_ids = torch.ones_like(input_ids).to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids, token_type_ids=token_type_ids)
            hidden_states = outputs.hidden_states

        token_embeddings = []
        seq_len = input_ids.size(1)
        for idx in range(seq_len):
            layers = [hidden_states[-i][0][idx] for i in range(1, 5)]
            token_embeddings.append(torch.stack(layers).sum(dim=0))

        target_vector = token_embeddings[first_start]
        similarities = []
        for idx, token_str in enumerate(tokens):
            if idx in range(first_start, first_end + 1):
                continue
            if token_str in {self.tokenizer.cls_token, self.tokenizer.sep_token}: # type: ignore
                continue
            score = F.cosine_similarity(
                target_vector.unsqueeze(0),
                token_embeddings[idx].unsqueeze(0),
                dim=1
            ).item()
            similarities.append((token_str, score))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

    def get_sentence_embedding(self, sentence: str) -> torch.Tensor:
        """
        Returns a sentence embedding by mean-pooling token embeddings.
        """
        marked_text = f"{self.tokenizer.cls_token} {sentence} {self.tokenizer.sep_token}"
        encodings = self.tokenizer.encode_plus(
            marked_text,
            add_special_tokens=False,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        input_ids = encodings["input_ids"].to(self.device) # type: ignore
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
        """
        Returns a list of (token, vector) for every token in the sentence.
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
        input_ids = encodings["input_ids"].to(self.device) # type: ignore
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

    def visualize(self, sentence: str, method: str = "pca"):
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
    
    def save_to_file(self, sentence: str, path: str, format: str = "pt"):
        """
        Save all token vectors from a sentence to a file.
        :param format: 'pt' (torch), 'npy' (numpy), or 'json'
        """
        token_vecs = self.get_all_token_vectors(sentence)
        if format == "pt":
            torch.save({t: v for t, v in token_vecs}, path)
        elif format == "npy":
            np.save(path, {t: v.cpu().numpy() for t, v in token_vecs}) # type: ignore
        elif format == "json":
            data = {t: v.cpu().tolist() for t, v in token_vecs}
            with open(path, "w") as f:
                json.dump(data, f)
        else:
            raise ValueError("Format must be one of: pt, npy, json")
