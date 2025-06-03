# models/disamb_model.py
import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel
from typing import List, Tuple


class DisambModel:
    """
    A small wrapper around a BERT model to:
      1. Extract a single “target word” embedding from its contextual representation.
      2. Find the top‐k most similar tokens (context words) to that target embedding.

    Expect that the underlying `model` is a HuggingFace transformer with
    `output_hidden_states=True`. The tokenizer should be the corresponding tokenizer.
    """

    def __init__(
        self,
        model: BertModel,
        tokenizer: BertTokenizer,
        device: torch.device
    ) -> None:
        """
        :param model: A `BertModel` (or any HF model that returns hidden_states).
        :param tokenizer: The matching `BertTokenizer`.
        :param device: torch.device ("cuda" or "cpu") to load tensors into.
        """
        # Ensure the model returns hidden states
        model.config.output_hidden_states = True
        model.eval()  # we’re doing inference

        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device

    @staticmethod
    def _find_subword_span(
        tokens: List[str],
        target_tokens: List[str]
    ) -> Tuple[int, int]:
        """
        Find the first continuous span in `tokens` that matches `target_tokens`.
        Returns (start_index, end_index). Raises ValueError if not found.
        """
        span_len = len(target_tokens)
        for idx in range(len(tokens) - span_len + 1):
            if tokens[idx : idx + span_len] == target_tokens:
                return idx, idx + span_len - 1

        raise ValueError(f"Target tokens {target_tokens} not found in token list.")

    # In disamb_model.py
    def forward(self, input_sentence: str, target_word: str) -> torch.Tensor:
        # 1) Tokenize and add special tokens
        marked_text = f"{self.tokenizer.cls_token} {input_sentence} {self.tokenizer.sep_token}"
        tokens = self.tokenizer.tokenize(marked_text)
        target_tokens = self.tokenizer.tokenize(target_word.lower())

        # Log if sentence is too long
        if len(tokens) > 512:
            print(f"Warning: Input sentence exceeds 512 tokens (length={len(tokens)}): {input_sentence[:100]}...")

        # 2) Locate where the target subwords appear
        try:
            tgt_start, tgt_end = self._find_subword_span(tokens, target_tokens)
        except ValueError:
            raise ValueError(f"Target word '{target_word}' not found in sentence.")

        # 3) Convert to IDs, build token_type_ids at once
        encodings = self.tokenizer.encode_plus(
            marked_text,
            add_special_tokens=False,  # already manually added
            max_length=512,  # Ensure truncation
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encodings["input_ids"].to(self.device)
        token_type_ids = torch.ones_like(input_ids).to(self.device)

        # Re-check if target word is still present after truncation
        new_tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        try:
            self._find_subword_span(new_tokens, target_tokens)
        except ValueError:
            raise ValueError(f"Target word '{target_word}' was truncated from sentence.")

        # 4) Forward pass (no gradient needed)
        with torch.no_grad():
            outputs = self.model(input_ids, token_type_ids=token_type_ids)
            hidden_states = outputs.hidden_states

        # 5) For each subword token, sum the last 4 layers
        subword_embs = []
        for idx in range(tgt_start, tgt_end + 1):
            last_four = [hidden_states[-layer_idx][0][idx] for layer_idx in range(1, 5)]
            combined = torch.stack(last_four, dim=0).sum(dim=0)
            subword_embs.append(combined)

        # 6) Average across all subword embeddings
        target_emb = torch.stack(subword_embs, dim=0).mean(dim=0)
        return target_emb

    def get_context_words(
        self,
        sentence: str,
        target_word: str,
        top_k: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Return the top_k tokens (strings) in the sentence (excluding [CLS]/[SEP] and the target itself)
        that are most similar (cosine) to the target word embedding.

        :param sentence: Full sentence containing target_word.
        :param target_word: A single word that appears in `sentence`.
        :param top_k: Number of similar tokens to return.
        :return: A list of (token_str, cosine_similarity_score), sorted descending.
        """
        # 1) Tokenize and add special tokens
        marked_text = f"{self.tokenizer.cls_token} {sentence} {self.tokenizer.sep_token}"
        tokens = self.tokenizer.tokenize(marked_text)
        target_tokens = self.tokenizer.tokenize(target_word.lower())

        # 2) Find all subword spans that match the target (we’ll use only the first occurrence for similarity)
        try:
            first_start, first_end = self._find_subword_span(tokens, target_tokens)
        except ValueError:
            return []

        # 3) Convert to IDs / token_type_ids
        encodings = self.tokenizer.encode_plus(
            marked_text,
            add_special_tokens=False,
            return_tensors="pt",
        )
        input_ids = encodings["input_ids"].to(self.device)            # [1, seq_len]
        token_type_ids = torch.ones_like(input_ids).to(self.device)    # single‐sentence segment IDs

        # 4) Forward pass and collect all token embeddings
        with torch.no_grad():
            outputs = self.model(
                input_ids,
                token_type_ids=token_type_ids
            )
            hidden_states = outputs.hidden_states

        # Build a list of token embeddings (averaged last 4 layers) for every token index
        token_embeddings = []
        seq_len = input_ids.size(1)
        for idx in range(seq_len):
            layers_to_sum = [
                hidden_states[-layer_idx][0][idx]
                for layer_idx in range(1, 5)
            ]
            token_embeddings.append(torch.stack(layers_to_sum, dim=0).sum(dim=0))

        # 5) Get the “target vector” (first occurrence)
        target_vector = token_embeddings[first_start]

        # 6) Compute cosine similarities (skipping [CLS]/[SEP] and any subword indices of target)
        similarities: List[Tuple[str, float]] = []
        for idx, token_str in enumerate(tokens):
            if idx in range(first_start, first_end + 1):
                continue
            if token_str in {self.tokenizer.cls_token, self.tokenizer.sep_token}:
                continue

            sim_score = F.cosine_similarity(
                target_vector.unsqueeze(0),
                token_embeddings[idx].unsqueeze(0),
                dim=1
            ).item()
            similarities.append((token_str, sim_score))

        # 7) Sort descending by similarity, return top_k
        similarities.sort(key=lambda pair: pair[1], reverse=True)
        return similarities[:top_k]
