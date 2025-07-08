"""
sentence_processing.py

Utilities for extracting sentences from a block of text that contain a specific target word,
while ensuring compatibility with BERT tokenizer input limits. Sentences are cleaned,
filtered, and truncated if necessary to fit within token length constraints.

"""

import re
from functools import lru_cache
from nltk.tokenize import sent_tokenize
from transformers import BertTokenizer


# TODO: Update this to use global app.state waala tokenizer
@lru_cache(maxsize=1)
def _get_tokenizer():
    """
    Load and cache the BERT tokenizer (bert-base-uncased) for efficient reuse.

    Returns:
        BertTokenizer: Tokenizer instance for BERT.
    """
    return BertTokenizer.from_pretrained("bert-base-uncased")


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


def _filter_sentences(sentences_list: list[str]) -> list[str]:
    """
    Applies text standardization to a list of sentences.

    Each sentence is cleaned using the `_standardize_text` function
    and returned in the same order.

    Args:
        sentences_list (list[str]): A list of raw sentence strings.

    Returns:
        list[str]: A list of standardized (cleaned) sentences.
    """
    return [_standardize_text(sentence) for sentence in sentences_list]


def get_sentences_with_target_word(
    text_content: str,
    target_word: str,
    frequency_limit: int = 100,
    max_length: int = 510
) -> list[tuple[str, str]]:
    tokenizer = _get_tokenizer()
    text = " ".join(text_content.split()).replace("\n", " ").replace("\r", " ")
    sentences = sent_tokenize(text)
    target_sentences = []

    for sent in sentences:
        if re.search(rf'\b{re.escape(target_word.lower())}\b', sent.lower()):
            tokens = tokenizer.tokenize(f"{tokenizer.cls_token} {sent} {tokenizer.sep_token}")
            if len(tokens) <= max_length:
                target_sentences.append((sent, _standardize_text(sent)))
            else:
                words = sent.split()
                target_idx = next((i for i, w in enumerate(words) if w.lower() == target_word.lower()), None)
                if target_idx is not None:
                    window_size = (max_length - 10) // 2
                    start = max(0, target_idx - window_size)
                    end = min(len(words), target_idx + window_size + 1)
                    truncated_sent = " ".join(words[start:end])
                    tokens = tokenizer.tokenize(f"{tokenizer.cls_token} {truncated_sent} {tokenizer.sep_token}")
                    if len(tokens) <= max_length and target_word.lower() in truncated_sent.lower():
                        target_sentences.append((truncated_sent, _standardize_text(truncated_sent)))

    return target_sentences[:frequency_limit]
